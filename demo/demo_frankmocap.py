# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle

############# input parameters  #############
from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer
from datetime import datetime

from bodymocap.body_bbox_detector import BodyPoseEstimator
from handmocap.hand_bbox_detector import HandBboxDetector
from integration.copy_and_paste import integration_copy_paste

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
from rich.console import Console

CONSOLE = Console()

HAND_BBOX_LIST = {'left_hand': [], 'right_hand': []}
HAND_BBOX_CHANGE_RATE_X = {'left_hand': [], 'right_hand': []}
HAND_BBOX_CHANGE_RATE_Y = {'left_hand': [], 'right_hand': []}
BODY_BBOX_LIST = list()
BODY_BBOX_CHANGE_RATE = list()


def __interpolate_hand_bb(hand_bbox_list, out_dir, img_path, interpolation_count,
                          stabilize=False, weight_deaden={'past': 1, 'future': 0.5}):
    """Interpolates using frames from past and future if bboxes are saved accordingly.
       Uses weights to weigh closer frames heavier.

       Can also interpolate for existing frames if `stabilize=True`
       `weight_deaden` is used to further deaden the influence of frames.
       1 means nothing happens. """
    path = osp.join(out_dir, 'bbox')

    if not osp.exists(path):
        CONSOLE.print('No file with bounding boxes found. Generate BBoxes first (look at demo_options).', style='bold red')
        return
    else:
        CONSOLE.print(f'Processing {img_path}', style='green')
        img_id = int(osp.split(img_path)[-1][:5])

        # if values of both hands are true, there is something wrong with the model's output
        if np.array_equal(hand_bbox_list[0]['left_hand'], hand_bbox_list[0]['right_hand']):
            stabilize = True

        for hand in ['left_hand', 'right_hand']:
            if img_id < interpolation_count:
                continue
            try:
                hand_bbox = hand_bbox_list[0][hand]
            except TypeError:
                # no hand detected
                hand_bbox = None

            if hand_bbox is not None and not stabilize:
                continue

            if hand_bbox is None:
                hand_bbox = [0, 0, 0, 0]

            count = 0
            for id in range(img_id-interpolation_count, img_id+interpolation_count+1):
                count += 1
                try:
                    saved_bb = json.load(open(osp.join(path, f'{id:05}_bbox.json')))
                except FileNotFoundError:
                    # bbox out of bounds (img_interpolation_count is too much)
                    continue
                try:
                    saved_hand_bbox = saved_bb['hand_bbox_list'][0][hand]
                except IndexError:
                    # no hand bounding box inf exists
                    continue

                if saved_hand_bbox is not None:
                    if hand_bbox[0] == 0 and hand_bbox[1] == 0 and hand_bbox[2] == 0 and hand_bbox[3] == 0:
                        # initialized with first frame
                        hand_bbox = saved_hand_bbox
                    else:
                        # past frames, weight normal
                        weight = (count % (interpolation_count + 1)) / (interpolation_count + 1)
                        # for future frames weight reverses
                        if count > interpolation_count:
                            weight = (1 - weight) * weight_deaden['future']
                        else:
                            weight *= weight_deaden['past']

                        hand_bbox = [hand_bbox[j] * (1-weight)  + saved_hand_bbox[j] * weight for j in range(4)]

            if hand_bbox == [0, 0, 0, 0]:
                # stuff like this is needed to be consistent with existing data structures
                hand_bbox = None
            else:
                hand_bbox = np.array(hand_bbox)
                CONSOLE.print(f'Interpolated {hand}: {hand_bbox}', style='yellow')
            hand_bbox_list[0][hand] = hand_bbox

    return hand_bbox_list


def __stabilize_hand_bbox(hand_bbox_list, out_dir, img_path, interpolation_count,
                          x_w_thresh = 0.2, y_h_thresh = 1, look_back=5):
    """Stabilize hand bounding box if changes are sudden and drastic
       from frame to frame.

       Threshold: If the X coordinate and width change by more than `x_w_thresh`
       and `look_back` frames do not exceed this threshold, then replace current
       bbox by the one immediately preceeding it.

       `x_w_thresh` and `look_back` should be more fine grained than body bboxes"""
    CONSOLE.print(hand_bbox_list, style='green')
    global HAND_BBOX_LIST, HAND_BBOX_CHANGE_RATE_X, HAND_BBOX_CHANGE_RATE_Y

    for hand in ['left_hand', 'right_hand']:
        bbox = hand_bbox_list[0][hand]
        if bbox is None:
            continue
        count = len(HAND_BBOX_LIST[hand])
        count_rate = len(HAND_BBOX_CHANGE_RATE_X[hand])

        HAND_BBOX_LIST[hand].append(bbox)
        # not enough boxes to look back to
        if count < look_back:
            return hand_bbox_list

        bbox_prev = HAND_BBOX_LIST[hand][count-1]
        if bbox_prev is not None:
            change_x = abs(bbox[0] - bbox_prev[0]) / bbox[0]
            change_w = abs(bbox[2] - bbox_prev[2]) / bbox[2]
            change_y = abs(bbox[1] - bbox_prev[1]) / bbox[1]
            change_h = abs(bbox[3] - bbox_prev[3]) / bbox[3]
            HAND_BBOX_CHANGE_RATE_X[hand].append((change_x, change_w))
            HAND_BBOX_CHANGE_RATE_Y[hand].append((change_y, change_h))
            CONSOLE.print(f'{hand} -> Change X: {change_x}; Change W: {change_w} Change Y: {change_y}; Change H: {change_h}', style='yellow')

            if change_x > x_w_thresh and change_w > x_w_thresh:
                # if the past changes were relatively stable, it means current
                # bbox prediction has had a sudden large change
                avg_past_change_x, avg_past_change_w = 0, 0
                for i in range(look_back):
                    avg_past_change_x += HAND_BBOX_CHANGE_RATE_X[hand][count_rate-1-i][0]
                    avg_past_change_w += HAND_BBOX_CHANGE_RATE_X[hand][count_rate-1-i][1]

                if (avg_past_change_x / look_back < x_w_thresh) and (avg_past_change_w / look_back < x_w_thresh):
                    HAND_BBOX_CHANGE_RATE_X[hand].pop()
                    HAND_BBOX_CHANGE_RATE_Y[hand].pop()
                    HAND_BBOX_LIST[hand].pop()
                    hand_bbox_list[0][hand] = HAND_BBOX_LIST[hand][count - 1]
                    HAND_BBOX_LIST[hand].append(HAND_BBOX_LIST[hand][count - 1])
                    HAND_BBOX_CHANGE_RATE_X[hand].append(HAND_BBOX_CHANGE_RATE_X[hand][count_rate-1])
                    HAND_BBOX_CHANGE_RATE_Y[hand].append(HAND_BBOX_CHANGE_RATE_Y[hand][count_rate-1])

                    CONSOLE.print(f'Replaced existing {hand} bbox with previous one', style='yellow')
                    # interpolate
                    # hand_bbox_list = __interpolate_hand_bb(hand_bbox_list, out_dir, img_path, interpolation_count, True)
                    # HAND_BBOX_LIST[hand].append(hand_bbox_list[0][hand])
                    # HAND_BBOX_CHANGE_RATE[hand].append((
                    #     abs(hand_bbox_list[0][hand][0] - bbox_prev[0]) / hand_bbox_list[0][hand][0],
                    #     abs(hand_bbox_list[0][hand][2] - hand_bbox_list[0][hand][2]) / hand_bbox_list[0][hand][2]))
            elif change_y > y_h_thresh and change_h > y_h_thresh:
                avg_past_change_y, avg_past_change_h = 0, 0
                for i in range(look_back):
                    avg_past_change_y += HAND_BBOX_CHANGE_RATE_Y[hand][count_rate-1-i][0]
                    avg_past_change_h += HAND_BBOX_CHANGE_RATE_Y[hand][count_rate-1-i][1]

                if (avg_past_change_y / look_back < y_h_thresh) and (avg_past_change_h / look_back < y_h_thresh):
                    HAND_BBOX_CHANGE_RATE_X[hand].pop()
                    HAND_BBOX_CHANGE_RATE_Y[hand].pop()
                    HAND_BBOX_LIST[hand].pop()
                    hand_bbox_list[0][hand] = HAND_BBOX_LIST[hand][count - 1]
                    HAND_BBOX_LIST[hand].append(HAND_BBOX_LIST[hand][count - 1])
                    HAND_BBOX_CHANGE_RATE_X[hand].append(HAND_BBOX_CHANGE_RATE_X[hand][count_rate-1])
                    HAND_BBOX_CHANGE_RATE_Y[hand].append(HAND_BBOX_CHANGE_RATE_Y[hand][count_rate-1])

                    CONSOLE.print(f'Replaced existing {hand} bbox with previous one', style='yellow')

    return hand_bbox_list


def __stabilize_body_bbox(body_bbox_list, x_w_thresh = 0.20, look_back = 5):
    """Stabilize body bounding box if changes are sudden and drastic
       from frame to frame.

       Threshold: If the X coordinate and width change by more than `x_w_thresh`
       and `look_back` frames do not exceed this threshold, then replace current
       bbox by the one immediately preceeding it."""
    global BODY_BBOX_LIST, BODY_BBOX_CHANGE_RATE
    bb = body_bbox_list[0]
    count = len(BODY_BBOX_LIST)
    count_rate = len(BODY_BBOX_CHANGE_RATE)
    BODY_BBOX_LIST.append(bb)
    # not enough boxes to look back to
    if count < look_back:
        return body_bbox_list

    change_x = abs(bb[0] - BODY_BBOX_LIST[count-1][0]) / bb[0]
    change_w = abs(bb[2] - BODY_BBOX_LIST[count-1][2]) / bb[2]
    BODY_BBOX_CHANGE_RATE.append((change_x, change_w))
    CONSOLE.print('Body BBox rate changes', style='green')
    CONSOLE.print(f'Change in X: {change_x}', style='green')
    CONSOLE.print(f'Change in W: {change_w}', style='green')

    if change_x > x_w_thresh and change_w > x_w_thresh:
        # if the past changes were relatively stable, it means current
        # bbox prediction had a sudden large change
        avg_past_change_x, avg_past_change_w = 0, 0
        for i in range(look_back):
            avg_past_change_x += BODY_BBOX_CHANGE_RATE[count_rate-1-i][0]
            avg_past_change_w += BODY_BBOX_CHANGE_RATE[count_rate-1-i][1]

        if avg_past_change_x / look_back < x_w_thresh and avg_past_change_w / look_back < x_w_thresh:
            CONSOLE.print('Replaced body bbox with previous one', style='yellow')
            BODY_BBOX_CHANGE_RATE.pop()
            BODY_BBOX_CHANGE_RATE.append(BODY_BBOX_CHANGE_RATE[count_rate - 1])
            BODY_BBOX_LIST.pop()
            BODY_BBOX_LIST.append(BODY_BBOX_LIST[count - 1])
            return [np.array([el for el in BODY_BBOX_LIST[count - 1]])]
    return body_bbox_list



def __enlarge_body_bbox(body_bbox_list, rate=0.05):
    """Enlarge body bounding box.
       Scale the `w` and `h` components by `rate`
       Adjust the rectange origin (x, y) by substracting half the scale difference"""
    new_bbox_list = list()
    bb = body_bbox_list[0]
    CONSOLE.print(f'Body BBox Before: {bb}', style='yellow')

    # adjust (x, y)
    new_bb = [int(bb[0] - (bb[0] * rate / 2)), int(bb[1] - (bb[1] * rate / 2))]

    # scale (w, h)
    new_bb.append(int(bb[2] + bb[2] * rate))
    new_bb.append(int(bb[3] + bb[3] * rate))

    CONSOLE.print(f'Body BBox After: {new_bb}', style='green')
    new_bbox_list.append(np.array(new_bb))

    return new_bbox_list


def __filter_bbox_list(body_bbox_list, hand_bbox_list, single_person):
    # (to make the order as consistent as possible without tracking)
    bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
    hand_bbox_list = [hand_bbox_list[i] for i in idx_big2small]

    if single_person and len(body_bbox_list)>0:
        body_bbox_list = [body_bbox_list[0], ]
        hand_bbox_list = [hand_bbox_list[0], ]

    return body_bbox_list, hand_bbox_list


def run_regress(
    args, img_original_bgr,
    body_bbox_list, hand_bbox_list, bbox_detector,
    body_mocap, hand_mocap, image_path
):
    cond1 = len(body_bbox_list) > 0 and len(hand_bbox_list) > 0
    cond2 = not args.frankmocap_fast_mode

    # use pre-computed bbox or use slow detection mode
    if cond1 or cond2:
        if not cond1 and cond2:
            # run detection only when bbox is not available
            body_pose_list, body_bbox_list, hand_bbox_list, _ = \
                bbox_detector.detect_hand_bbox(img_original_bgr.copy())
        else:
            print("Use pre-computed bounding boxes")
        assert len(body_bbox_list) == len(hand_bbox_list)

        if len(body_bbox_list) < 1:
            return list(), list(), list()

        # sort the bbox using bbox size
        # only keep on bbox if args.single_person is set
        body_bbox_list, hand_bbox_list = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person)

        if args.enlarge_body_bbox:
            body_bbox_list = __enlarge_body_bbox(body_bbox_list)
        if args.stabilize_body_bbox:
            body_bbox_list = __stabilize_body_bbox(body_bbox_list)
        if args.interpolate_hand_bb:
            hand_bbox_list = __interpolate_hand_bb(
                hand_bbox_list, args.out_dir, image_path, args.interpolate_hand_bb_count)
        if args.stabilize_hand_bbox:
            hand_bbox_list = __stabilize_hand_bbox(
                hand_bbox_list, args.out_dir, image_path, args.interpolate_hand_bb_count)

        # hand & body pose regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True)
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(hand_bbox_list) == len(pred_hand_list)
        assert len(pred_hand_list) == len(pred_body_list)

    else:
        _, body_bbox_list = bbox_detector.detect_body_bbox(img_original_bgr.copy())

        if len(body_bbox_list) < 1: 
            return list(), list(), list()

        # sort the bbox using bbox size 
        # only keep on bbox if args.single_person is set
        hand_bbox_list = [None, ] * len(body_bbox_list)
        body_bbox_list, _ = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person)

        # body regression first 
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_body_list)

        # get hand bbox from body
        hand_bbox_list = body_mocap.get_hand_bboxes(pred_body_list, img_original_bgr.shape[:2])
        assert len(pred_body_list) == len(hand_bbox_list)

        # hand regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(pred_hand_list) 

    # integration by copy-and-paste
    integral_output_list = integration_copy_paste(
        pred_body_list, pred_hand_list, body_mocap.smpl, img_original_bgr.shape)
    
    return body_bbox_list, hand_bbox_list, integral_output_list


def run_frank_mocap(args, bbox_detector, body_mocap, hand_mocap, visualizer):
    #Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)

    cur_frame = args.start_frame
    video_frame = 0
    while True:
        # load data
        load_bbox = False

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
          # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)

        elif input_type == 'webcam':
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break
        print("--------------------------------------")

        # bbox detection
        if not load_bbox:
            body_bbox_list, hand_bbox_list = list(), list()

        # regression (includes integration)
        body_bbox_list, hand_bbox_list, pred_output_list = run_regress(
            args, img_original_bgr, 
            body_bbox_list, hand_bbox_list, bbox_detector,
            body_mocap, hand_mocap, image_path)

        # save the obtained body & hand bbox to json file
        if args.save_bbox_output: 
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        if len(body_bbox_list) < 1: 
            print(f"No body deteced: {image_path}")
            continue

        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualization
        res_img = visualizer.visualize(
            img_original_bgr,
            pred_mesh_list = pred_mesh_list,
            body_bbox_list = body_bbox_list,
            hand_bbox_list = hand_bbox_list)

       # show result in the screen
        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            ImShow(res_img)

        # save result image
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'frank'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        print(f"Processed : {image_path}")

    # save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()

def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    hand_bbox_detector =  HandBboxDetector('third_view', device)

    #Set Mocap regressor
    body_mocap = BodyMocap(args.checkpoint_body_smplx, args.smpl_dir, device = device, use_smplx= True)
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    run_frank_mocap(args, hand_bbox_detector, body_mocap, hand_mocap, visualizer)


if __name__ == '__main__':
    main()