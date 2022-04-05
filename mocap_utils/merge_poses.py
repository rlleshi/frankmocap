import glob
import os.path as osp
import pickle
import numpy as np
import moviepy.editor as mpy

from rich.console import Console
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path


POSE_EXT = '.npy'
CONSOLE = Console()


def parse_args():
    parser = ArgumentParser(
        prog='merge all single-video poses according to MMAction2')
    parser.add_argument('src_dir', help='pose directory with frankmocap results')
    parser.add_argument('video_dir', help='source video directory')
    parser.add_argument('ann', help='annotation file')
    parser.add_argument('--out-dir',
                        default='mocap_output/',
                        help='out video directory')
    parser.add_argument('--level-src',
                        type=int,
                        default=2,
                        choices=[1, 2, 3],
                        help='directory level to find poses')
    parser.add_argument('--level-video',
                        type=int,
                        default=2,
                        choices=[1, 2, 3],
                        help='directory level to find videos')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def get_video(videos, id):
    """Get the video based on its id.
    """
    return [video
           for video in videos if osp.splitext(osp.basename(video))[0] == id]


def get_label(labels, id):
    """Get the label based on the annotation .txt file and video id
    """
    return [label.split(' ')[1]
           for label in labels if osp.splitext(osp.basename(label))[0] == id]


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    items = glob.glob(args.src_dir + '/*' * args.level_src)
    poses = [item for item in items if item.endswith(POSE_EXT)]
    videos = glob.glob(args.video_dir + '/*' * args.level_video)
    with open(args.ann, 'r') as a:
        labels = a.readlines()
    labels = [label.strip() for label in labels]

    default_img_shape = (456, 256)
    result = []

    for pose in tqdm(poses):
        content = np.load(pose)
        id = osp.splitext(osp.basename(pose))[0]

        video = get_video(videos, id)
        if not video:
            img_shape = default_img_shape
        else:
            img_shape = mpy.VideoFileClip(video[0]).size

        label = get_label(labels, id)
        if not label:
            # CONSOLE.print(f'ID {id} does not exist @{args.ann}')
            continue
        else:
            label = int(label[0])

        content = np.swapaxes(content, 0, 1)
        if content.shape[1] <= 10:
            CONSOLE.print(f'@{pose} has less than 10 frames. Skipping...',
                          style='yellow')
            continue

        new_content = {
            'total_frames': content.shape[1],
            'keypoint': content,
            'frame_dir': id,
            'img_shape': img_shape,
            'original_shape': img_shape,
            'label': label
        }
        result.append(new_content)

    out_file = osp.splitext(osp.basename(args.ann))[0] + '.pkl'
    with open(out_file, 'wb') as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
