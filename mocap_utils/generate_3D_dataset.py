import glob
import os
import os.path as osp
from argparse import ArgumentParser
from itertools import repeat
from multiprocessing import cpu_count
from pathlib import Path
import subprocess
import pickle
import shutil

import numpy as np
from rich.console import Console
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


CONSOLE = Console()

VIDEO_EXTS = ['mp4']
SCRIPT = 'demo.demo_frankmocap'


def cleanup(path):
    """Delete the image frames of the mocap dir
    """
    to_delete = osp.join(path, 'rendered')
    if osp.isdir(to_delete):
        shutil.rmtree(to_delete)


def merge_pkl(path, out):
    """ Merge all pkl files from frankmocap and store them in a single numpy file.

    Args:
        path ([str]): [path to mocap dir that stores the pkl files]
        out ([str]): video name
    """
    if not osp.isdir(path):
        CONSOLE.print(f'Corrupted video: {out}', style='red')
        return

    result = []
    mocap_dir = osp.join(path, 'mocap')
    pkls = sorted(os.listdir(mocap_dir), key=lambda x: int(x[:5]))

    for pkl in tqdm(pkls):
        with open(osp.join(mocap_dir, pkl), 'rb') as f:
            content = pickle.load(f)

        if content['pred_output_list'][0] is not None:
            result.append(list(map(abs,
                content['pred_output_list'][0]['pred_body_joints_img'][:25])))

    np.save(osp.join(path, f'{out}.npy'), np.expand_dims(np.array(result), 1))


def parse_args():
    parser = ArgumentParser(prog='generate 3D Poses dataset with frankmocap')
    parser.add_argument('--src-dir',
                        default='data/',
                        help='source video directory')
    parser.add_argument('--out-dir',
                        default='frankmocap/mocap_output/',
                        help='out video directory')
    parser.add_argument('--num-processes',  # TODO: ggf.
                        type=int,
                        default=(cpu_count() - 2 or 1),
                        help='number of processes used')
    parser.add_argument('--level',
                        type=int,
                        default=1,
                        choices=[1, 2, 3],
                        help='directory level to find videos')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def extract_3d_pose(items):
    video, args = items
    video_name = osp.splitext(osp.basename(video))[0]
    out_dir = osp.join(args.out_dir, video_name)
    CONSOLE.print(f'Processing {video}', style='bold green')
    if osp.exists(out_dir):
        CONSOLE.print(f'{video} has already been processed. Skipping...', style='yellow')
        return

    subargs = [
        'python',
        '-m',
        SCRIPT,
        '--input_path',
        video,
        '--out_dir',
        out_dir,
        '--save_pred_pkl',
        '--single_person',
        '--no_display'
        # '--no_video_out'
    ]
    result = subprocess.run(subargs, capture_output=True)
    error = result.stderr.decode('utf-8')
    if error:
        CONSOLE.print(error, style='red')

    merge_pkl(out_dir, video_name)
    cleanup(out_dir)


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    items = glob.glob(args.src_dir + '/*' * args.level)

    videos = [
        item for item in items if any(
            item.endswith(ext) for ext in VIDEO_EXTS)
    ]

    process_map(extract_3d_pose,
                zip(videos, repeat(args)),
                max_workers=1,
                total=len(videos))


if __name__ == '__main__':
    main()
