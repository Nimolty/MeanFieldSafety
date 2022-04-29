from tqdm import tqdm
from ipdb import set_trace
import pickle
import numpy as np
import time
import argparse
import sys
import os
import cv2
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Envs.RLEnvSortingBall import RLSorting
from utils import exists_or_mkdir

def snapshot(env, file_name):
    img = env.render(256)
    cv2.imwrite(file_name, img)

def collect_data(num_samples, num_boxes, wall_bound, max_vel, time_freq = 240, is_gui=False, normalize = True, suffix=''):
    env_kwargs = {
        'n_boxes': num_boxes,
        'wall_bound':wall_bound,
        'max_action': max_vel,
        'time_freq': time_freq,
        'is_gui': is_gui,
        'normalize': normalize,
    }
    env = RLSorting(**env_kwargs)

    exists_or_mkdir('./dataset/')
    debug_path = f'./dataset/sorting_{suffix}/'
    exists_or_mkdir(debug_path)
    samples = []

    # in the debug folder "./dataset", only
    with tqdm(total=num_samples) as pbar:
        while len(samples) < num_samples:
            cur_state = env.reset()
            samples.append(cur_state)
            snapshot(env, f'./dataset/debug.png')
            pbar.update(1)
    env.close()
    samples = np.stack(samples, axis=0)
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1e5)
    parser.add_argument('--n_box', type=int, default=12)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--max_vel', type=float, default=0.3)
    parser.add_argument('--time_freq', type=int, default=240)
    parser.add_argument('--is_gui', type=bool, default=False)
    parser.add_argument('--wall_bound', type=float, default=0.3)
    parser.add_argument('--normalize', type=bool, default=False)

    args = parser.parse_args()

    n_box = args.n_box
    wall_bound = args.wall_bound
    n_samples = args.n_samples
    max_vel = args.max_vel
    time_freq = args.time_freq
    is_gui = args.is_gui
    normalize = args.normalize
    dataset_path = f'./dataset/{args.data_name}.pth'

    ts = time.time()
    data_samples = collect_data(n_samples, n_box, wall_bound, max_vel, time_freq, is_gui, normalize)
    with open(dataset_path, 'wb') as f:
        pickle.dump(data_samples, f)
    print('### data collection done! takes {:.2f} to collect {} samples ###'.format(time.time() - ts, n_samples))

