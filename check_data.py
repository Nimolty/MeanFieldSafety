from numpy import random
import pybullet as p
import pybullet_data
from ipdb import set_trace
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time
import random
import math
import copy
import pickle
import cv2
import numpy as np
from tqdm import tqdm, trange
from collections import deque
import argparse
import functools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


from torch_geometric.nn import knn_graph
from torch_geometric.nn import radius_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, EdgeConv
from torch_scatter import scatter_max, scatter_mean

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Envs.RLEnvSortingBall import RLSorting
from utils import exists_or_mkdir, string2bool, images_to_video, save_video
from Algorithms.SDE import ScoreModelGNN, ScoreModelGNNMini, MiniUpdate, \
    marginal_prob_std, diffusion_coeff, loss_fn, \
    Euler_Maruyama_sampler, pc_sampler, ode_sampler, loss_fn_cond

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def snapshot(env, file_name):
    img = env.render(256)
    cv2.imwrite(file_name, img)

def collect_data(num_samples, num_boxes, wall_bound, max_vel, time_freq = 240, is_gui=False, normalize = False, suffix=''):
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

class GraphDataset:
    def __init__(self, items, radius):
        self.items = items
        self.radius = radius

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        cur_data = self.items[item] # [max_num, 2]
        num_nodes = cur_data.shape[0]
        # # then random mask
        augmented_num_nodes = random.randint(8, num_nodes)
        indices = np.random.choice(a=num_nodes, size=augmented_num_nodes, replace=False)
        augmented_data = cur_data[indices]
        assert augmented_data.shape[0] == augmented_num_nodes
        edge_index = radius_graph(augmented_data, r=self.radius)

        return Data(x=augmented_data, edge_index=edge_index)


def visualize_states(eval_states, env, logger, nrow, suffix):
    # states -> images
    # if scores, then try to visualize gradient
    imgs = []
    for box_state in eval_states:
        env.set_state(box_state)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    logger.add_image(f'Images/dynamic_{suffix}', grid, epoch)


def save_video(env, states, save_path, simulation=False, fps = 50, render_size = 256, suffix='avi'):
    imgs = []
    for _, state in tqdm(enumerate(states), desc='Saving video'):
        env.set_state(state, simulation)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    images_to_video(save_path+f'.{suffix}', batch_imgs, fps, (render_size, render_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--n_box', type=int, default=12)
    parser.add_argument('--knn', type=int, default=5)
    parser.add_argument('--r', type=float, default=0.2)
    parser.add_argument('--wall_bound', type=float, default=0.3)
    parser.add_argument('--n_epoches', type=int, default=1000)
    parser.add_argument('--n_samples', type=int, default=1e5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--tscale', type=float, default=1.)
    parser.add_argument('--sigma', type=float, default=25.)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--render_size', type=int, default=256)
    parser.add_argument('--max_vel', type=float, default=0.3)
    parser.add_argument('--time_freq', type=int, default=240)
    parser.add_argument('--is_gui', type=bool, default=False)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--arch', type=str, choices=['old_model', 'Mini', 'MiniUpdate', 'MiniDouble'], default='old_model')


    # load args
    args = parser.parse_args()
    n_box = args.n_box
    wall_bound = args.wall_bound
    n_samples = args.n_samples
    batch_size = args.batch_size
    workers = args.workers
    lr = args.lr
    beta1 = args.beta1
    render_size = args.render_size
    max_vel = args.max_vel
    time_freq = args.time_freq
    is_gui = args.is_gui
    normalize = args.normalize
    hidden_dim = args.hidden_dim
    embed_dim = args.embed_dim
    arch = args.arch

    dataset_path = f'./dataset/Sorting_SDE_Support_n1e5_ball1x10_bound0.2.pth'

    # create log path
    exists_or_mkdir('./logs')
    ckpt_path = f'./logs/{args.exp_name}/'
    exists_or_mkdir(ckpt_path)
    eval_path = f'./logs/{args.exp_name}/test_batch/'
    exists_or_mkdir(eval_path)
    tb_path = f'./logs/{args.exp_name}/tb'
    exists_or_mkdir(tb_path)

    # init writer
    writer = SummaryWriter(tb_path)
    if os.path.exists(dataset_path):
        print('### found existing dataset ###')
        with open(dataset_path, 'rb') as f:
            data_samples = pickle.load(f)
        dataset = torch.tensor(data_samples)
        # set_trace()
        dataset = dataset[: int(n_samples)]
    else:
        print('### not found existing dataset, start collecting data ###')
        ts = time.time()

        data_samples = collect_data(n_samples, n_box, wall_bound, max_vel, time_freq, is_gui, normalize)
        with open(dataset_path, 'wb') as f:
            pickle.dump(data_samples, f)
        dataset = torch.tensor(data_samples)
        print('### data collection done! takes {:.2f} to collect {} samples ###'.format(time.time() - ts, n_samples))

    print(f'Dataset Size: {len(dataset)}')
    # dataset = dataset.reshape(-1, dataset.shape[-1])
    # set_trace() # [-0.176, 0.1777]
    # 这里要根据normalize 处理下dataset!
    if args.normalize:
        dataset /= (args.wall_bound - 0.025)
    set_trace()
    # dataset.mean() = tensor(-4.8699e-05)
    # dataset.mean(0) =
    #         [-5.9267e-04, -2.1678e-04, -7.6259e-05,  1.4428e-05,  3.9765e-04,
    #          2.0899e-05, -1.9335e-05,  6.1818e-05, -1.9156e-04,  3.6431e-05,
    #         -1.9802e-04,  3.4872e-05,  1.1651e-04,  6.1469e-05, -1.2354e-04,
    #         -5.2700e-04,  9.7050e-05,  4.6578e-04, -3.6587e-04,  3.0162e-05]
    # dataset.mean() / dataset.max() = -0.0003
    #
