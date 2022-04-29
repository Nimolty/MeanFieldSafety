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
        augmented_num_nodes = random.randint(num_nodes, num_nodes)
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

    dataset_path = f'./dataset/{args.data_name}.pth'

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
        dataset = dataset[: n_samples]
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
    
    dataset = GraphDataset(dataset.reshape(-1, n_box, 2), args.r)
    a = dataset[0]

    # convert dataset
    # knn = args.knn
    r = args.r if args.normalize else args.r * args.wall_bound
    test_r = 0.4
    # edge = knn_graph(dataset[0].reshape(n_box, 2), knn)
    # edge = radius_graph(dataset[0].reshape(n_box, 2), r)

    # dataset = list(map(lambda x: Data(x=x, edge_index=edge), dataset.reshape(dataset.shape[0], n_box, 2)))
    # MMD！这里改了 Radius Graph 之后忘记改graph要针对每个数据了！
    # print('Start making dataset')
    # dataset = list(map(lambda x: Data(x=x, edge_index=radius_graph(x.reshape(n_box, 2), r)), dataset.reshape(dataset.shape[0], n_box, 2)))
    # print('Dataset prepared!')

    # # convert samples to dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    # init test env and visualize a real batch
    env_kwargs = {
        'n_boxes': n_box,
        'wall_bound': wall_bound,
        'max_action': max_vel,
        'time_freq': time_freq,
        'is_gui': is_gui,
        'normalize': normalize,
    }
    test_env = RLSorting(**env_kwargs)
    test_env.reset()

    # init SDE-related params
    sigma = args.sigma  # @param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    # create models, optimizers, and loss
    if arch == 'old_model':
        score = ScoreModelGNN(marginal_prob_std_fn, n_box, hidden_dim, embed_dim) # old model
    elif arch == 'Mini':
        score = ScoreModelGNNMini(marginal_prob_std_fn, n_box, hidden_dim, embed_dim)
    elif arch == 'MiniUpdate':
        score = MiniUpdate(marginal_prob_std_fn, n_box, hidden_dim, embed_dim)
    elif arch == 'MiniDouble':
        score = MiniDouble(marginal_prob_std_fn, n_box, hidden_dim, embed_dim)
    score.to(device)

    optimizer = optim.Adam(score.parameters(), lr=lr, betas=(beta1, 0.999))

    num_epochs = args.n_epoches
    print("Starting Training Loop...")
    for epoch in trange(num_epochs):
        # For each batch in the dataloader
        for i, real_data in enumerate(dataloader):
            # augment data first
            # set_trace()
            real_data = real_data.to(device)

            # calc score-matching loss
            # loss = loss_fn(score, real_data, marginal_prob_std_fn, scale=args.tscale) # t \in [0, scale]
            set_trace()
            loss = loss_fn_cond(score, real_data, marginal_prob_std_fn, scale=args.tscale) # t \in [0, scale]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add writer
            writer.add_scalar('train_loss', loss, i + epoch*len(dataloader))
        # start eval
        with torch.no_grad():
            if (epoch+1) % 20 == 0:
                test_num = 16
                # use different suffix to identify each sampler's results
                # t0 = 0.01，前面t从1～0.01的过程没意义
                # in_process_sample, res = ode_sampler(score, marginal_prob_std_fn, diffusion_coeff_fn, num_nodes=n_box,
                #                                      knn=knn, t0=0.01, batch_size=test_num, num_steps=500)
                # in_process_sample, res = ode_sampler(score, marginal_prob_std_fn, diffusion_coeff_fn, bound=1 if args.normalize else args.wall_bound, num_nodes=n_box,
                #                                     r=r, t0=args.tscale, batch_size=test_num, num_steps=250)
                in_process_sample, res = ode_sampler(score, marginal_prob_std_fn, diffusion_coeff_fn, bound=1 if args.normalize else args.wall_bound, num_nodes=n_box,
                                                    r=test_r, t0=args.tscale, batch_size=test_num, num_steps=250)
                visualize_states(res.view(test_num, -1), test_env, writer, 4,  suffix='Neural_ODE_sampler|Final Sample')
                # # 检查训练数据
                # visualize_states(real_data.x.view(-1, n_box*2)[0:test_num], test_env, writer, 4,  suffix='Training Data')

            if (epoch + 1) % 100 == 0 and epoch > 40:
                # visualize the proce
                save_video(test_env, in_process_sample.cpu().numpy(), save_path=eval_path + f'{epoch+1}', suffix='mp4')

                # save model
                torch.save(score.cpu().state_dict(), ckpt_path + f'score.pt')
                score.to(device)

    test_env.close()

