import os
from re import I
import time
import math
import copy
import pickle
import argparse
import functools
from collections import deque

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from scipy import integrate
from ipdb import set_trace
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, EdgeConv
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from utils import exists_or_mkdir, string2bool, images_to_video
from torch_geometric.nn import radius_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EdgeConvUpdate(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean') #  "Max" aggregation.
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.Tanh(),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.Tanh(),
            nn.Linear(out_channels, 2, bias=False),
        )

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        tmp = x_j - x_i # tmp has shape [E, in_channels]
        return self.mlp(tmp)

    class EdgeConvUpdate(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr='mean')  # "Max" aggregation.
            self.mlp = nn.Sequential(
                nn.Linear(in_channels*2, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )

        def forward(self, x, edge_index):
            # x has shape [N, in_channels]
            # edge_index has shape [2, E]
            return self.propagate(edge_index, x=x)

        def message(self, x_i, x_j):
            # x_i has shape [E, in_channels]
            # x_j has shape [E, in_channels]
            # tmp = x_j - x_i  # tmp has shape [E, in_channels]
            tmp = torch.cat([x_i, x_j], dim=-1)
            return self.mlp(tmp)


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreModelGNN(nn.Module):
    def __init__(self, marginal_prob_std_func, num_nodes, hidden_dim=64, embed_dim=32):
        super(ScoreModelGNN, self).__init__()
        self.num_nodes = num_nodes

        self.init_lin = nn.Linear(2, hidden_dim)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        init_dim = hidden_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(init_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = EdgeConv(self.mlp1)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embed_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = EdgeConv(self.mlp2)
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embed_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2),
        )
        self.conv3 = EdgeConv(self.mlp3)

        self.marginal_prob_std = marginal_prob_std_func

    def forward(self, state_inp, t):
        # t.shape == [bs, 1]
        x, edge_index, batch = state_inp.x, state_inp.edge_index, state_inp.batch

        # we can norm here as ncsn's code did
        init_feature = F.relu(self.init_lin(x))

        # get t feature
        # t -> [bs*30, embed_dim]
        bs = t.shape[0]
        x_sigma = F.relu(self.embed(t.squeeze(1))).unsqueeze(1).repeat(1, self.num_nodes, 1).view(bs * self.num_nodes, -1)

        # start massage passing from init-feature
        x = F.relu(self.conv1(init_feature, edge_index))
        x = torch.cat([x, x_sigma], dim=-1)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.conv3(x, edge_index)

        # normalize the output
        # 注意当t=0时，marginal_prob_std = 0，其实意思就是梯度无穷大
        x = x / (self.marginal_prob_std(t.repeat(1, self.num_nodes).view(bs * self.num_nodes, -1)) + 1e-7)
        return x

class ScoreModelGNNMini(nn.Module):
    def __init__(self, marginal_prob_std_func, num_nodes, hidden_dim=64, embed_dim=32):
        super(ScoreModelGNNMini, self).__init__()
        self.num_nodes = num_nodes

        self.init_lin = nn.Linear(2, hidden_dim)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        init_dim = hidden_dim
        self.mlp1 = nn.Sequential(
            nn.Linear((init_dim + embed_dim) * 2, hidden_dim),  # 看pt-geometry 理解edge conv细节
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2),
        )
        self.conv1 = EdgeConv(self.mlp1)

        self.marginal_prob_std = marginal_prob_std_func

    def forward(self, state_inp, t):
        # t.shape == [bs, 1]
        x, edge_index, batch = state_inp.x, state_inp.edge_index, state_inp.batch
        # we can norm here as ncsn's code did
        init_feature = F.relu(self.init_lin(x))

        # get t feature
        bs = t.shape[0]
        x_sigma = F.relu(self.embed(t.squeeze(1))).unsqueeze(1).repeat(1, self.num_nodes, 1).view(bs * self.num_nodes,                                                                                  -1)

        # start massage passing from init-feature
        x = torch.cat([init_feature, x_sigma], dim=-1)
        x = self.conv1(x, edge_index)

        # normalize the output
        # 注意当t=0时，marginal_prob_std = 0，其实意思就是梯度无穷大
        x = x / (self.marginal_prob_std(t.repeat(1, self.num_nodes).view(bs * self.num_nodes, -1)) + 1e-7)
        return x

class MiniUpdate(nn.Module):
    def __init__(self, marginal_prob_std_func, num_nodes, hidden_dim=64, embed_dim=32):
        super(MiniUpdate, self).__init__()
        self.num_nodes = num_nodes

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(True),
        )

        self.conv_spatial = EdgeConvUpdate(2, hidden_dim)

        self.tail = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2, bias=False),
            # nn.Tanh(),
        )

        self.marginal_prob_std = marginal_prob_std_func

    def forward(self, state_inp, t):
        # t.shape == [bs, 1]
        x, edge_index, batch = state_inp.x, state_inp.edge_index, state_inp.batch

        # get t feature
        # bs = t.shape[0]
        # set_trace()
        t_expand = t[batch]
        t_feat = self.embed(t_expand.squeeze(1))
        # t_feat = F.relu(self.embed(t.squeeze(1))).unsqueeze(1).repeat(1, self.num_nodes, 1).view(bs*self.num_nodes, -1)
        # random_t = random_t[x.batch]

        # start massage passing from init-feature
        
        feat_spatial = self.conv_spatial(x, edge_index)
        # set_trace()
        # h = torch.tanh(feat_spatial * t_feat)
        h = feat_spatial
        # h = torch.cat([feat_spatial, t_feat], dim=-1)
        # h = self.tail(h)

        # normalize the output
        # out = h / (self.marginal_prob_std(t.repeat(1, self.num_nodes).view(bs*self.num_nodes, -1))+1e-7)
        # out = h / (self.marginal_prob_std(t_expand)+1e-7)
        out = h
        return out


def marginal_prob_std(t, sigma):
    # t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma ** t

# for unconditional ball-arrangement
def loss_fn(model, x, marginal_prob_std_func, n_box, eps=1e-5, scale=1., r=0.8):
    random_t = torch.rand(x.x.shape[0]//n_box, device=device) * (1. - eps) + eps
    random_t *= scale
    # -> [bs, 1]
    random_t = random_t.unsqueeze(-1)
    z = torch.randn_like(x.x)
    # -> [bs*n_box, 1]
    std = marginal_prob_std_func(random_t).repeat(1, n_box).view(-1, 1)
    perturbed_x = copy.deepcopy(x)
    perturbed_x.x += z * std
    # Graph 也得改！！！！
    batch_size = x.x.shape[0]//n_box
    x_batch = torch.tensor([i for i in range(batch_size) for _ in range(n_box)], dtype=torch.int64).to(device)
    # set_trace()
    perturbed_x.edge_index = radius_graph(perturbed_x.x.reshape(-1, 2), r=r, batch=x_batch)
    output = model(perturbed_x, random_t)

    bs = random_t.shape[0]
    loss_ = torch.mean(torch.sum(((output * std + z)**2).view(bs, -1)), dim=-1)
    return loss_

def loss_fn_cond(model, x, marginal_prob_std_func, eps=1e-5, scale=1., r=0.8):
    # set_trace()
    batchsize = x.ptr.shape[0]-1
    # 加一丁点noise，反正没坏处
    x.x = x.x*(1-eps) + torch.randn_like(x.x, device=device) * eps
    # Ver1 : t-range
    # random_t = torch.rand(batchsize, device=device) * (1. - eps) + eps
    # random_t *= scale
    # Ver2 : fix t
    random_t = torch.ones(batchsize, device=device) * scale
    # -> [bs, 1]
    random_t = random_t.unsqueeze(-1)
    # [bs, 1] -> [num_nodes, 1]
    random_t_expand = random_t[x.batch]
    # z: [num_nodes, 3]
    z = torch.randn_like(x.x)
    # std: [num_nodes, 1]
    std = marginal_prob_std_func(random_t_expand)
    perturbed_x = copy.deepcopy(x)
    perturbed_x.x += z * std
    # 注意edge得重新算一次
    perturbed_x.edge_index = radius_graph(perturbed_x.x, r=r, batch=perturbed_x.batch)
    output = model(perturbed_x, random_t)
    # perturbed_x_ = copy.deepcopy(perturbed_x)
    # perturbed_x_.x = -perturbed_x_.x
    # output_ = model(perturbed_x_, random_t)
    # set_trace()
    # output: [num_nodes, 3]
    node_l2 = torch.sum((output * std + z)**2, dim=-1)
    batch_l2 = scatter_sum(node_l2, x.batch, dim=0)
    loss_ = torch.mean(batch_l2)
    return loss_

def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=500,
                           device='cuda',
                           eps=1e-3):
    # start from prior distribution, this 't' is only used to compute the sigma of prior dist
    t = torch.ones(batch_size, device=device).unsqueeze(-1)
    k = 30 - 1
    init_x = torch.rand(batch_size*30, 2, device=device) * marginal_prob_std(t).repeat(1, 30).view(-1, 1)
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(30)], dtype=torch.int64)
    init_x = Data(x=init_x, edge_index=knn_graph(init_x.cpu(), k=k, batch=init_x_batch),
                       batch=init_x_batch).to(device)
    # init_x: Data(x=[480, 2], edge_index=[2, 13920], batch=[480])
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    states = []
    with torch.no_grad():
        for time_step in time_steps:
            # [bs, 1]
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # [bs*30, 1]
            g = diffusion_coeff(batch_time_step).repeat(1, 30).view(-1, 1)
            # [bs*30, 2]
            mean_x = x.x + (g ** 2) * score_model(x, batch_time_step) * step_size
            # [bs*30, 2]
            x.x = (mean_x + torch.sqrt(step_size) * g * torch.randn_like(x.x)).clamp(-1, 1)
            # Do not include any noise in the last sampling step.
            states.append(x.x.view(-1).unsqueeze(0))

    
    return torch.cat(states, dim=0), mean_x

#@title Define the Predictor-Corrector sampler (double click to expand or collapse)

def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=500, 
               snr=0.16,                
               device='cuda',
               eps=1e-3,
               n=30):
    t = torch.ones(batch_size, device=device).unsqueeze(-1)
    k = n - 1
    init_x = torch.randn(batch_size * n, 2, device=device) * marginal_prob_std(t).repeat(1, n).view(-1, 1)
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(n)], dtype=torch.int64)
    init_x = Data(x=init_x, edge_index=knn_graph(init_x.cpu(), k=k, batch=init_x_batch),
            batch=init_x_batch).to(device)
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    noise_norm = np.sqrt(n)
    x = init_x
    states = []
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x.x = x.x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x.x)      

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step).repeat(1, n).view(-1, 1)
            mean_x = x.x + (g ** 2) * score_model(x, batch_time_step) * step_size
            # [bs*30, 2]
            x.x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x.x)
            states.append(x.x.view(-1).unsqueeze(0))

    # The last step does not include any noise
    return torch.cat(states, dim=0), mean_x

# def ode_sampler(score_model,
#                 marginal_prob_std,
#                 diffusion_coeff,
#                 batch_size=64,
#                 atol=1e-5,
#                 rtol=1e-5,
#                 device='cuda',
#                 eps=1e-3,
#                 num_nodes=12,
#                 knn=4,
#                 t0=1,
#                 num_steps=None,
#                 scale=0.02):
#     t = torch.ones(batch_size, device=device).unsqueeze(-1) * 0.1
#
#     k = knn
#     n = num_nodes
#     # Create the latent code
#     init_x = torch.randn(batch_size * n, 2, device=device) \
#         * marginal_prob_std(t).repeat(1, n).view(-1, 1)
#     init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(n)], dtype=torch.int64)
#     edge_index = knn_graph(init_x.cpu(), k=k, batch=init_x_batch)
#
#     shape = init_x.shape
#
#     def score_eval_wrapper(sample, time_steps):
#         """A wrapper of the score-based model for use by the ODE solver."""
#         with torch.no_grad():
#             score = score_model(sample, time_steps)
#         return score.cpu().numpy().reshape((-1,))
#
#     def ode_func(t, x):
#         """The ODE function for use by the ODE solver."""
#         x = Data(x=torch.tensor(x.reshape(-1, 2), dtype=torch.float32), edge_index=edge_index, batch=init_x_batch).to(device)
#         time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
#         g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
#         return  0.5 * (g**2) * score_eval_wrapper(x, time_steps)
#
#     # # Run the black-box ODE solver.
#     # t_eval = None
#     # if num_steps is not None:
#     #     # num_steps, from t0 -> eps
#     t_eval = np.linspace(t0, eps, num_steps)
#     x = init_x.reshape(-1).cpu().numpy()
#     x = np.clip(x, -1, 1)
#     xs = [x.reshape((-1, num_nodes*2))[0]]
#     for t in t_eval:
#         vels = ode_func(t, x)
#         x += vels * scale
#         x = np.clip(x, -1, 1)
#         xs.append(x.reshape((-1, num_nodes*2))[0]) # 因为过程结果我们只用来存视频，而视频我们只存一个
#
#     # res = integrate.solve_ivp(ode_func, (t0, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
#     # xs = torch.clamp(torch.tensor(res.y[:n* 2], device=device).T, min=-1.0, max=1.0)
#     # x = torch.clamp(torch.tensor(res.y[:, -1], device=device).reshape(shape), min=-1.0, max=1.0)
#     # set_trace()
#     return torch.tensor(xs, device=device), torch.tensor(x, device=device)

def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                bound,
                batch_size=64,
                atol=1e-5,
                rtol=1e-5,
                device='cuda',
                eps=1e-3,
                num_nodes=12,
                r=0.2,
                t0=1,
                num_steps=None,
                scale=0.02):
    t = torch.ones(batch_size, device=device).unsqueeze(-1) * 0.1

    n = num_nodes
    # Create the latent code
    init_x = torch.randn(batch_size * n, 2, device=device) * marginal_prob_std(t).repeat(1, n).view(-1, 1) * bound
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(n)], dtype=torch.int64)
    # edge_index = radius_graph(init_x.cpu(), r=r, batch= init_x_batch)
    # edge_index = knn_graph(init_x.cpu(), k=k, batch=init_x_batch)

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,))

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        inp_x = torch.tensor(x.reshape(-1, 2), dtype=torch.float32)
        # 血妈坑！竟然没有根据当前的x来决定edge！！
        edge_index = radius_graph(inp_x, r=r, batch=init_x_batch)
        inp_x = Data(x=inp_x, edge_index=edge_index, batch=init_x_batch).to(
            device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return 0.5 * (g ** 2) * score_eval_wrapper(inp_x, time_steps)

    # # Run the black-box ODE solver.
    # t_eval = None
    # if num_steps is not None:
    #     # num_steps, from t0 -> eps
    # t_eval = np.linspace(t0, eps, num_steps)
    x = init_x.reshape(-1).cpu().numpy()
    x = np.clip(x, -bound, bound)
    xs = [x.reshape((-1, num_nodes * 2))[0]]
    for _ in range(num_steps):
        vels = ode_func(t0, x)
        x += vels * scale
        x = np.clip(x, -bound, bound)
        xs.append(x.reshape((-1, num_nodes * 2))[0])  # 因为过程结果我们只用来存视频，而视频我们只存一个

    # res = integrate.solve_ivp(ode_func, (t0, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    # xs = torch.clamp(torch.tensor(res.y[:n* 2], device=device).T, min=-1.0, max=1.0)
    # x = torch.clamp(torch.tensor(res.y[:, -1], device=device).reshape(shape), min=-1.0, max=1.0)
    # set_trace()
    return torch.tensor(xs, device=device), torch.tensor(x, device=device)