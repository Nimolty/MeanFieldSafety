# -*- coding: utf-8 -*-
"""
Created on Sat May  7 16:21:52 2022

@author: lenovo
"""
import os
import sys
import argparse
import functools
from turtle import left
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Envs.RLEnvSortingBall import RLSorting
from utils import exists_or_mkdir, string2bool, images_to_video, save_video
from Algorithms.SDE import ScoreModelGNN, ScoreModelGNNMini, MiniUpdate, \
    marginal_prob_std, diffusion_coeff, loss_fn, \
    Euler_Maruyama_sampler, pc_sampler, ode_sampler, loss_fn_cond
from Envs.test_ball_score import draw_circle, vis_scores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='M5D1_r04')
    parser.add_argument('--sigma', type=float, default=25.)
    parser.add_argument('--n_boxes', type = int,default=10)
    parser.add_argument('--radius', type=float, default=0.4)
    
    args = parser.parse_args()
    exp_name = args.exp_name
    sigma = args.sigma
    n_boxes = args.n_boxes
    r = args.radius
    padding = 0.03
    score_grid = 40
    arrowwidth = 0.01
    
    exists_or_mkdir('./logs')
    f_path1 = f'./logs/{exp_name}/normed_score.png'
    f_path2 = f'./logs/{exp_name}/unnormed_score.png'
    #exists_or_mkdir(f_path)
    score_path = f'./logs/{exp_name}/score.pt'
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score = MiniUpdate(marginal_prob_std_fn, n_boxes, hidden_dim=256, embed_dim=128)
    score.load_state_dict(torch.load(score_path))
    score.to(device)
    
    f = score.conv_spatial.mlp.cpu()
    fig1 = vis_scores(model=f.cpu(), radius=r, left_bound=-0.4, right_bound=0.4,\
    savefig=None, prefix='sup', norm=True, axis=True,\
    padding=padding, grid_size=score_grid, arrowwidth=arrowwidth)
    fig2 = vis_scores(model=f.cpu(), radius=r, left_bound=-0.4, right_bound=0.4,\
    savefig=None, prefix='sup', norm=False, axis=True,\
    padding=padding, grid_size=score_grid, arrowwidth=arrowwidth)
        
    fig1.savefig(f_path1)
    fig2.savefig(f_path2)























