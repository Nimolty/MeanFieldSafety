# -*- coding: utf-8 -*-
"""
Created on Sun May  8 01:02:31 2022

@author: lenovo
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 22:21:35 2022

@author: lenovo
"""
from Envs.RLEnvSortingBall import RLSorting
from ipdb import set_trace
import numpy as np
from scipy.spatial import distance_matrix
from math import *
from functools import partial
from tqdm import tqdm, trange
import cv2
import time
import functools
import scipy.misc
from copy import deepcopy
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph, radius_graph

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Algorithms.SDE import ScoreModelGNN, ScoreModelGNNMini, MiniUpdate,  marginal_prob_std, diffusion_coeff
from Algorithms.pyorca.pyorca import Agent
from Envs.test_ball_score import vis_scores
from utils import exists_or_mkdir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_video(env, states, save_path, simulation=False, fps=50, render_size=256, camera_hight=1.0, suffix='avi'):
    imgs = []
    for _, state in tqdm(enumerate(states), desc='Saving video'):
        # set_trace()
        env.set_state(state, simulation)
        img = env.render(render_size, camera_hight=camera_hight)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    images_to_video(save_path+f'.{suffix}', batch_imgs, fps, (render_size, render_size))


def save_video2(env, states, save_path, simulation=False, fps = 50, render_size = 256, suffix='avi'):
    imgs = []
    for _, state in tqdm(enumerate(states), desc='Saving video'):
        # set_trace()
        env.set_state(state, simulation)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    images_to_video(save_path+f'.{suffix}', batch_imgs, fps, (render_size, render_size))

def images_to_video(path, images, fps, size):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for item in images:
        out.write(item)
    out.release()

def snapshot(env, file_name):
    img = env.render(256)
    cv2.imwrite(file_name, img)

def get_cur_state_np(agents):
    return np.stack([agent.position for agent in agents])
    # ?????????agents???????????????????????????

def assign_tar_vels(agents, vels):
    for agent, vel in zip(agents, vels):
        agent.pref_velocity = np.array(vel)
    # ???tar???pref_velocity??????????????????????????????vel

def comp_sup_vel(state, t, score,r):
    # set_trace()
    # state_inp = state * SCALE
    state_inp = state
    with torch.no_grad():
        # !!! MMD ????????????????????????state??????edge graph??????
        #print(state)
        #print(radius_graph)
        edge = radius_graph(torch.tensor(state, device=device).view(-1, 2), r)
        convert_to_inp = partial(Data, edge_index=edge)
        state_ts = convert_to_inp(x=torch.tensor(state_inp).to(device))
        inp_batch = Batch.from_data_list([state_ts])
        vels = score(inp_batch, t) # ??????normalize???sup vel?????????????????????????????????
        # vels = out_score*MAX_VEL/(torch.max(torch.abs(out_score)) + 1e-7) # norm to max_vel = MAX_VEL
        # print(vels.max())
    return vels.cpu().numpy()
    # ????????????????????????????????????score??????????????????state???????????????si(\theta)??????


def normalise_vels(vels, MAX_VEL):
    vels = np.array(vels)
    max_vel_norm = np.max(np.abs(vels))
    scale_factor = MAX_VEL / max_vel_norm
    scale_factor = np.min([scale_factor, 1])
    new_vels = scale_factor * vels
    return new_vels

def dummy_collsion_checker(agents):
    n = len(agents)
    cur_states = get_cur_state_np(agents) # [n, 2]
    distancs = distance_matrix(cur_states, cur_states) # [n, n]
    return np.sum(distancs < RADIUS) - n # ??????????????????????????????0

def distance(pose1, pose2):
    """ compute Euclidean distance for 2D """
    return sqrt((pose1[0]-pose2[0])**2+(pose1[1]-pose2[1])**2)+0.001

def reach(p1, p2, bound=0.5):
    if distance(p1,p2)< bound:
        return True
    else:
        return False

def compute_V_des(X, goal, V_max):
    V_des = []
    for i in range(len(X)):
        dif_x = [goal[i][k]-X[i][k] for k in range(2)]
        norm = distance(dif_x, [0, 0])
        norm_dif_x = [dif_x[k]*V_max[k]/norm for k in range(2)]
        V_des.append(norm_dif_x[:])
        if reach(X[i], goal[i], 0.01):
            V_des[i][0] = 0
            V_des[i][1] = 0
    return V_des

    
    
def plan(SUP_RATE, MAX_VEL, EXP_NAME = 'M5D1_r04',N_BOXES=10,WALL_BOUND=0.2,neighbor_std = 0.3, R=0.4,NUM_SCALE=10,\
         agent_radius=0.025/(0.2-0.025),SIGMA=25.,PB_FREQ=4,dt=1/50,DURATION=5,t0=1e-2,is_gui=False, \
         IS_SIM=True,IS_GOAL=True,IS_SUPPORT=True,VIDEO_NUM=0):
    MAX_VEL = MAX_VEL
    SUP_RATE = SUP_RATE
    padding = 0.03
    score_grid = 40
    arrowwidth = 0.010
    
    time_freq = int(1 / dt)
    env_kwargs = {
        'n_boxes' : N_BOXES,
        'wall_bound': WALL_BOUND,
        'time_freq':time_freq * PB_FREQ,
        'is_gui' : is_gui, 
        'max_action' : MAX_VEL,
        'normalize' : True,
        }
    test_env = RLSorting(**env_kwargs) #???????????????????????????
    test_env.seed(0)
    
    # ?????????????????????????????????????????????????????????????????????
    init_state = test_env.reset()
    if neighbor_std > 0:
        target_state = test_env.reset(init_state, neighbor_std)
    else:
        target_state = test_env.reset()
    test_env.set_state(init_state)
    
    n_objs = test_env.n_boxes
    init_state = init_state.reshape((n_objs, -1))
    target_state = target_state.reshape((n_objs, -1))
    #print(target_state)
    
    # ?????????pretrained???score????????????????????????
    sup_path = f'./logs/{EXP_NAME}/score.pt'
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=SIGMA)
    score_support = MiniUpdate(marginal_prob_std_fn, n_objs, hidden_dim=256, embed_dim=128)
    # ????????????????????????scoremodel???????????????????????????????????????????????????????????????
    # ??????pretrained?????????
    score_support.load_state_dict(torch.load(sup_path))
    
    # ????????????????????????????????????????????????????????????????????????????????????
    #r = 0.2
    #edge_graph = radius_graph(torch.ones((n_objs, 2)).to(device), r)
    # ?????????????????????get_sup_vel????????????????????????????????????????????????
    # comp_des_vel ???????????????????????????????????????
    t = torch.tensor([t0]).unsqueeze(0).to(device) # t???shape???torch.Size([1,1])
    get_sup_vel = partial(comp_sup_vel, score=score_support.to(device), r=R)
    sup_vel = get_sup_vel(init_state, t)
    
    comp_des_vel = partial(compute_V_des, goal=target_state.tolist(), V_max = [MAX_VEL] * (n_objs))
    tar_vel = comp_des_vel(init_state.tolist())
    
    # ????????????????????????????????????????????????????????????SUPPORT???IS_GOAL?????????????????????????????????????????????????????????????????????bool???
    if IS_GOAL:
        if IS_SUPPORT:
            obj_vel = sup_vel * SUP_RATE + normalise_vels(tar_vel, MAX_VEL)
        else:
            obj_vel = tar_vel
    else:
        obj_vel = sup_vel
    obj_vel = normalise_vels(obj_vel,MAX_VEL)
    
    agents = []
    for i in range(init_state.shape[0]):
        agents.append(Agent(init_state[i], (0.0, 0.0), agent_radius, MAX_VEL, np.array(obj_vel[i])))
    
    #--------------------------------------
    # simulation starts
    states_np = []
    total_steps = int(DURATION/dt)
    collision_num = 0
    vel_errs = []
    vel_errs_mean = []
    for step in range(total_steps):
        try:
            cur_state = get_cur_state_np(agents)
            states_np.append(deepcopy(cur_state).reshape(-1))
            
            tar_vels = comp_des_vel(cur_state.tolist())
            #print(cur_state - target_state)
            sup_vels = get_sup_vel(cur_state, t)
            
            if IS_GOAL:
                if IS_SUPPORT:
                    obj_vels = sup_vels * SUP_RATE + normalise_vels(tar_vels, MAX_VEL)
                else:
                    obj_vels = tar_vels
            else:
                obj_vels = sup_vels
            obj_vels = normalise_vels(obj_vels, MAX_VEL)
            assign_tar_vels(agents, obj_vels)
            
            if IS_SIM:
                '''sim-based'''
                # update current vels for each agent
                for i, agent in enumerate(agents):
                    agent.velocity = obj_vels[i]
                new_state, _, infos = test_env.step(np.array(obj_vels), step_size=PB_FREQ)
                vel_errs.append(infos['vel_err'])
                vel_errs_mean.append(infos['vel_err_mean'])
                for i, agent in enumerate(agents):
                    agent.position = new_state[2*i:2*i+2]
            else:
                #obj_vels *= 1 / BOUND # ????????????????????????
                '''rule-based update, not sim'''
                for i, agent in enumerate(agents):
                    agent.velocity = obj_vels[i]
                    agent.position += agent.velocity * dt
        except Exception as e:
            print(e)
            break
        collision_num += infos['collision_num'] if IS_SIM else dummy_collision_checker(agents)
    

    final_state = get_cur_state_np(agents)
    delta_pos = np.sqrt(np.sum((final_state - target_state)**2, axis=1))
    
    # safety
    if IS_SIM:
        print(
            f'### Average vel err: {sum(vel_errs) / len(vel_errs)} || Mean vel err: {sum(vel_errs_mean) / len(vel_errs_mean)}###')
        print(
            f'###All delta_pos : {delta_pos.sum()} || Mean delta_pos : {np.mean(delta_pos)} || Max delta_pos : {np.max(delta_pos)}###')
    if collision_num == 0:
        print("###Totally safe###")
    else:
        print(f'### Mean Collision Num: {collision_num / total_steps} || Total Collision Num: {collision_num} ###')
    
    if collision_num == 0 and np.max(delta_pos) < 0.05:
        print('Find it')
    test_env.close()
    
    return np.max(delta_pos), np.mean(delta_pos), collision_num, collision_num/total_steps
    # set_trace()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',type=str,default='M5D1_r04')
    parser.add_argument('--date', type=int, default= 518)
    parser.add_argument('--n_boxes', type = int,default=10)
    parser.add_argument('--dist_r', type=int, default=3)
    parser.add_argument('--scale', type=float, default=2)
    parser.add_argument('--sup_rate_init', type=float,default=0.05)
    parser.add_argument('--max_vel_ratio', type=float, default=0.1)
    parser.add_argument('--neighbor_std', type=float, default=8)
    parser.add_argument('--duration', type=float, default=5.)
    
    args = parser.parse_args()
    exp_name = args.exp_name
    n_boxes = args.n_boxes
    sup_rate_init = args.sup_rate_init
    scale = args.scale
    dist_r = args.dist_r
    date = args.date
    wall_bound = 0.2 * scale * np.sqrt(int(n_boxes / 10))
    max_vel_ratio = args.max_vel_ratio
    neighbor_std = args.neighbor_std
    radius = dist_r * 0.025 / (wall_bound - 0.025)
    
    path = f'./logs/M5D1_r04/{date}'
    exists_or_mkdir(path)
    out = open(f'./logs/M5D1_r04/{date}/record_N{n_boxes}_WB{round(wall_bound,2)}_dr{dist_r}_nb{neighbor_std}.txt', 'w')
    sup_rate = [sup_rate_init + i * 0.01 for i in range(100)]
    max_vel_ratio = [max_vel_ratio + i * 0.015 for i in range(200)]
    for i in tqdm(sup_rate):
        for j in max_vel_ratio:
            max_vel = j * (wall_bound - 0.025)
            max_, mean_, coll_num, mean_coll_num = plan(i, max_vel,exp_name, n_boxes, wall_bound, neighbor_std, radius, DURATION=args.duration)
            if mean_coll_num < 0.1 or max_ < 0.1:
                max_ = format(max_, '.4f')
                mean_ = format(mean_, '.4f')
                max_vel = format(max_vel, '.4f')
                j_ = format(j, '.4f')
                out.write(f'sup_rate : {i}, max_vel : {max_vel}, max_vel_ratio : {j_}, radius : {round(radius, 4)}' + f'\n')
                out.write(f'Mean delta pos {mean_}, Max delta pos {max_}, Total collision Num : {coll_num}, Mean collision num : {mean_coll_num}' + f'\n')
    out.close()









