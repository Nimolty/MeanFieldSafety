from Envs.RLEnvSortingBall import RLSorting
from ipdb import set_trace
import numpy as np
from scipy.spatial import distance_matrix
from math import *
from functools import partial
from tqdm import tqdm
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
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Algorithms.SDE import ScoreModelGNN, ScoreModelGNNMini, MiniUpdate, marginal_prob_std, diffusion_coeff
from Algorithms.pyorca.pyorca import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MARGIN = 0.10
BOUND = 0.2
RADIUS = 0.025*(1+MARGIN) / (BOUND-0.025)
N_BOXES = 10
WALL_BOUND = 0.2
MAX_VEL = 0.3
DURATION = 5
KNN = 7
dt = 1/50
tau = 10*dt
t0 = 0.01 # for target score
PB_FREQ = 4 # 一个dt的时间，在PB的模拟器里切成PB_FREQ步（为了更模拟更精细）
is_gui = False
SIGMA=25

IS_SUPPORT = True
IS_GOAL = False
SUP_RATE = 0.9
IS_SIM = True

EXP_NAME = "M3D22_n1e5_10balls_bound02_Mini_sigma25_unnormalize_radius0.2"

def save_video(env, states, save_path, simulation=False, fps = 50, render_size = 256, suffix='avi'):
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

def assign_tar_vels(agents, vels):
    for agent, vel in zip(agents, vels):
        agent.pref_velocity = np.array(vel)

def comp_sup_vel(state, t, score, edge):
    with torch.no_grad():
        convert_to_inp = partial(Data, edge_index=edge)
        state_ts = convert_to_inp(x=torch.tensor(state).to(device))
        inp_batch = Batch.from_data_list([state_ts])
        out_score = score(inp_batch, t)
        vels = out_score*MAX_VEL/(torch.max(torch.abs(out_score)) + 1e-7) # norm to max_vel = MAX_VEL
        # print(vels.max())
    return vels.cpu().numpy()

def normalise_vels(vels):
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
    return np.sum(distancs < RADIUS) - n # 因为自己和自己一定是0

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

def plan():
    time_freq = int(1 / dt)
    env_kwargs = {
        'n_boxes': N_BOXES,
        'wall_bound': WALL_BOUND,
        'time_freq': time_freq*PB_FREQ,
        'is_gui': is_gui,
        'max_action': MAX_VEL,
        'normalize': False,
    }
    test_env = RLSorting(**env_kwargs)
    test_env.seed(0)

    #TODO
    target_state = test_env.reset()
    snapshot(test_env, 'goal.png')
    init_state = test_env.reset()
    snapshot(test_env, 'init.png')

    n_objs = test_env.n_boxes
    init_state = init_state.reshape((n_objs, -1))
    target_state = target_state.reshape((n_objs, -1))

    # ------------------------------
    # Init Target Score
    sup_path = f'./logs/{EXP_NAME}/score.pt'

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=SIGMA)
    score_support = ScoreModelGNNMini(marginal_prob_std_fn, n_objs, hidden_dim = 128, embed_dim=64)
    # score_support = ScoreModelGNN(marginal_prob_std_fn, n_objs, hidden_dim=64, embed_dim=32)
    # score_support = MiniUpdate(marginal_prob_std_fn, n_objs, hidden_dim=64, embed_dim=32)
    # score_support = MiniUpdate(marginal_prob_std_fn, n_objs, hidden_dim=64, embed_dim=32)
    score_support.load_state_dict(torch.load(sup_path))

    # edge_graph = knn_graph(torch.ones((n_objs, 2)).to(device), KNN)
    r = 0.2
    edge_graph = radius_graph(torch.ones((n_objs, 2)).to(device), r)
    t = torch.tensor([t0]).unsqueeze(0).to(device)
    get_sup_vel = partial(comp_sup_vel, score=score_support.to(device), edge=edge_graph)
    sup_vel = get_sup_vel(init_state, t)

    comp_des_vel = partial(compute_V_des, goal=target_state.tolist(), V_max=[MAX_VEL] * (n_objs))
    tar_vel = comp_des_vel(init_state.tolist())

    if IS_GOAL:
        if IS_SUPPORT:
            obj_vel = sup_vel * SUP_RATE + tar_vel
        else:
            obj_vel = tar_vel
    else:
        obj_vel = sup_vel
    # TODO norm(tar+sup）-> norm(norm(tar)+sup)
    obj_vel = normalise_vels(obj_vel)

    agents = []
    for i in range(init_state.shape[0]):
        agents.append(Agent(init_state[i], (0., 0.), RADIUS, MAX_VEL, np.array(obj_vel[i])))


    # ------------------------------
    # simulation starts
    states_np = []
    total_steps = int(DURATION/dt)
    collision_num = 0
    vel_errs = []
    vel_errs_mean = []
    for step in tqdm(range(total_steps)):
        try:
            # compute desired vel to goal, and assign cur vels
            cur_state = get_cur_state_np(agents)
            states_np.append(deepcopy(cur_state).reshape(-1))

            tar_vels = comp_des_vel(cur_state.tolist())
            sup_vels = get_sup_vel(cur_state, t)

            if IS_GOAL:
                if IS_SUPPORT:
                    obj_vels = sup_vels * SUP_RATE + tar_vels
                else:
                    obj_vels = tar_vels
            else:
                obj_vels = sup_vels
            obj_vels = normalise_vels(obj_vels)
            # print(tar_vels)
            assign_tar_vels(agents, obj_vels)

            if IS_SIM:
                ''' sim-based '''
                # update current vels for each agent
                for i, agent in enumerate(agents):
                    agent.velocity = obj_vels[i]
                # step vels in sim
                # set_trace()
                # old_state = get_cur_state_np(agents)
                new_state, _, infos = test_env.step(np.array(obj_vels), step_size=PB_FREQ)
                vel_errs.append(infos['vel_err'])
                vel_errs_mean.append(infos['vel_err_mean'])
                # print(f'cur max vel error: {vel_err}')
                # np.sum(((new_state - old_state.reshape(-1)) / dt - new_vels.reshape(-1)*1/BOUND)**2) # 0.005956928933356269
                # delta = np.sum(np.abs((new_state - old_state.reshape(-1))/dt - new_vels.reshape(-1)*1/BOUND)) # 0.005956928933356269
                # print(infos['collision_num'])
                # print(delta)
                # set_trace()
                # update positions according to sim
                for i, agent in enumerate(agents):
                    agent.position = new_state[2 * i:2 * i + 2]
            else:
                # 要转化为物理速度
                obj_vels *= 1 / BOUND
                # old_state = get_cur_state_np(agents)
                ''' rule-based update, not sim '''
                for i, agent in enumerate(agents):
                    agent.velocity = obj_vels[i]
                    agent.position += agent.velocity * dt
                # new_state = get_cur_state_np(agents)
                # delta = np.sum(np.abs((new_state - old_state)/dt - new_vels))  # 0.005956928933356269
                # print(delta)
                # set_trace()

        except Exception as e:
            print(e)
            break
        collision_num += infos['collision_num'] if IS_SIM else dummy_collsion_checker(agents)

    #safety
    if IS_SIM:
        print(
            f'### Average vel err: {sum(vel_errs) / len(vel_errs)} || Mean vel err: {sum(vel_errs_mean) / len(vel_errs_mean)}###')
    if collision_num == 0:
        print("Totally safe")
    else:
        print(f'### Mean Collision Num: {collision_num / total_steps}, Total Collision Num: {collision_num} ###')
    # set_trace()
    save_video(test_env, states_np, save_path=f'./logs/{EXP_NAME}', fps=len(states_np) // DURATION, suffix='mp4')


if __name__ == "__main__":
    plan()
