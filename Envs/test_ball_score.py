import os
import argparse
import functools
from turtle import left
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.utils.tensorboard import SummaryWriter
#from losses.dsm import *
#from losses.sliced_sm import *

from ipdb import set_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True
    
def marginal_prob_std(t, sigma):
    # t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))
    
def draw_circle(radius=4, point_num=500, color='red', alpha=0.3):
    angle = np.linspace(0, 2 * np.pi, point_num)

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    fig = plt.figure()
    plt.plot(x, y, c=color, alpha=1)
    ax = plt.gca()
    cir = Circle(xy=(0.0, 0.0), radius=radius, alpha=alpha, color=color)
    ax.add_patch(cir)
    return fig

def vis_scores(
        model,
        left_bound=-1.,
        right_bound=1.,
        radius=0.8,
        savefig=None,
        prefix=None,
        device=None,
        grid_size=20,
        log_norm=True,
        axis=False,
        padding=0.1,
        arrowwidth=0.010,
    ):
    print('Start drawing learned gradient field')
    mesh = []
    x = np.linspace(left_bound, right_bound, grid_size)
    y = np.linspace(left_bound, right_bound, grid_size)
    for i in x:
        for j in y:
            mesh.append(np.asarray([i, j]))
            
    mesh = np.stack(mesh, axis=0)
    mesh = torch.from_numpy(mesh).float()
    if device is not None:
        mesh = mesh.to(device)

    scores = -model(mesh.detach())
    # scores = torch.clamp(scores, -1e10, 5)
    # scores = torch.tanh(scores)*10
    # normalize the norm of scores
    # score_norms = torch.sqrt(torch.sum(scores ** 2, dim=-1, keepdim=True))
    # scores /= score_norms
    # set_trace()
    if log_norm:
        score_norms = torch.sqrt(torch.sum(scores ** 2, dim=-1, keepdim=True))
        scores /= score_norms
        scores *= torch.log(score_norms) * 2
 
    mesh = mesh.detach().numpy()
    scores = scores.detach().numpy()
    fig = draw_circle(radius=radius)
    # np.sum(scores**2, axis=-1).min() = 0.07408305
    # np.sum(scores**2, axis=-1).max() = 4505.5337
    plt.quiver(mesh[:, 0], mesh[:, 1], scores[:, 0], scores[:, 1], width=arrowwidth)  # default width: 0.005
    # plt.grid(False)
    plt.axis('Square')
    if not axis:
        plt.axis('off')
    step_size = (right_bound - left_bound) / 10
    plt.xticks(np.arange(left_bound, right_bound, step=step_size))
    plt.yticks(np.arange(left_bound, right_bound, step=step_size))
    plt.xlim((left_bound, right_bound))
    plt.ylim((left_bound, right_bound))
    
    plt.close()
    return fig
    # if savefig is not None:
    #     plt.savefig(savefig + "/{}_scores.png".format(prefix), bbox_inches='tight', pad_inches = padding)
    #     plt.close()
    # else:
    #     plt.show()
        
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
        x, edge_index, batch = state_inp.x, state_inp.edge_index, state_inp.batch

        # start massage passing from init-feature
        feat_spatial = self.conv_spatial(x, edge_index)
        h = feat_spatial
        out = h
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_sup', type=str, default='')
    parser.add_argument('--eval_name', type=str, default='')
    
    parser.add_argument('--n_box', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=25.)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    
    parser.add_argument('--radius', type=float, default=0.8)
    
    parser.add_argument('--gin_bindings', nargs='+')
    args = parser.parse_args()
    
    n_box = args.n_box
    sigma = args.sigma
    radius = args.radius
    hidden_dim = args.hidden_dim
    embed_dim = args.embed_dim
    ckpt_path = f'score.pt'
    eval_path = f'./logs'
    exists_or_mkdir(eval_path)
    
    writer = SummaryWriter(f'./logs')
    is_axis = True
    padding = 0.03
    score_grid = 40
    arrowwidth = 0.010 # default 0.005
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    sup_score = MiniUpdate(marginal_prob_std_fn, n_box, hidden_dim, embed_dim)
    sup_score.load_state_dict(torch.load(ckpt_path))
    f = sup_score.conv_spatial.mlp
 
    # draw gradient fields
    vis_scores(model=f.cpu(), radius=radius, savefig=eval_path, prefix='sup', log_norm=True, axis=is_axis,
               padding=padding, grid_size=score_grid, arrowwidth=arrowwidth)
