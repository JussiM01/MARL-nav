import argparse
import os
import math
import torch

from marlnav.animation import init_render
from marlnav.environment import Env
from marlnav.models import MAPPO
from marlnav.utils import set_params, set_all_seeds, check_rews # NOTE: LAST ONE IS FOR TESTING. REMOVE LATER ?


def main(params, mode):

    env = Env(params['env'])

    if mode == 'training':
        num_total = params['model']['num_total']
        num_parallel = params['model']['num_parallel']
        buffer_len = params['model']['buffer_len']
        num_repeats = num_total // (buffer_len * num_parallel)
        mappo = MAPPO(params['model'], env)

        for i in range(num_repeats):
            print('repeat', i+1) # NOTE: FOR DEBUGGING. CHOOSE A BETTER PROGESS LOGGING FOR ACTUAL USE
            mappo.get_data()
            mappo.train_actor()
            mappo.train_critic()
        mappo.save_stats(params)

    elif mode == 'rendering':
        renderer = init_render(env, params)
        renderer.run()

    elif mode == 'reward_check':
        check_rews(
            env,
            params['animation']['max_step'],
            params['animation']['parallel_index'],
            params['animation']['agent_index']
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('-se', '--seed', type=int,
        help='value of the random seed (optional, default is None).')
    parser.add_argument('-mx', '--max_x_value', type=float, default=1500.0,
        help='maximum value for the x-coordinates')
    parser.add_argument('-my', '--max_y_value', type=float, default=750.0,
        help='maximum value for the y-coordinates')

    # animation/plot args
    parser.add_argument('-fx', '--fig_size_x', type=float, default=10.0,
        help='animation plot width in centimeters')
    parser.add_argument('-fy', '--fig_size_y', type=float, default=5.0,
        help='animation plot height in centimeters')
    parser.add_argument('-pi', '--parallel_index', type=int, default=0, # NOTE: CHANGE LATER?
        help='index of the rendered environment in the parallelization axis')
    parser.add_argument('-ai', '--agent_index', type=int, default=0, # NOTE: CHANGE LATER?
        help='index of the agent for whose rewards are plotted')
    parser.add_argument('-in', '--interval', type=int, default=10,
        help='interval param for the animation (small is fast).')
    parser.add_argument('-ra', '--random', action='store_true',
        help='Stochastic policy (default: predicted mean), action: store_true')
    parser.add_argument('-w', '--weights_file', type=str,
        help='Name of the actor model weights file used for policy rendering.')

    # env args
    parser.add_argument('-np', '--num_parallel', type=int, default=2, # NOTE: DEFAULT=2 FOR TESTING, change this later?
        help='number of the parallel enviroments')
    parser.add_argument('-na', '--num_agents', type=int, default=3,
        help='number of agents in a single environment')
    parser.add_argument('-no', '--num_obstacles', type=int, default=3, # NOTE: DEFAULT=1 FOR TESTING, change this later?
        help='number of obstacles in a single environment')
    parser.add_argument('-ms', '--max_step', type=int, default=1000, # NOTE: DEFAULT=100 FOR TESTING, change this later?
        help='maximum number of time steps in the simulation')
    parser.add_argument('-el', '--episode_len', type=int, default=200, # NOTE: DEFAULT=100 FOR TESTING, change this later?
        help='maximum number od steps in an episode')
    parser.add_argument('-mis', '--min_speed', type=float, default=3.,
        help='Minimum cut-off value for the speed.')
    parser.add_argument('-mas', '--max_speed', type=float, default=10., # NOTE: CHANGE THIS LATER?
        help='Maximum cut-off value for the speed.')
    parser.add_argument('-mia', '--min_accel', type=float, default=-0.5, # NOTE: CHANGE THIS LATER?
        help='Minimum cut-off value for the acceleration.')
    parser.add_argument('-maa', '--max_accel', type=float, default=0.5, # NOTE: CHANGE THIS LATER?
        help='Maximum cut-off value for the acceleration.')
    parser.add_argument('-rf', '--risk_factor', type=float, default=0., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the risk loss.')
    parser.add_argument('-df', '--distance_factor', type=float, default=0., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the distance reward.')
    parser.add_argument('-hf', '--heading_factor', type=float, default=500., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the heading reward.')
    parser.add_argument('-tf', '--target_factor', type=float, default=500., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the target reward.')
    parser.add_argument('-sf', '--soft_factor', type=float, default=500., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the smooth target distance reward.')
    parser.add_argument('-bf', '--bond_factor', type=float, default=10., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the bond distance reward.')

    # model specific args
    parser.add_argument('-hs', '--hidden_size', type=int, default=50, # NOTE: CHANGE THIS LATER?
        help='Hidden layer size of the models.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, # NOTE: CHANGE THIS LATER?
        help='Learning rate for the training fo actor and critic models.')
    parser.add_argument('-ec', '--ent_const', type=float, default=0.001, # NOTE: CHANGE THIS LATER?
        help='Weight constant for the entropy loss.')
    parser.add_argument('-ep', '--epsilon', type=float, default=0.01, # NOTE: CHANGE THIS LATER?
        help='Epsilon parameter for the loss clipping.')
    parser.add_argument('-g', '--gamma', type=float, default=0.9, # NOTE: CHANGE THIS LATER?
        help='Gamma parameter for the cummulative rewards.')
    parser.add_argument('-nt', '--num_total', type=int, default=1000000, # NOTE: CHANGE THIS LATER?
        help='Number of total steps to be executed (parallel included).')
    parser.add_argument('-bl', '--buffer_len', type=int, default=1000, # NOTE: CHANGE THIS LATER?
        help='Length parameter for the buffer.')
    parser.add_argument('-ne', '--num_epochs', type=int, default=50, # NOTE: CHANGE THIS LATER?
        help='Number of training epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, default=1000, # NOTE: CHANGE THIS LATER?
        help='Mini-batch size (should be smaller or equal to buffer_len).')

    # init args
    parser.add_argument('-re', '--rendering', action='store_true',
        help='rendering option (no training), action: store_true' )
    parser.add_argument('-sa', '--sampling_style', type=str, default='sampler', # NOTE: FOR TESTING
        help='sampling style, should be either `policy` or `sampler`') # REMOVE THIS LATER ?
    parser.add_argument('-rc', '--reward_check', action='store_true', # NOTE: FOR DEBUGGING/TESTING ONLY!
        help='Runs fixed dynamics for checking the rewards from saved plots') # REMOVE THIS LATER ?
    parser.add_argument('-sn', '--sampler_num', type=int, default=-1, # NOTE: FOR DEBUGGING/TESTING ONLY!
        help='number code of the chosen params and mock_sampler') # REMOVE THIS LATER ?
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.seed is not None:
        set_all_seeds(args.seed)

    if args.rendering:
        mode = 'rendering'
    elif args.reward_check:  # NOTE: FOR TESTING. REMOVE LATER ?
        mode = 'reward_check'
    else:
        mode = 'training'

    params = set_params(args)
    main(params, mode)
