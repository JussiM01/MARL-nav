import argparse
import os
import math
import torch

from marlnav.animation import init_render
from marlnav.environment import Env
from marlnav.models import MAPPO
from marlnav.utils import load_config, set_all_seeds, check_rews # NOTE: LAST ONE IS FOR TESTING. REMOVE LATER ?


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
    parser.add_argument('-mis', '--min_speed', type=float, default=0.,
        help='Minimum cut-off value for the speed.')
    parser.add_argument('-mas', '--max_speed', type=float, default=10., # NOTE: CHANGE THIS LATER?
        help='Maximum cut-off value for the speed.')
    parser.add_argument('-mia', '--min_accel', type=float, default=-0.5, # NOTE: CHANGE THIS LATER?
        help='Minimum cut-off value for the acceleration.')
    parser.add_argument('-maa', '--max_accel', type=float, default=0.5, # NOTE: CHANGE THIS LATER?
        help='Maximum cut-off value for the acceleration.')
    parser.add_argument('-rf', '--risk_factor', type=float, default=50., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the risk loss.')
    parser.add_argument('-df', '--distance_factor', type=float, default=5., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the distance reward.')
    parser.add_argument('-hf', '--heading_factor', type=float, default=10., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the heading reward.')
    parser.add_argument('-tf', '--target_factor', type=float, default=500., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the target reward.')
    parser.add_argument('-sf', '--soft_factor', type=float, default=100., # NOTE: CHANGE THIS LATER?
        help='Weight factor for the smooth target distance reward.')

    # model specific args
    parser.add_argument('-hs', '--hidden_size', type=int, default=50, # NOTE: CHANGE THIS LATER?
        help='Hidden layer size of the models.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, # NOTE: CHANGE THIS LATER?
        help='Learning rate for the training fo actor and critic models.')
    parser.add_argument('-ec', '--ent_const', type=float, default=0.01, # NOTE: CHANGE THIS LATER?
        help='Weight constant for the entropy loss.')
    parser.add_argument('-ep', '--epsilon', type=float, default=0.01, # NOTE: CHANGE THIS LATER?
        help='Epsilon parameter for the loss clipping.')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, # NOTE: CHANGE THIS LATER?
        help='Gamma parameter for the cummulative rewards.')
    parser.add_argument('-nt', '--num_total', type=int, default=1000000, # NOTE: CHANGE THIS LATER?
        help='Number of total steps to be executed (parallel included).')
    parser.add_argument('-bl', '--buffer_len', type=int, default=1000, # NOTE: CHANGE THIS LATER?
        help='Length parameter for the buffer.')
    parser.add_argument('-ne', '--num_epochs', type=int, default=10, # NOTE: CHANGE THIS LATER?
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

    ### NOTE: This section should be temporary or refactored to a JSON-file ####
    if args.sampler_num == -1:

        if args.num_agents != 3:
            raise ValueError

        params = { # NOTE: THIS ONE IS FOR TESTING TRIANGLE INITIALIZATION
            'init': {
                'init_method': 'triangle',
                'device': device,
                'num_parallel': args.num_parallel,
                'ags_cent_x': 150.,
                'ags_cent_y': 375.,
                'ags_dist': 40.,
                'tar_pos_x': 1350.,
                'tar_pos_y': 375.,
                'num_obs': args.num_obstacles,
                'noisy_ags': False,
                # 'noisy_ags': True, # TEST FIRST WITH THE STATIC AGENT STATES CASE
                'ags_std': 0.01,
                'angle_range': math.pi/6,
                'obst_min_x': 500.,
                'obst_max_x': 1000.,
                'obst_min_y': 250.,
                'obst_max_y': 500.
                },
            'sampler': {
                'sample_method': 'const_sampler', # TESTING FIRST THE CONSTANT ACTIONS CASE
                'device': device,
                'num_parallel': args.num_parallel,
                'num_agents': args.num_agents,
            }
        }

    elif args.sampler_num == 0:
        params = { # NOTE: THIS ONE IS FOR ACCELERATION TESTING
            'init': {
                'init_method': 'mock_init',
                'mock_states': [
                    [
                    [550., 100., 0., 1., 0.],
                    [750., 100., 0., 1., 0.],
                    [950., 100., 0., 1., 5.]
                    ],
                    [
                    [550., 100., 0., 1., 0.],
                    [750., 100., 0., 1., 0.],
                    [950., 100., 0., 1., 5.]
                    ]],
                'mock_obstacles': [
                    [
                    [1400., 375.],
                    ],
                    [
                    [1400., 375.],
                    ]], # NOTE: only one obstacle per parallel env (for now)
                'mock_target': [
                    [
                    [1400., 700.],
                    ],
                    [
                    [1400., 700.],
                    ]],
                'device': device,
            },
            'sampler': {
                'sampler_num': 0,
                'sample_method': 'mock_sampler',
                'actions':
                    [
                    [[0., 5.], [0., 0.1], [0., -0.05]],
                    [[0., 5.], [0., 0.1], [0., -100.]]
                    ],
                'device': device,
                'max_step': args.max_step,
            }
        }

    elif args.sampler_num == 1:
        params = { # NOTE: THIS ONE IS FOR REWARD TESTING
            'init': {
                'init_method': 'mock_init',
                'mock_states': [
                    [
                    [750. -300./math.sqrt(3), 375., 0., 1., 3./math.sin(math.pi/3)],
                    [750., 375., 0., 1., 3.],
                    [750. +300./math.sqrt(3), 375., 0., 1., 3./math.sin(math.pi/3)]
                    ],
                    [
                    [450, 675., 1., 0., 2*300.*math.sin(math.radians(0.9))],
                    [750., 675., 0., -1., 6.],
                    [1050., 675., -1., 0., 2*300.*math.sin(math.radians(0.9))]
                    ]],
                'mock_obstacles': [
                    [
                    [900., 475.]
                    ], # NOTE: only one obstacle per parallel env (for now)
                    [
                    [750., 75.]
                    ]],
                'mock_target': [
                    [
                    [750., 675.]
                    ],
                    [
                    [750., 475.]
                    ]],
                'device': device,
            },
            'sampler': {
                'sampler_num': 1,
                'sample_method': 'mock_sampler',
                'actions':
                    [
                    [[0.,0.], [0., 0.], [0., 0.]],
                    [[-math.radians(1.8), 0.], [0., 0.], [math.radians(1.8), 0.]]
                    ],
                'device': device,
                'max_step': args.max_step,
            }
        }

    else:
        raise NotImplementedError

#     params = {
#         'init': {
#             'init_method': 'mock_init',
#             'mock_states': [
#                 [
#                 [750. -300./math.sqrt(3), 375., 0., 1., 3./math.sin(math.pi/3)],
#                 [750., 375., 0., 1., 3.],
#                 [750. +300./math.sqrt(3), 375., 0., 1., 3./math.sin(math.pi/3)]
#                 ],
#                 [
#                 [450, 675., 1., 0., 2*300.*math.sin(math.radians(0.9))],
#                 [750., 675., 0., -1., 6.],
#                 [1050., 675., -1., 0., 2*300.*math.sin(math.radians(0.9))]
#                 ]],
#             'mock_obstacles': [
#                 [
#                 [750., 475.]
#                 ], # NOTE: only one obstacle per parallel env (for now)
#                 [
#                 [750., 75.]
#                 ]],
#             'mock_target': [
#                 [
#                 [750., 675.]
#                 ],
#                 [
#                 [1050, 75.]
#                 ]],
#             'device': device,
#         },
#         'sampler': {
#             'sample_method': 'mock_sampler',
#             'actions':
#                 [
#                 [0., 0., 0.],
#                 [-math.radians(1.8), 0., math.radians(1.8)]
#                 ],
#             'device': device,
#             'max_step': args.max_step,
#         }
#     }
#     ############################################################################

    # ### NOTE: This section should be temporary or refactored to a JSON-file ####
    # params = {
    #     'init': {
    #         'init_method': 'mock_init',
    #         'mock_states': [
    #             [
    #             [550., 100., 0., 1., 3.],
    #             [750., 100., 0., 1., math.sqrt(2)*3.],
    #             [950., 100., 0., 1., math.sqrt(2)*3.]
    #             ],
    #             [
    #             [750., 675., 1., 0., 2*300.*math.sin(math.radians(0.9))],
    #             [750., 575., 1., 0., 2*200.*math.sin(math.radians(0.9))],
    #             [750., 475., 1., 0., 2*100.*math.sin(math.radians(0.9))]
    #             ]],
    #         'mock_obstacles': [
    #             [
    #             [550., 375.]
    #             ], # NOTE: only one obstacle per parallel env (for now)
    #             [
    #             [750., 675.]
    #             ]],
    #         'mock_target': [
    #             [
    #             [550., 700.]
    #             ],
    #             [
    #             [750., 75.]
    #             ]],
    #         'device': device,
    #     },
    #     'sampler': {
    #         'sample_method': 'mock_sampler',
    #         'actions':
    #             [
    #             [0., 0., 0.],
    #             [-math.radians(1.8), -math.radians(1.8), -math.radians(1.8)]
    #             ],
    #         'device': device,
    #         'max_step': args.max_step,
    #     }
    # }
    # ############################################################################

################################################################################
# MAYBE REFACTOR THIS SECTION TO A FUNCTION WHICH IS DEFINED IN utils.py? ######
# ALSO THE INIT PARAMS ABOVE SHOULD BE MOVED THERE & MAYBE READ FROM A FILE ####

    max_dist = math.sqrt(args.max_x_value**2 + args.max_y_value**2)

    min_obs = [-math.pi, 0.] # target_angle & target_distance
    min_obs += args.num_obstacles * [-math.pi]# obstacles_angles
    min_obs += args.num_obstacles * [0.] # obstacles_distances
    min_obs += (args.num_agents -1) * [-math.pi] # others_angles
    min_obs += (args.num_agents -1) * [0.] # others_distances

    max_obs = [math.pi, max_dist] # target_angle & target_distance
    max_obs += args.num_obstacles * [math.pi]# obstacles_angles
    max_obs += args.num_obstacles * [max_dist] # obstacles_distances
    max_obs += (args.num_agents -1) * [math.pi] # others_angles
    max_obs += (args.num_agents -1) * [max_dist] # others_distances

    normalizer_params = {
        'device': device,
        'num_agents': args.num_agents,
        'min_obs': min_obs,
        'max_obs': max_obs,
        }

    scaler_params = {
        'device': device,
        'num_agents': args.num_agents,
        'min_action': [-math.pi, args.min_accel],
        'max_action': [math.pi, args.max_accel],
        }

    obs_size = 12 # NOTE: THIS MAY CHANGE IN THE FUTURE !
              #(for example if velocity differences are added to observations)
    model_params = {
        'actor': {
            'input_size': obs_size,
            'hidden_size': args.hidden_size,
        },
        'critic': {
            'input_size': obs_size * args.num_agents,
            'hidden_size': args.hidden_size,
        },
        'num_agents': args.num_agents,
        'device': device,
        'lr': args.learning_rate,
        'ent_const': args.ent_const,
        'epsilon': args.epsilon,
        'gamma': args.gamma,
        'num_total': args.num_total,
        'num_parallel': args.num_parallel,
        'buffer_len': args.buffer_len,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'action_size': 2,
        'normalizer': normalizer_params,
        'scaler': scaler_params,
        }

    params = {
        'model': model_params,
        'animation': {
            'size_x': args.fig_size_x,
            'size_y': args.fig_size_y,
            'x_max': args.max_x_value,
            'y_max': args.max_y_value,
            'num_agents': args.num_agents,
            'action_size': 2,
            'parallel_index': args.parallel_index,
            'agent_index': args.agent_index, # NOTE: USED ONLY FOR REWARDS PLOTTING
            'sampling_style': args.sampling_style,
            'random': args.random,
            'weights_file': args.weights_file,
            'max_step': args.max_step,
            'interval': args.interval,
            'normalizer': normalizer_params,
            'scaler': scaler_params,
        },
        'env': {
            'device': device,
            'num_parallel': args.num_parallel,
            'num_agents': args.num_agents,
            'num_obstacles': args.num_obstacles,
            'x_bound': args.max_x_value,
            'y_bound': args.max_y_value,
            'max_step': args.max_step,
            'episode_len': args.episode_len,
            'min_speed': args.min_speed,
            'max_speed': args.max_speed,
            'min_accel': args.min_accel,
            'max_accel': args.max_accel,
            'risk_factor': args.risk_factor,
            'distance_factor': args.distance_factor,
            'heading_factor': args.heading_factor,
            'target_factor': args.target_factor,
            'soft_factor': args.soft_factor,
            'sampler': params['sampler'],
            'init': params['init'],
        },
    }
################################################################################

    if args.seed is not None:
        set_all_seeds(args.seed)

    if args.rendering:
        mode = 'rendering'
    elif args.reward_check:  # NOTE: FOR TESTING. REMOVE LATER ?
        # params = load_config(...) # NOTE: CREATE THE STATE TEST CONFIG FILE.
        mode = 'reward_check'
    else:
        mode = 'training'

    main(params, mode)
