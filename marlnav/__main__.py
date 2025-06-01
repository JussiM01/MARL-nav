import argparse
import math
import torch



from marlnav.animation import Animation
from marlnav.dynamics import DynamicsModel
from marlnav.utils import load_config, plot_states_and_rews # NOTE: LAST ONE IS FOR TESTING. REMOVE LATER ?


def main(params, mode):

    dynamics_model = DynamicsModel(params['model'])

    if mode == 'rendering':
        renderer = Animation(dynamics_model, params['animation'])
        renderer.run()

    elif mode == 'plot_saving': # NOTE: FOR TESTING. REMOVE LATER ?
        plot_states_and_rews(
            dynamics_model,
            params['animation']['max_step'],
            params['animation']['batch_index'],
            params['animation']['agent_index']
            )

    elif mode == 'training':
        raise NotImplementedError
        # for i in range(params['num_steps']):
        #     dynamics_model.update()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('-mx', '--max_x_value', type=float, default=1500.0,
        help='maximum value for the x-coordinates')
    parser.add_argument('-my', '--max_y_value', type=float, default=750.0,
        help='maximum value for the y-coordinates')

    # animation/plot args
    parser.add_argument('-fx', '--fig_size_x', type=float, default=10.0,
        help='animation plot width in centimeters')
    parser.add_argument('-fy', '--fig_size_y', type=float, default=5.0,
        help='animation plot height in centimeters')
    parser.add_argument('-sia', '--size_agents', type=int, default=10,
        help='size of the agents in the animation')
    parser.add_argument('-sio', '--size_obstacles', type=int, default=100,
        help='size of the obstacle in the animation')
    parser.add_argument('-sit', '--size_target', type=int, default=100,
        help='size of the target in the animation')
    parser.add_argument('-bi', '--batch_index', type=int, default=0, # NOTE: CHANGE LATER?
        help='index of the rendered environment in the batch')
    parser.add_argument('-ai', '--agent_index', type=int, default=0, # NOTE: CHANGE LATER?
        help='index of the agent for whose rewards are plotted')
    parser.add_argument('-in', '--interval', type=int, default=10,
        help='interval param for the animation (small is fast).')

    # model args
    parser.add_argument('-ba', '--batch_size', type=int, default=2, # NOTE: DEFAULT=2 FOR TESTING, change this later?
        help='number of enviroments in the batch')
    parser.add_argument('-na', '--num_agents', type=int, default=3,
        help='number of agents in a single environment')
    parser.add_argument('-no', '--num_obstacles', type=int, default=1, # NOTE: DEFAULT=1 FOR TESTING, change this later?
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

    # init args
    parser.add_argument('-re', '--rendering', action='store_true',
        help='rendering option (no training), action: store_true' )
    parser.add_argument('-sa', '--sampling_style', type=str, default='sampler', # NOTE: FOR TESTING
        help='sampling style, should be either `policy` or `sampler`') # REMOVE THIS LATER ?

    parser.add_argument('-pl', '--plot_saving', action='store_true', # NOTE: FOR DEBUGGING/TESTING ONLY!
        help='Run test for states and rewards and save plots') # REMOVE THIS LATER ?
    parser.add_argument('-sn', '--sampler_num', type=int, default=0, # NOTE: FOR DEBUGGING/TESTING ONLY!
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
                'batch_size': args.batch_size,
                'ags_cent_x': 150.,
                'ags_cent_y': 375.,
                'ags_dist': 20.,
                'tar_pos_x': 1350.,
                'tar_pos_y': 375.,
                'num_obs': 10,
                'noisy_ags': False,
                # 'noisy_ags': True, # TEST FIRST WITH THE STATIC AGENT STATES CASE
                'ags_std': 0.01,
                'angle_range': math.pi/6,
                'obst_min_x': 300.,
                'obst_max_x': 1200.,
                'obst_min_y': 150.,
                'obst_max_y': 600.
                },
            'sampler': {
                'sample_method': 'const_sampler', # TESTING FIRST THE CONSTANT ACTIONS CASE
                'device': device,
                'batch_size': args.batch_size,
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
                    ]], # NOTE: only one obstacle per batch (for now)
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
                    ], # NOTE: only one obstacle per batch (for now)
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
#                 ], # NOTE: only one obstacle per batch (for now)
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
    #             ], # NOTE: only one obstacle per batch (for now)
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

    params = {
        'animation': {
            'size_x': args.fig_size_x,
            'size_y': args.fig_size_y,
            'x_max': args.max_x_value,
            'y_max': args.max_y_value,
            'size_agents': args.size_agents,
            'color_agents': (0, 0, 0, 1),
            'size_obstacles': args.size_obstacles,
            'color_obstacles': (1, 0, 0, 1), # NOTE: FIX A PROPER VALUE!
            'size_target': args.size_target,
            'color_target': (0, 1, 0, 1), # NOTE: FIX A PROPER VALUE!
            'batch_index': args.batch_index,
            'agent_index': args.agent_index, # NOTE: USED ONLY FOR REWARDS PLOTTING
            'sampling_style': args.sampling_style,
            'max_step': args.max_step,
            'interval': args.interval,
        },
        'model': {
            'device': device,
            'batch_size': args.batch_size,
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

    if args.rendering:
        mode = 'rendering'
    elif args.plot_saving:  # NOTE: FOR TESTING. REMOVE LATER ?
        # params = load_config(...) # NOTE: CREATE THE STATE TEST CONFIG FILE.
        mode = 'plot_saving'
    else:
        mode = 'training'

    main(params, mode)
