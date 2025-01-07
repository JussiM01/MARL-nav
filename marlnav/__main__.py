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
        # plot_states_and_rews(...)
        raise NotImplementedError

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

    # fig args
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

    # model args
    parser.add_argument('-ba', '--batch_size', type=int, default=2, # NOTE: DEFAULT=2 FOR TESTING, change this later?
        help='number of enviroments in the batch')
    parser.add_argument('-na', '--num_agents', type=int, default=3,
        help='number of agents in a single environment')
    parser.add_argument('-no', '--num_obstacles', type=int, default=10,
        help='number of obstacles in a single environment')
    parser.add_argument('-ms', '--max_step', type=int, default=200, # NOTE: DEFAULT=200 FOR TESTING, change this later?
        help='maximum number of time steps in the simulation')

    # init args
    # parser.add_argument('-re', '--rendering', action='store_true',
    #     help='rendering option (no training), action: store_true' )
    parser.add_argument('-re', '--rendering', type=bool, default=True, # NOTE: FOR DEBUGGING/TESTING ONLY!
        help='rendering option (no training)') # REPLACE LATER OR USE THE VERSION ABOVE?
    parser.add_argument('-sa', '--sampling_style', type=str, default='sampler', # NOTE: FOR TESTING
        help='sampling style, should be either `policy` or `sampler`')
    parser.add_argument('-pl', '--plot_saving', action='store_true', # NOTE: FOR DEBUGGING/TESTING ONLY!
        help='Run test for states and rewards and save plots') # REMOVE THIS LATER ?

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### NOTE: This section should be temporary or refactored to a JSON-file ####
    mock_params = {
        'init': {
            'init_method': 'mock_init',
            'mock_states': [
                [
                [550., 100., 0., 1., 3.],
                [750., 100., 0., 1., math.sqrt(2)*3.],
                [950., 100., 0., 1., math.sqrt(2)*3.]
                ],
                [
                [750., 675., 1., 0., 2*300.*math.sin(math.radians(0.9))],
                [750., 575., 1., 0., 2*200.*math.sin(math.radians(0.9))],
                [750., 475., 1., 0., 2*100.*math.sin(math.radians(0.9))]
                ]],
            'mock_obstacles': [
                [
                [550., 375.]
                ], # NOTE: only one obstacle per batch (for now)
                [
                [750., 675.]
                ]],
            'mock_target': [
                [
                [550., 700.]
                ],
                [
                [750., 75.]
                ]],
            'device': device,
        },
        'sampler': {
            'sample_method': 'mock_sampler',
            'angles':
                [
                [0., 0., 0.],
                [-math.radians(1.8), -math.radians(1.8), -math.radians(1.8)]
                ],
            'device': device,
            'max_step': args.max_step,
        }
    }
    ############################################################################

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
            'sampling_style': args.sampling_style,
            'max_step': args.max_step,
        },
        'model': {
            'device': device,
            'batch_size': args.batch_size,
            'num_agents': args.num_agents,
            'num_obstacles': args.num_obstacles,
            'x_bound': args.max_x_value,
            'y_bound': args.max_y_value,
            'max_step': args.max_step,
            'sampler': mock_params['sampler'],
            'init': mock_params['init'],
        },
    }

    if args.rendering:
        mode = 'rendering'
    elif args.plot_saving  # NOTE: FOR TESTING. REMOVE LATER ?
        # params = load_config(...) # NOTE: CREATE THE STATE TEST CONFIG FILE.
        mode = 'plot_saving'
    else:
        mode = 'training'

    main(params, mode)
