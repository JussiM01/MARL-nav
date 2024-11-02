import argparse

from marlnav.animation import Animation
from marlnav.model import DynamicsModel


def main(params, mode):

    dynamics_model = DynamicsModel(params['model'])

    if mode == 'rendering':
        renderer = Animation(dynamics_model, params['animation'])
        renderer.run()

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
    parser.add_argument('-sa', '--size_agents', type=int, default=10,
        help='size of the agents in the animation')

    # model args
    parser.add_argument('-ba', '--batch_size', type=int,
        help='number of enviroments in the batch')
    parser.add_argument('-nb', '--num_agents', type=int, default=300,
        help='number of agents in a single environment')

    parser.add_argument('-mo', '--mode', type=str, defult='rendering', # NOTE: Change this to something else
        help='Mode of the enviroment: `rendering` or `training`')

    args = parser.parse_args()

    ### NOTE: This section should be temporary or refactored to a JSON-file ####
    mock_params = {
        'init': {
            'init_method': 'mock_init',
            'mock_states': [
                [
                [550., 100., 0., 1., 3.],
                [750., 100., 0., 1., 3.],
                [950., 100., 0., 1., 3.]
                ],
                [
                [750., 675., 1., 0., 300.*math.sin(math.radians(0.45))],
                [750., 575., 1., 0., 200.*math.sin(math.radians(0.45))],
                [750., 475., 1., 0., 100.*math.sin(math.radians(0.45))]
                ]],
            'mock_obstacles': [
                [
                [550., 375.]
                ], # only one obstacle per batch (for now)
                [
                [750., 475.]
                ]],
            'mock_target': [
                [
                [550., 700.]
                ],
                [
                [750., 475.]
                ]],
        },
        'sampler': {
            'sample_method': 'mock_sampler',
            'angles':
                [
                [0., 0.01, 0.5 * 0.01], # NOTE: EXPERIMENT WITH THE VALUES!
                [math.radians(0.9) * math.radians(0.9), math.radians(0.9)]
                ],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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
        },
        'init': mock_params['init'],
        'model': {
            'batch_size': batch_size,
            'num_agents': args.num_agents,
            'x_bound': args.max_x_value,
            'y_bound': args.max_y_value,
            'max_step': args.max_step,
        },
        'sampler': mock_params['sampler'],
    }

    if args.rendering:
        mode = 'rendering'
    else:
        mode = 'training'

    main(params, mode)
