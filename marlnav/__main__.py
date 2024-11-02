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

    args = parser.parse_args()

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
        'model': {
            'batch_size': batch_size,
            'num_agents': args.num_agents,
            'x_bound': args.max_x_value,
            'y_bound': args.max_y_value,
            'max_step': args.max_step,
        },
    }

    if args.rendering:
        mode = 'rendering'
    else:
        mode = 'training'

    main(params, mode)
