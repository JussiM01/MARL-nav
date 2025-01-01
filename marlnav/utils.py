import math
import matplotlib.pyplot as plt
import numpy as np
import torch

from collections import namedtuple


Observations = namedtuple('Observations', ['target_angle', 'target_distance',
    'obstacles_angles', 'obstacles_distances', 'others_angles',
    'others_distances']) 


def mock_init(params):
    """Mock intialization for testing."""
    states = torch.tensor(params['mock_states']).to(params['device'])
    obstacles = torch.tensor(params['mock_obstacles']).to(params['device'])
    target = torch.tensor(params['mock_target']).to(params['device'])

    return states, obstacles, target


def random_states(params):
    """Initializes random states of the environment."""
    if params['init_method'] == 'mock_init':
        return mock_init(params)
    else:
        raise NotImplementedError


class MockSampler(object):
    """Mock sampler for testing the dynamics model and its visualization."""

    def __init__(self, params):
        (angle00, angle01, angle02) = params['angles'][0]
        (angle10, angle11, angle12) = params['angles'][1]
        device = params['device']

        self._angle01 = (-0.25*math.pi if i == 0
            else 0.5*math.pi*(-1)**((i//50)%2+1) if (i%50 == 0) else 0.
            for i in range(params['max_step']))
        self._angle02 = (-0.25*math.pi if i == 0
            else 0.5*math.pi*(-1)**((i//25)%2+1) if (i%25 == 0) else 0.
            for i in range(params['max_step']))
        self._angles1 = ([0.5*angle10, 0.5*angle11, 0.5*angle12] if i == 0
            else [angle10, angle11, angle12]
            for i in range(params['max_step']))
        self.angle_batch = (torch.tensor([[0., next(self._angle01),
            next(self._angle02)], next(self._angles1)]).to(device)
            for i in range(params['max_step']))

    def __call__(self):
        return next(self.angle_batch)


def action_sampler(params):
    """Samples a random action batch."""
    if params['sample_method'] == 'mock_sampler':
        return MockSampler(params)
    else:
        raise NotImplementedError


def init_animation(params, agents_pos, obstacles_pos, target_pos):

    fig = plt.figure(figsize=(params['size_x'], params['size_y']))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, params['x_max']), ax.set_xticks([])
    ax.set_ylim(0, params['y_max']), ax.set_yticks([])

    agents_scatter = ax.scatter(agents_pos[:, 0], agents_pos[:, 1], # NOTE: CHECK the drawing order (should this be last?)
        s=params['size_agents'], lw=0.5, c=np.array([params['color_agents']]))
    obs_scatter = ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1],
        s=params['size_obstacles'], lw=0.5,
        c=np.array([params['color_obstacles']]))
    target_scatter = ax.scatter(target_pos[:, 0], target_pos[:, 1], # NOTE: CHEKC DIMS! this is only sngle position
        s=params['size_target'], lw=0.5, c=np.array([params['color_target']]))

    return fig, agents_scatter, obs_scatter, target_scatter
