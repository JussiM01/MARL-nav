import torch

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
    elif:
        raise NotImplementedError


class MockSampler(object):
    """Mock sampler for testing the dynamics model and its visualization."""

    def __init__(self, params):
        (angle00, angle01, angle02) = params['angles'][0]
        (angle10, angle11, angle12) = params['angles'][1]
        device = params['device']

        self.angle_batch = (torch.tensor(
            [[angle00*(-1)*i, angle01*(-1)*i, angle01*(-1)*i],
            [angle10, angle11, angle12]]).to(device)
            for i in range(params['max_step']))

    def __call__(self):
        return next(self.angle_batch)


def action_sampler(params):
    """Samples a random action batch."""
    if params['sample_method'] == 'mock_sampler':
        return MockSampler(params)
    elif:
        raise NotImplementedError


def init_animation(params, agents_pos, obstacles_pos, target_pos):

    fig = plt.figure(figsize=(params['size_x'], params['size_y']))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, params['x_max']), ax.set_xticks([])
    ax.set_ylim(0, params['y_max']), ax.set_yticks([])

    agents_scatter = ax.scatter(agents_pos[:, 0], agents_pos[:, 1], # NOTE: CHECK the drawing order (should this be last?)
        s=params['size_agents'], lw=0.5, c=np.array([params['color_agents']]))
    obs_scatter = ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1],
        s=params['size_obs'], lw=0.5, c=np.array([params['color_obs']]))
    target_scatter = ax.scatter(target_pos[:, 0], target_pos[:, 1], # NOTE: CHEKC DIMS! this is only sngle position
        s=params['size_target'], lw=0.5, c=np.array([params['color_target']]))

    return fig, agents_scatter, obs_scatter, target_scatter


mock_params = {
    'init': {
        'mock_states': [
            [
            [pos00x, pos00y, dir00x, dir00y, speed00],
            [pos01x, pos01y, dir01x, dir01y, speed01],
            [pos02x, pos02y, dir02x, dir02y, speed02]
            ],
            [
            [pos10x, pos10y, dir10x, dir10y, speed10],
            [pos11x, pos11y, dir11x, dir11y, speed11],
            [pos12x, pos12y, dir12x, dir12y, speed12]
            ]],
        'mock_obstacles': [
            [
            [pos00x, pos00y]
            ], # only one obstacle per batch (for now)
            [
            [pos10x, pos10y]
            ]],
        'mock_target': [
            [
            [pos00x, pos00y]
            ],
            [
            [pos10x, pos10y]
            ]],
    },
    'sampler': {
        'angles':
            [
            [angle00, angle01, angle02],
            [angle10, angle11, angle12]
            ],
    'device': device,
    }
}
