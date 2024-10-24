import torch


def mock_init(params):
    """Mock intialization for testing."""
    raise NotImplementedError


def random_states(params):
    """Initializes random states of the environment."""
    if params['init_method'] == 'mock_init':
        return mock_init(params)
    elif:
        raise NotImplementedError


class MockSampler(object):
    """Mock sampler for testing the dynamics model and its visualization."""

    def __init__(self, params):
        self.params = params

    def __call__(self):
        raise NotImplementedError


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
