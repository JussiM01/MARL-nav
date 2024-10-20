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
