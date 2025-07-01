import torch

from torch import nn
from torch.distributions import MultivariateNormal


class Actor(nn.Module):
    """Actor model for the multi-agent PPO algorithm."""

    def __init__(self, input_size, hidden_size):
        super(Actor, self).__init__()
        self.flatten = nn.Flatten(start_dim=0, end_dim=1)
        self.fc1   = nn.Linear(input_size, hidden_size)
        torch.nn.init.orthogonal_(self.fc1.weight)
        self.fc_mu = nn.Linear(hidden_size, 2)
        torch.nn.init.orthogonal_(self.fc_mu.weight)
        self.fc_std = nn.Linear(hidden_size, 2)
        torch.nn.init.orthogonal_(self.fc_std.weight)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        sigma = torch.vmap(torch.diag)(std) # NOTE: vmap is needed since std is a batch of vectors
        # NOTE: MAKE SURE SIGMA IS POSITVE (for example add 1e-8 to it?)
        dist = MultivariateNormal(mu, sigma)
        actions = dist.sample()
        log_props = dist.log_prob(actions)

        return actions.view((-1, 3, 2)), log_probs.view((-1, 3))


class Critic(nn.Module):
    """Critic model for the multi-agent PPO algorithm."""

    def __init__(self, input_size, hidden_size):
        super(Actor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        torch.nn.init.orthogonal_(self.fc1.weight)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        torch.nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        value = self.fc2(x)

        return value


class PPO(object):
    """Multi-agent PPO model with separate actor and critic models."""

    def __init__(self, params):
        self.actor = Actor(**params['actor'])
        self.critic = Critic(**params['critic'])
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.max_rew = float("-inf")
        self.buffer = []

    def process_rewards(self):

        raise NotImplementedError

    def train_actor(self):

        raise NotImplementedError

    def train_critic(self):

        raise NotImplementedError

    def plot_results(self):

        raise NotImplementedError

    def _actor_loss(self): # NOTE: ADD THE PARAMETERS !

        raise NotImplementedError

    def _critic_loss(self): # NOTE: ADD THE PARAMETERS !

        raise NotImplementedError
