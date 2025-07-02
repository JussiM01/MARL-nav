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

        return dist


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


class MAPPO(object):
    """Multi-agent PPO model with separate actor and critic models."""

    def __init__(self, params):
        self.device = params['device']
        self.actor = Actor(**params['actor']).to(self.device)
        self.critic = Critic(**params['critic']).to(self.device)
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.buffer = []
        self._max_rew = float("-inf")
        self._mean_rew = 0.

    def process_rewards(self):

        curr_rew = torch.zeros([num_parallel], dtype=float).to(self.device)
        # Changing the rewards to cummulative rewards in a backward loop:
        for i in range(len(self.buffer) - 1, -1, -1):
            rew, done = self.buffer[i][-2], self.buffer[i][-1]
            curr_rew = torch.where(done, 0., rew + self.gamma * curr_rew)
            self.buffer[i][-2] = curr_rew

        std, mean_rew = torch.std_mean(
            [self.buffer[i][-2] for i in range(len(self.buffer))])

        for i in range(len(self.buffer)): # Normalizing the rewards
            self.buffer[i][-2] = (self.buffer[i][-2] - avg_rew) / (std + 1e-12)

        self._mean_rew = mean_rew

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
