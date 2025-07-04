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
        # NOTE: MAKE SURE SIGMA IS POSITVE (for example add 1e-12 to it?)
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

    def __init__(self, params, env):
        self.num_agents = params['num_agents']
        self.device = params['device']
        self.env = env
        self.obs = self.env.observations # set the inital observetions
        self.actor = Actor(**params['actor']).to(self.device)
        self.critic = Critic(**params['critic']).to(self.device)
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=params['lr'], maximize=True)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=params['lr'], minimize=True)
        self.ent_const = params['ent_const']
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.buffer_len = params['buffer_len']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']
        self.buffer = []
        self._max_rew = float("-inf")
        self._mean_rew = 0.

    @torch.no_grad()
    def get_data(self):

        self.buffer = []
        for j in range(self.buffer_len):
            dist = self.actor(self.obs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            actions = actions.view(-1, num_agents, action_size)
            new_obs, rewards, terminated, truncated = env.step(actions)
            done = torch.logical_or(terminated, truncated)
            values = self.critic(obs)
            self.buffer += [obs, actions, log_probs, values, rewards, done]
            self.obs = new_obs

        ###############################################
        # Add here progress logging & model(s) saving #
        # if self._mean_rew > self._max_rew ?)        #
        ###############################################

    def process_rewards(self):

        curr_rew = torch.zeros([num_parallel], dtype=float).to(self.device)
        # Changing the rewards to cummulative rewards in a backward loop:
        for i in range(self.buffer_len - 1, -1, -1):
            rew, done = self.buffer[i][-2], self.buffer[i][-1]
            curr_rew = torch.where(done, 0., rew + self.gamma * curr_rew)
            self.buffer[i][-2] = curr_rew

        std, mean_rew = torch.std_mean(
            [self.buffer[i][-2] for i in range(self.buffer_len)])

        for i in range(self.buffer_len): # Normalizing the rewards
            self.buffer[i][-2] = (self.buffer[i][-2] - avg_rew) / (std + 1e-12)

        self._mean_rew = mean_rew

    def train_actor(self):

        print('Training the actor ({0} epochs).\n'.format(self.num_epochs))
        for i in range(self.num_epochs):
            print('Epoch {0}.\n'.format(j+1))
            for j in range(self.buffer_len // self.batch_size):
                start = j * batch_size
                if start + self.batch_size < self.buffer_len:
                    end = start + self.batch_size
                else:
                    end = -1
                mini_batch = self.buffer[start:end]
                self.actor_optimizer.zero_grad()
                loss = self._actor_loss(mini_batch)
                loss.backward()
                self.actor_optimizer.step()

    def train_critic(self):

        print('Training the critic ({0} epochs).\n'.format(self.num_epochs))
        for i in range(self.num_epochs):
            print('Epoch {0}.\n'.format(j+1))
            for j in range(self.buffer_len // self.batch_size):
                start = j * batch_size
                if start + self.batch_size < self.buffer_len:
                    end = start + self.batch_size
                else:
                    end = -1
                mini_batch = self.buffer[start:end]
                self.critic_optimizer.zero_grad()
                loss = self._critic_loss(mini_batch)
                loss.backward()
                self.critic_optimizer.step()

    def plot_results(self):

        raise NotImplementedError

    def _actor_loss(self, mini_batch):

        size = len(mini_batch)
        obs = torch.cat([batch[i][0] for i in range(size)], dim=0)
        actions = torch.cat([batch[i][1] for i in range(size)], dim=0)
        log_probs = torch.cat([batch[i][2] for i in range(n)], dim=0)
        values = torch.cat([batch[i][3] for i in range(size)], dim=0)
        rewards = torch.cat([mini_batch[i][4] for i in range(size)], dim=0)

        dist = ppo.actor(obs)
        new_log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        advantages = rewards - values
        advantages = advantages.repeat(self.num_agents)

        margin = self.epsilon # NOTE: IS ANNEALING NEEDED & SHOULD THIS BE A DIFFERENT EPSILON ?
        # margin = self.epsilon * annealing # NOTE: WHERE THESE COME ?! (should this be differnt epsilon?)
        ratios = torch.exp(new_log_probs - log_probs)

        catenated = torch.cat([ratios * advantages,
            torch.clip(ratios, 1 - margin, 1 + margin) * advantages)], dim=1)
        clip_loss = torch.mean(torch.min(catenated, dim=1))
        entropy_loss = torch.mean(entropies)

        return clip_loss + self.ent_const * entropy_loss

    def _critic_loss(self, mini_batch):

        size = len(mini_batch)
        obs = torch.cat([mini_batch[i][0] for i in range(size)], dim=0)
        values = torch.cat([mini_batch[i][3] for i in range(size)], dim=0)
        rewards = torch.cat([mini_batch[i][4] for i in range(size)], dim=0)
        new_values = self.critic(obs)

        diff = (new_values - rewards)**2
        clamped = torch.clamp(new_values, min=(values - self.epsilon), # CHECK THAT THIS IS THE RIGHT EPSILON !
            max=(values + self.epsilon))
        clamped_diff = (clamped - rewards)**2
        critic_loss = torch.mean( # NOTE: assumes that len(shape) = 1 for both
            torch.max(torch.cat([diff, clamped_diff], dim=1), dim=1))

        return critic_loss
