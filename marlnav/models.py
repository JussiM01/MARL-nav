import csv
import json
import os
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from datetime import datetime
from marlnav.utils import ActionScaler, ObsNormalizer


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
        std = nn.functional.softplus(self.fc_std(x))
        sigma = torch.vmap(torch.diag)(std) # NOTE: vmap is needed since std is a batch of vectors
        # NOTE: MAKE SURE SIGMA IS POSITVE (for example add 1e-12 to it?)
        dist = MultivariateNormal(mu, sigma)

        return dist


class Critic(nn.Module):
    """Critic model for the multi-agent PPO algorithm."""

    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
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
        self.num_parallel = params['num_parallel']
        self.action_size = params['action_size']
        self.device = params['device']
        self.env = env
        self.obs = None
        self.actor = Actor(**params['actor']).to(self.device)
        self.critic = Critic(**params['critic']).to(self.device)
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=params['lr'], maximize=True)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=params['lr'], maximize=False)
        self.ent_const = params['ent_const']
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.buffer_len = params['buffer_len']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']
        self.buffer = []
        self._normalize = ObsNormalizer(params['normalizer'])
        self._scale_up = ActionScaler(params['scaler'])
        self._wpath = os.path.join(os.getcwd(), 'weights')
        self._ppath = os.path.join(os.getcwd(), 'plots')
        self._lpath = os.path.join(os.getcwd(), 'logs')
        os.makedirs(self._wpath, exist_ok=True)
        os.makedirs(self._ppath, exist_ok=True)
        os.makedirs(self._lpath, exist_ok=True)
        self._time = datetime.now().strftime('%Y%m%d%H%M%S')
        self._actor_path = os.path.join(self._wpath, self._time + '_actor.pt')
        self._critic_path = os.path.join(self._wpath, self._time + '_critic.pt')
        self._max_rew = float("-inf")
        self._mean_rew = 0.
        self._logs = {
            'epi_stats': {
                'trunc': [],
                'col': [],
                'tar':[],
            },
            'mean_rews': [],
            'actor': [],
            'critic': [],
            }

    @torch.no_grad()
    def get_data(self):

        self.buffer = []
        self.obs = self._normalize(self.env.observations()) # set the inital observations
        for j in range(self.buffer_len):
            print('step', j+1) # NOTE: FOR DEBUGGING. CHOOSE A BETTER PROGESS LOGGING FOR ACTUAL USE
            dist = self.actor(self.obs)
            actions = dist.sample() # sampled actions should have values form -1. to 1.
            log_probs = dist.log_prob(actions) # (or atleast most likely be in that range)
            actions = actions.view(-1, self.num_agents, self.action_size) # sampled actions are used in training
            scaled_actions = self._scale_up(actions) # up scaled actions (with true scale) are used by the env
            new_obs, rewards, terminated, truncated = self.env.step(scaled_actions) # MAYBE CLIPPING IS NEEDED ?
            done = torch.logical_or(terminated, truncated) # (since actions are sampled from gaussian dist)
            values = self.critic(self.obs)
            self.buffer += [[self.obs, actions, log_probs, values, rewards, done]]
            self.obs = self._normalize(new_obs) # only normalized observations (-1. to 1.) are used everywhere

        self._process_rewards()
        self._update_epi_stats()

        if self._mean_rew > self._max_rew:
            torch.save(self.actor.state_dict(), self._actor_path)
            torch.save(self.critic.state_dict(), self._critic_path)

    def _process_rewards(self):

        curr_rew = torch.zeros([self.num_parallel], dtype=float).to(self.device)
        # Changing the rewards to cummulative rewards in a backward loop:
        for i in range(self.buffer_len - 1, -1, -1):
            rew, done = self.buffer[i][-2], self.buffer[i][-1]
            curr_rew = torch.where(done, 0., rew + self.gamma * curr_rew)
            self.buffer[i][-2] = curr_rew

        std, mean_rew = torch.std_mean(
            torch.cat([self.buffer[i][-2] for i in range(self.buffer_len)]))

        for i in range(self.buffer_len): # Normalizing the rewards
            self.buffer[i][-2] = (self.buffer[i][-2] - mean_rew) / (std + 1e-12)

        self._mean_rew = mean_rew
        print('MEAN_REW', mean_rew.item())
        self._logs['mean_rews'] += [mean_rew.item()]


    def _update_epi_stats(self):

        self._logs['epi_stats']['trunc'] += [self.env._num_trunc]
        self._logs['epi_stats']['col'] += [self.env._num_col]
        self._logs['epi_stats']['tar'] += [self.env._num_tar]
        self.env._num_trunc = 0
        self.env._num_col = 0
        self.env._num_tar = 0

    def train_actor(self):

        print('Training the actor ({0} epochs).\n'.format(self.num_epochs))
        for i in range(self.num_epochs):
            print('Epoch {0}.\n'.format(i+1))
            for j in range(self.buffer_len // self.batch_size):
                print('BATCH', j+1) # NOTE: FOR DEBUGGING. CHOOSE A BETTER PROGESS LOGGING FOR ACTUAL USE
                start = j * self.batch_size
                if start + self.batch_size < self.buffer_len:
                    end = start + self.batch_size
                else:
                    end = -1
                mini_batch = self.buffer[start:end]
                self.actor_optimizer.zero_grad()
                loss = self._actor_loss(mini_batch)
                loss.backward()
                self.actor_optimizer.step()
                print('ACTOR LOSS', loss.item()) # NOTE: FOR DEBUGGING. CHOOSE A BETTER PROGESS LOGGING FOR ACTUAL USE
                self._logs['actor'] += [loss.item()]

    def train_critic(self):

        print('Training the critic ({0} epochs).\n'.format(self.num_epochs))
        for i in range(self.num_epochs):
            print('Epoch {0}.\n'.format(i+1))
            for j in range(self.buffer_len // self.batch_size):
                print('BATCH', j+1) # NOTE: FOR DEBUGGING. CHOOSE A BETTER PROGESS LOGGING FOR ACTUAL USE
                start = j * self.batch_size
                if start + self.batch_size < self.buffer_len:
                    end = start + self.batch_size
                else:
                    end = -1
                mini_batch = self.buffer[start:end]
                self.critic_optimizer.zero_grad()
                loss = self._critic_loss(mini_batch)
                loss.backward()
                self.critic_optimizer.step()
                print('CRITIC LOSS', loss.item()) # NOTE: FOR DEBUGGING. CHOOSE A BETTER PROGESS LOGGING FOR ACTUAL USE
                self._logs['critic'] += [loss.item()]

    def save_stats(self, full_params):

        rew_file = os.path.join(self._ppath, self._time + '_mean_rews.png')
        act_file = os.path.join(self._ppath, self._time + '_act_loss.png')
        cri_file = os.path.join(self._ppath, self._time + '_cri_loss.png')
        epi_file = os.path.join(self._ppath, self._time + '_epi_stats.png')

        self._create_plot(
            self._logs['mean_rews'], 'rollot_num', 'Mean Rewards', rew_file)
        self._create_plot(
            self._logs['actor'], 'batch_num', 'Actor Losses', act_file)
        self._create_plot(
            self._logs['critic'], 'batch_num', 'Critic Losses', cri_file)

        par_file = os.path.join(self._lpath, self._time + '_params.json')

        with open(par_file, 'w') as f:
            json.dump(full_params, f, indent=4, sort_keys=True)

        rew_logfile = os.path.join(self._lpath, self._time + '_mean_rews.csv')
        act_logfile = os.path.join(self._lpath, self._time + '_act_loss.csv')
        cri_logfile = os.path.join(self._lpath, self._time + '_cri_loss.csv')
        epi_logfile = os.path.join(self._lpath, self._time + '_epi_stats.csv')

        self._create_logfile(
            [[num] for num in self._logs['mean_rews']], rew_logfile)
        self._create_logfile(
            [[num] for num in self._logs['actor']], act_logfile)
        self._create_logfile(
            [[num] for num in self._logs['critic']], cri_logfile)

        self._save_epi_stats(epi_file, epi_logfile)

    def _create_plot(self, stats, xlabel, title, filename):

        fig, ax = plt.subplots(1, 1)
        ax.set(xlabel=xlabel, ylabel='value')
        ax.plot(stats)
        fig.suptitle(title)
        fig.savefig(filename)

    def _save_epi_stats(self, plotfile, logfile):

        fig, ax = plt.subplots(1, 1)
        ax.set(xlabel='rollout', ylabel='value')
        ax.plot(self._logs['epi_stats']['trunc'], color='blue',
            label='truncated')
        ax.plot(self._logs['epi_stats']['col'], color='red',
            label='collisions')
        ax.plot(self._logs['epi_stats']['tar'], color='green',
            label='target reached')
        ax.legend()
        fig.suptitle('Episode endings')
        fig.savefig(plotfile)

        num_rows = len(self._logs['epi_stats']['trunc']) # NOTE: MAYBE ADD self.num_repeats PARAM ?
        with open(logfile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Truncated', 'Collisions', 'Target reached'])
            writer.writerows([[self._logs['epi_stats']['trunc'][i],
                self._logs['epi_stats']['col'][i],
                self._logs['epi_stats']['tar'][i]] for i in range(num_rows)])

    def _create_logfile(self, value_list, filename):

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Value'])
            writer.writerows(value_list)

    def _actor_loss(self, mini_batch):

        size = len(mini_batch)
        obs = torch.cat([mini_batch[i][0] for i in range(size)], dim=0)
        actions = torch.cat([mini_batch[i][1] for i in range(size)], dim=0)
        log_probs = torch.cat([mini_batch[i][2] for i in range(size)], dim=0)
        values = torch.cat([mini_batch[i][3] for i in range(size)], dim=0)
        rewards = torch.cat([mini_batch[i][4] for i in range(size)], dim=0)

        dist = self.actor(obs)
        actions = actions.view(
            self.num_parallel * self.num_agents * size, self.action_size)
        new_log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        rewards = rewards.repeat(self.num_agents)
        values = torch.squeeze(values).repeat(self.num_agents)
        advantages = rewards - torch.squeeze(values)

        margin = self.epsilon # NOTE: IS ANNEALING NEEDED & SHOULD THIS BE A DIFFERENT EPSILON ?
        # margin = self.epsilon * annealing # NOTE: WHERE THESE COME ?! (should this be differnt epsilon?)
        ratios = torch.exp(new_log_probs - log_probs)

        clip_loss = torch.mean(
            torch.minimum(ratios * advantages,
                torch.clip(ratios, 1 - margin, 1 + margin) * advantages)
            )
        entropy_loss = torch.mean(entropies)

        return clip_loss + self.ent_const * entropy_loss

    def _critic_loss(self, mini_batch):

        size = len(mini_batch)
        obs = torch.cat([mini_batch[i][0] for i in range(size)], dim=0)
        values = torch.cat([mini_batch[i][3] for i in range(size)], dim=0)
        rewards = torch.cat([mini_batch[i][4] for i in range(size)], dim=0)

        values = torch.squeeze(values)
        new_values = torch.squeeze(self.critic(obs))
        diff = (new_values - rewards)**2
        clamped = torch.clamp(new_values, min=(values - self.epsilon), # CHECK THAT THIS IS THE RIGHT EPSILON !
            max=(values + self.epsilon))
        clamped_diff = (clamped - rewards)**2
        critic_loss = torch.mean(torch.maximum(diff, clamped_diff))

        return critic_loss
