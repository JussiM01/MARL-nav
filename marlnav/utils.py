import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

from collections import namedtuple
from torch.distributions import MultivariateNormal


Observations = namedtuple('Observations', ['target_angle', 'target_distance',
    'obstacles_angles', 'obstacles_distances', 'others_angles',
    'others_distances'])


class MockInitializer(object):
    """Mock intializer for testing."""
    def __init__(self, params):
        self.states = torch.tensor(params['mock_states']).to(params['device'])
        self.obstacles = torch.tensor(
            params['mock_obstacles']).to(params['device'])
        self.target = torch.tensor(params['mock_target']).to(params['device'])

    def __call__(self):
        return self.states, self.obstacles, self.target


class TriangleIntitializer(object):
    """Intial state sampler for three agents and the environment."""

    def __init__(self, params):
        self.device = params['device']
        self.init_method = params['init_method']
        self.num_parallel = params['num_parallel']
        self.ags_cent_x = params['ags_cent_x']
        self.ags_cent_y = params['ags_cent_y']
        self.ags_dist = params['ags_dist']
        self.tar_pos_x = params['tar_pos_x']
        self.tar_pos_y = params['tar_pos_y']
        self.num_obs = params['num_obs']
        self.noisy_ags = int(params['noisy_ags'])
        self.ags_std = params['ags_std']
        self.angle_range = params['angle_range']
        self.obs_min_x = params['obst_min_x']
        self.obs_max_x = params['obst_max_x']
        self.obs_min_y = params['obst_min_y']
        self.obs_max_y = params['obst_max_y']

        self._obs_x_range = self.obs_max_x - self.obs_min_x
        self._obs_y_range = self.obs_max_y - self.obs_min_y
        self._obs_mean_x = 0.5 * (self.obs_min_x + self.obs_max_x)
        self._obs_mean_y = 0.5 * (self.obs_min_y + self.obs_max_y)

        pos_const = 0.5 * self.ags_dist
        ags_pos = pos_const * torch.tensor(
            [[-1/math.sqrt(3), 1.], [2/math.sqrt(3), 0.],[-1/math.sqrt(3), -1.]]
            ).to(self.device)
        mean_pos = torch.unsqueeze(
            torch.tensor([self.ags_cent_x, self.ags_cent_y]).to(self.device),
            dim=0).repeat(3,1)
        ags_pos = ags_pos + mean_pos
        ags_dir = torch.tensor([[1., 0.], [1., 0.], [1., 0.]]).to(self.device)

        self.ags_pos = torch.unsqueeze(
            ags_pos, dim=0).repeat(self.num_parallel, 1, 1)
        self.ags_dir = torch.unsqueeze(
            ags_dir, dim=0).repeat(self.num_parallel, 1, 1)
        target = torch.unsqueeze(torch.tensor(
            [self.tar_pos_x, self.tar_pos_y]).to(self.device), dim=0)
        self.target = torch.unsqueeze(
            target, dim=0).repeat(self.num_parallel, 1, 1)
        self.speeds = torch.zeros([self.num_parallel, 3, 1]).to(
            self.device)

        sigma = torch.diag(torch.tensor([self.ags_std, self.ags_std])).to(
            self.device)
        self.pos_noise = MultivariateNormal(torch.zeros(2).to(
            self.device), sigma)

    def __call__(self):
        states = self._sample_agents()
        obstacles = self._sample_obstacles()

        return states, obstacles, self.target

    def _sample_agents(self):
        pos_noise = self.ags_dist * self.pos_noise.sample((self.num_parallel, 3))
        angles = self.angle_range * (torch.rand(self.num_parallel, 3) - 0.5)
        rotated_dirs = self._rotate(self.ags_dir, self.noisy_ags * angles)
        positions = self.ags_pos + self.noisy_ags * pos_noise
        states = torch.cat([positions, rotated_dirs, self.speeds], dim=2)

        return states

    def _sample_obstacles(self):
        scaled_pos_x = self._obs_x_range * (
            torch.rand(self.num_parallel, self.num_obs, 1) - 0.5)
        scaled_pos_y = self._obs_y_range * (
            torch.rand(self.num_parallel, self.num_obs, 1) - 0.5)
        obs_pos_x = scaled_pos_x + self._obs_mean_x
        obs_pos_y = scaled_pos_y + self._obs_mean_y

        return torch.cat([obs_pos_x, obs_pos_y], dim=2).to(self.device)

    def _rotate(self, directions, angles):
        return torch.vmap(torch.vmap(self._rotate_one))(directions, angles)

    def _rotate_one(self, direction_vector, angle):
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)])]).to(self.device)

        return torch.matmul(rotation_matrix, direction_vector)


def init_sampler(params):
    """Initializes random states of the environment."""
    if params['init_method'] == 'mock_init':
        return MockInitializer(params)
    elif params['init_method'] == 'triangle':
        return TriangleIntitializer(params)


class MockSampler(object):  # NOTE: THIS ONE IS FOR ACCELERATION TESTING
    """Mock sampler for testing the dynamics model and its visualization."""

    def __init__(self, params):
        if params['sampler_num'] == 0:
            (action00, action01, action02) = params['actions'][0]
            (action10, action11, action12) = params['actions'][1]
            device = params['device']

            self.action_array = (torch.tensor([[action00, action01, action02],
                [action10, action11, action12]]).to(device)
                for i in range(params['max_step']))

        elif params['sampler_num'] == 1:
            (action00, action01, action02) = params['actions'][0]
            (action10, action11, action12) = params['actions'][1]
            device = params['device']

            self._action00 = ([-math.pi/6, 0.] if i == 0 else action00
                for i in range(params['max_step']))
            self._action02 = ([math.pi/6, 0.] if i == 0 else action02
                for i in range(params['max_step']))
            action1_half = [
                [0.5*action10[0], 0.],[0.5*action11[0], 0.],[0.5*action12[0], 0.]]
            self._actions1 = (
                action1_half if i == 0. else [action10, action11, action12]
                for i in range(params['max_step']))
            self.action_array = (torch.tensor([[next(self._action00), action01,
                next(self._action02)], next(self._actions1)]).to(device)
                for i in range(params['max_step']))

    def __call__(self):
        return next(self.action_array)

# class MockSampler(object):
#     """Mock sampler for testing the dynamics model and its visualization."""
#
#     def __init__(self, params):
#         (action00, action01, action02) = params['actions'][0]
#         (action10, action11, action12) = params['actions'][1]
#         device = params['device']
#
#         self._action01 = (-0.25*math.pi if i == 0
#             else 0.5*math.pi*(-1)**((i//50)%2+1) if (i%50 == 0) else 0.
#             for i in range(params['max_step']))
#         self._action02 = (-0.25*math.pi if i == 0
#             else 0.5*math.pi*(-1)**((i//25)%2+1) if (i%25 == 0) else 0.
#             for i in range(params['max_step']))
#         self._actions1 = ([0.5*action10, 0.5*action11, 0.5*action12] if i == 0
#             else [action10, action11, action12]
#             for i in range(params['max_step']))
#         self.action_array = (torch.tensor([[0., next(self._action01),
#             next(self._action02)], next(self._actions1)]).to(device)
#             for i in range(params['max_step']))
#
#     def __call__(self):
#         return next(self.action_array)

class ConstantSampler(object):
    """Constant action sampler for testing."""

    def __init__(self, params):
        self.actions = torch.tensor([params['num_agents']*[[0., 1.]]
            for i in range(params['num_parallel'])]).to(params['device'])

    def __call__(self):
        return self.actions


def action_sampler(params):
    """Samples a random action tensor."""
    if params['sample_method'] == 'mock_sampler':
        return MockSampler(params)
    elif params['sample_method'] =='const_sampler':
        return ConstantSampler(params)
    else:
        raise NotImplementedError


def init_animation(params, agents_pos, obstacles_pos, target_pos):

    fig = plt.figure(figsize=(params['size_x'], params['size_y']))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, params['x_max']), ax.set_xticks([])
    ax.set_ylim(0, params['y_max']), ax.set_yticks([])

    agents_scatter = ax.scatter(agents_pos[:, 0], agents_pos[:, 1], # NOTE: CHECK the drawing order (should this be last?)
        s=10, lw=0.5, c=np.array([(0., 0., 0., 1.)]))
    obs_scatter1 = ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1],
        s=2200, lw=0.5, c=np.array([(1., 0.5, 0.5, 1.)]))
    obs_scatter2 = ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1],
        s=1500, lw=0.5, c=np.array([(1., 0., 0., 1.)]))
    target_scatter = ax.scatter(target_pos[:, 0], target_pos[:, 1],
        s=2000, facecolors='w', lw=1.5, edgecolors='k', linestyle=':')

    return fig, agents_scatter, obs_scatter1, obs_scatter2, target_scatter


class ObsNormalizer(object):
    """Callable for concatenating and normalizing the observations."""

    def __init__(self, params):
        min_obs = torch.tensor(params['min_obs']).to(params['device']) # (2D vector)
        max_obs = torch.tensor(params['max_obs']).to(params['device']) # (2D vector)
        scale = 0.5 * (max_obs - min_obs)
        self.mean = 0.5 * (min_obs + max_obs)
        self.scale_tensor = torch.unsqueeze(torch.stack(
            [scale for i in range(params['num_agents'])], dim=0), dim=0)

    def __call__(self, obs):
        obs = torch.cat(obs, dim=2)
        return (obs - self.mean) / self.scale_tensor # scales the values to interval [-1., 1]


class ActionScaler(object):
    """Callable for scaling up the model output actions to correct scale."""

    def __init__(self, params):
        min_action = torch.tensor(params['min_action']).to(params['device']) # (2D vector)
        max_action = torch.tensor(params['max_action']).to(params['device']) # (2D vector)
        scale = 0.5 * (max_action - min_action)
        self.mean = 0.5 * (min_action + max_action)
        self.scale_tensor = torch.unsqueeze(torch.stack(
            [scale for i in range(params['num_agents'])], dim=0), dim=0)

    def __call__(self, actions):
        return (self.scale_tensor * actions) + self.mean


def set_all_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(filename, dir):
    """Returns cofigurtation dictonary read from a configuration file."""
    config_file = os.path.join('config_files', dir,  filename)
    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def save_plot(fig, filename, dir):
    """Saves a plot with given filename to a directory."""
    os.makedirs(dir, exist_ok=True) # NOTE: CHECK THAT THIS WORKS!

    fig.savefig(os.path.join(dir, filename))
    plt.close(fig)


def check_rews(env, num_steps, parallel_ind, agent_ind):
    """Saves test plots of the states and rewards."""
    neighbour_inds = list({0, 1, 2} - {agent_ind})
    first = neighbour_inds[0]
    second = neighbour_inds[1]

    target_angles = []
    target_distances = []
    all_obs_angels = []
    all_obs_distances = []
    angles_to_first = []
    distances_to_first = []
    angles_to_second = []
    distances_to_second = []
    rewards = []

    for i in range(num_steps):
        actions = env.sample_actions()
        obs, rew, _, _ = env.step(actions)
        # print('OBSTACLES DISTANCES: ', obs.obstacles_distances[parallel_ind,:,:])
        # print('OTHERS DISTANCES: ', obs.others_distances[parallel_ind,:,:])
        # print('TARGET DISTANCE: ', obs.target_distance[parallel_ind,:,:])
        # print('TARGET ANGLE: ', obs.target_angle[parallel_ind,:,:])
        # print('REWARDS: ', rew[parallel_ind])
        # # print('REWARDS: ', rew[parallel_ind,:]) # NOTE: USE THIS FOR DEBUGGING/TESTING NEW REWARDS
        # print('\n')
        target_angles += [obs.target_angle[parallel_ind, agent_ind,0].item()]
        target_distances += [obs.target_distance[parallel_ind, agent_ind,0].item()]
        all_obs_angels += [obs.obstacles_angles[parallel_ind, agent_ind,0].item()]
        all_obs_distances += [obs.obstacles_distances[parallel_ind, agent_ind,0].item()]
        angles_to_first += [obs.others_angles[parallel_ind, agent_ind, 0].item()]
        distances_to_first += [obs.others_distances[parallel_ind, agent_ind, 0].item()]
        angles_to_second += [obs.others_angles[parallel_ind, agent_ind, 1].item()]
        distances_to_second += [obs.others_distances[parallel_ind, agent_ind, 1].item()]
        rewards += [rew[parallel_ind].item()]
        # rewards += [rew[parallel_ind, agent_ind].item()] # NOTE: USE THIS FOR DEBUGGING/TESTING NEW REWARDS

    pi_plus = 3.5
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    axs[0, 0].plot(target_angles)
    axs[0, 0].set_title('Angle to target (rad)')
    axs[0, 0].set_ylim([-pi_plus, pi_plus])
    axs[0, 1].plot(target_distances)
    axs[0, 1].set_title('Distance to target')
    axs[1, 0].plot(all_obs_angels)
    axs[1, 0].set_title('Angle to obstacle (rad)')
    axs[1, 0].set_ylim([-pi_plus, pi_plus])
    axs[1, 1].plot(all_obs_distances)
    axs[1, 1].set_title('Distance to obstacle')
    axs[2, 0].plot(angles_to_first)
    axs[2, 0].set_title('Angle to agent {} (rad)'.format(first))
    axs[2, 0].set_ylim([-pi_plus, pi_plus])
    axs[2, 1].plot(distances_to_first)
    axs[2, 1].set_title('Distance to agent {}'.format(first))
    axs[3, 0].plot(angles_to_second)
    axs[3, 0].set_title('Angle to agent {} (rad)'.format(second))
    axs[3, 0].set_ylim([-pi_plus, pi_plus])
    axs[3, 1].plot(distances_to_second)
    axs[3, 1].set_title('Distance to agent {}'.format(second))
    fig.tight_layout(pad=5.0)

    for ax in axs.flat:
        ax.set(xlabel='step number', ylabel='value')

    fig.suptitle('States, parallel index: {0}, agent index: {1}'.format(
        parallel_ind, agent_ind))
    save_plot(fig, 'states_array_{0}_agent_{1}.png'.format(
        parallel_ind, agent_ind), 'plots')

    tar_fac = env._target_factor
    hea_fac = env._heading_factor
    dis_fac = env._distance_factor
    ris_fac = env._risk_factor
    sof_fac = env._soft_factor

    fig, ax = plt.subplots(1, 1)
    ax.set(xlabel='step number', ylabel='value')
    ax.plot(rewards)
    fig.suptitle('Rewards, parallel index: {0}, agent index: {1}'.format(
        parallel_ind, agent_ind)
        + '\n Factors: tar {0}, hea {1}'.format(tar_fac, hea_fac)
        + ', dis {0}, ris {1}, sof {2}'.format(dis_fac, ris_fac, sof_fac))
    save_plot(fig, 'rewards_B{0}A{1}T{2}H{3}D{4}R{5}S{6}.png'.format(
        parallel_ind, agent_ind, tar_fac, hea_fac, dis_fac, ris_fac, sof_fac),
        'plots')
