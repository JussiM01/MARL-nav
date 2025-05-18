import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
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
        angles0 = params['angles'][0]
        angles1 = params['angles'][1]

        (angle00, angle01, angle02) = params['angles'][0]
        (angle10, angle11, angle12,) = params['angles'][1]
        device = params['device']

        self._angle00 = ([-math.pi/6, 0.] if i == 0 else angle00
            for i in range(params['max_step']))
        self._angle02 = ([math.pi/6, 0.] if i == 0 else angle02
            for i in range(params['max_step']))
        angle1_half = [
            [0.5*angle10[0], 0.],[0.5*angle11[0], 0.],[0.5*angle12[0], 0.]]
        self._angles1 = (
            angle1_half if i == 0. else [angle10, angle11, angle12]
            for i in range(params['max_step']))
        self.angle_batch = (torch.tensor([[next(self._angle00), angle01,
            next(self._angle02)], next(self._angles1)]).to(device)
            for i in range(params['max_step']))

    def __call__(self):
        return next(self.angle_batch)

# class MockSampler(object):
#     """Mock sampler for testing the dynamics model and its visualization."""
#
#     def __init__(self, params):
#         (angle00, angle01, angle02) = params['angles'][0]
#         (angle10, angle11, angle12) = params['angles'][1]
#         device = params['device']
#
#         self._angle01 = (-0.25*math.pi if i == 0
#             else 0.5*math.pi*(-1)**((i//50)%2+1) if (i%50 == 0) else 0.
#             for i in range(params['max_step']))
#         self._angle02 = (-0.25*math.pi if i == 0
#             else 0.5*math.pi*(-1)**((i//25)%2+1) if (i%25 == 0) else 0.
#             for i in range(params['max_step']))
#         self._angles1 = ([0.5*angle10, 0.5*angle11, 0.5*angle12] if i == 0
#             else [angle10, angle11, angle12]
#             for i in range(params['max_step']))
#         self.angle_batch = (torch.tensor([[0., next(self._angle01),
#             next(self._angle02)], next(self._angles1)]).to(device)
#             for i in range(params['max_step']))
#
#     def __call__(self):
#         return next(self.angle_batch)


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


def plot_states_and_rews(env, num_steps, batch_ind, agent_ind):
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
        obs, rew, _, _, _ = env.step(actions)
        # print('OBSTACLES DISTANCES: ', obs.obstacles_distances[batch_ind,:,:])
        # print('OTHERS DISTANCES: ', obs.others_distances[batch_ind,:,:])
        # print('TARGET DISTANCE: ', obs.target_distance[batch_ind,:,:])
        # print('TARGET ANGLE: ', obs.target_angle[batch_ind,:,:])
        # print('REWARDS: ', rew[batch_ind])
        # # print('REWARDS: ', rew[batch_ind,:]) # NOTE: USE THIS FOR DEBUGGING/TESTING NEW REWARDS
        # print('\n')
        target_angles += [obs.target_angle[batch_ind, agent_ind,0].item()]
        target_distances += [obs.target_distance[batch_ind, agent_ind,0].item()]
        all_obs_angels += [obs.obstacles_angles[batch_ind, agent_ind,0].item()]
        all_obs_distances += [obs.obstacles_distances[batch_ind, agent_ind,0].item()]
        angles_to_first += [obs.others_angles[batch_ind, agent_ind, 0].item()]
        distances_to_first += [obs.others_distances[batch_ind, agent_ind, 0].item()]
        angles_to_second += [obs.others_angles[batch_ind, agent_ind, 1].item()]
        distances_to_second += [obs.others_distances[batch_ind, agent_ind, 1].item()]
        rewards += [rew[batch_ind].item()]
        # rewards += [rew[batch_ind, agent_ind].item()] # NOTE: USE THIS FOR DEBUGGING/TESTING NEW REWARDS

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

    fig.suptitle('States, batch index: {0}, agent index: {1}'.format(
        batch_ind, agent_ind))
    save_plot(fig, 'states_batch_{0}_agent_{1}.png'.format(
        batch_ind, agent_ind), 'plots')

    tar_fac = env._target_factor
    hea_fac = env._heading_factor
    dis_fac = env._distance_factor
    col_fac = env._collision_factor
    sof_fac = env._soft_factor

    fig, ax = plt.subplots(1, 1)
    ax.set(xlabel='step number', ylabel='value')
    ax.plot(rewards)
    fig.suptitle('Rewards, batch index: {0}, agent index: {1}'.format(
        batch_ind, agent_ind)
        + '\n Factors: tar {0}, hea {1}'.format(tar_fac, hea_fac)
        + ', dis {0}, col {1}, sof {2}'.format(dis_fac, col_fac, sof_fac))
    save_plot(fig, 'rewards_B{0}A{1}T{2}H{3}D{4}C{5}S{6}.png'.format(
        batch_ind, agent_ind, tar_fac, hea_fac, dis_fac, col_fac, sof_fac),
        'plots')
