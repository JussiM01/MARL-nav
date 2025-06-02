import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.animation import FuncAnimation
from marlnav.utils import init_animation


class Animation:
    """Animation of the agents' movements in the environment."""

    def __init__(self, env, params):

        self.env = env
        self.parallel_index = params['parallel_index']
        agents_pos = env.states[self.parallel_index,:,:2].cpu().numpy()
        obs_pos = env.obstacles[self.parallel_index,:,:].cpu().numpy()
        target_pos = env.target[self.parallel_index,:,:].cpu().numpy()
        fig, agents_scatter, obs_scatter, target_scatter = init_animation(
            params, agents_pos, obs_pos, target_pos)
        fig.canvas.manager.set_window_title('MARL-nav')
        self.fig = fig
        self.agents_scatter = agents_scatter # NOTE: Make sure that these are drawn in the right order!
        self.obs_scatter = obs_scatter       # agents should be drawn on top of obstacles and target if
        self.target_scatter = target_scatter # they ever come across/colide with each other.
        self.sampling_style = params['sampling_style']
        self.max_step = params['max_step']
        self.interval = params['interval']

    def update(self, frame_number):
        """Updates the agents' new positions to the `agents_scatter` object."""
        if self.sampling_style == 'policy':
            raise NotImplementedError
        elif self.sampling_style == 'sampler':
            actions = self.env.sample_actions()

        # self.env._move_agents(actions)
        obs, rew, _, _, _ = self.env.step(actions)
        # print('STEP_NUM: ', self.env._step_num[self.parallel_index].item())
        # print('OBSTACLES DISTANCES: ', obs.obstacles_distances[self.parallel_index,:,:])
        # print('OTHERS DISTANCES: ', obs.others_distances[self.parallel_index,:,:])
        # print('TARGET DISTANCE: ', obs.target_distance[self.parallel_index,:,:])
        # print('TARGET ANGLE: ', obs.target_angle[self.parallel_index,:,:])
        # print('REWARDS: ', rew[self.parallel_index])
        # # print('REWARDS: ', rew[parallel_ind,:]) # NOTE: USE THIS FOR DEBUGGING/TESTING NEW REWARDS
        # print('\n')
        updated_agents_pos = self.env.states[self.parallel_index,:,:2]
        self.agents_scatter.set_offsets(updated_agents_pos.cpu().numpy())

        updated_obs_pos = self.env.obstacles[self.parallel_index,:,:2]
        self.obs_scatter.set_offsets(updated_obs_pos.cpu().numpy())

        return (self.agents_scatter, self.obs_scatter)

    def run(self):
        """Runs the animation."""
        _ = FuncAnimation(self.fig, self.update, frames=self.max_step,
            repeat=False, interval=self.interval, blit=True)
        plt.show()
