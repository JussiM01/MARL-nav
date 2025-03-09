import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.animation import FuncAnimation
from marlnav.utils import init_animation


class Animation:
    """Animation of the agents' movements in the environment."""

    def __init__(self, model, params):

        self.model = model
        self.batch_index = params['batch_index']
        agents_pos = model.states[self.batch_index,:,:2].cpu().numpy()
        obs_pos = model.obstacles[self.batch_index,:,:].cpu().numpy()
        target_pos = model.target[self.batch_index,:,:].cpu().numpy()
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
            actions = self.model.sample_actions()

        # self.model._move_agents(actions)
        obs, rew, _, _, _ = self.model.step(actions)
        # print('STEP_NUM: ', self.model._step_num[self.batch_index].item())
        # print('OBSTACLES DISTANCES: ', obs.obstacles_distances[self.batch_index,:,:])
        # print('OTHERS DISTANCES: ', obs.others_distances[self.batch_index,:,:])
        # print('TARGET DISTANCE: ', obs.target_distance[self.batch_index,:,:])
        # print('TARGET ANGLE: ', obs.target_angle[self.batch_index,:,:])
        # print('REWARDS: ', rew[self.batch_index,:])
        # print('\n')
        updated_agents_pos = self.model.states[self.batch_index,:,:2]
        self.agents_scatter.set_offsets(updated_agents_pos.cpu().numpy())

        return (self.agents_scatter,)

    def run(self):
        """Runs the animation."""
        _ = FuncAnimation(self.fig, self.update, frames=self.max_step,
            repeat=False, interval=self.interval, blit=True)
        plt.show()
