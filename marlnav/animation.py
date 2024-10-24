import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.animation import FuncAnimation
from marlnav.utils import init_animation


class Animation:
    """Animation of the agents' movements in the environment."""

    def __init__(self, model, params):

        self.model = model
        agents_pos = model.agents_positions.cpu().numpy() # NOTE: FIX THIS! (right attribute & slicing)
        obs_pos = model.obstacles_positions.cpu().numpy() # NOTE: FIX THIS! (right attribute & slicing)
        target_pos = model.target_position.cpu().numpy() # NOTE: FIX THIS! (right attribute & slicing)
        fig, agents_scatter, obs_scatter, target_scatter = init_animation(
            params, agents_pos, obs_pos, target_pos)
        fig.canvas.set_window_title('MARL-nav')
        self.fig = fig
        self.agents_scatter = agents_scatter # NOTE: Make sure that these are drawn in the right order!
        self.obs_scatter = obs_scatter       # agents should be drawn on top of obstacles and target if
        self.target_scatter = target_scatter # they ever come across/colide with each other.

    def update(self, frame_number):

        self.model.update() # NOTE: FIX THIS! (right method & need actions from a model/sampler)
        updated_agents_pos = self.model.agents_positions # NOTE: FIX THIS! (right attribute & slicing)
        self.agents_scatter.set_offsets(updated_agents_pos.cpu().numpy())

        return (self.agents_scatter,)

    def run(self):

        _ = FuncAnimation(self.fig, self.update, interval=0, blit=True)
        plt.show()
