import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.animation import FuncAnimation
from marlnav.utils import init_animation


class Animation:
    """Animation of the agents' movements in the environment."""

    def __init__(self, model, params):

        self.model = model
        agents_positions = model.agents_positions.cpu().numpy() # NOTE: FIX THIS! (right attribute & slicing)
        fig, agents_scatter = init_animation(params, agents_positions)
        fig.canvas.set_window_title('MARL-nav')
        self.fig = fig
        self.agents_scatter = agents_scatter # NOTE: Needsd another scatters/artists for the obstacles and the target

    def update(self, frame_number):

        self.model.update() # NOTE: FIX THIS! (right method & need actions from a model/sampler)
        updated_agents_pos = self.model.agents_positions
        self.agents_scatter.set_offsets(updated_agents_pos.cpu().numpy())

        return (self.agents_scatter,) # NOTE: Needsd another scatters/artists for the obstacles and the target

    def run(self):

        _ = FuncAnimation(self.fig, self.update, interval=0, blit=True)
        plt.show()
