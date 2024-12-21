import math
import numpy
import torch

from marlnav.utils import random_states, action_sampler


class DynamicsModel(object):
    """Batched model for the agents' states and the environment."""

    def __init__(self, params):

        self.params = params
        self.device = params['device']
        self.batch_size = params['batch_size']
        self.num_agents = params['num_agents']
        self.max_step = params['max_step']
        self.init = params['init']
        self._sampler = action_sampler(params['sampler'])

        states, obstacles, target = random_states(self.init) # NOTE: REFACTOR THIS (and reset)!

        self.states = states
        self.obstacles = obstacles
        self.target = target
        self.step_num = 1


    def reset(self): # NOTE: REFACTOR THIS!
        """Resets the agents' and env states, returns observations and info."""
        states, obstacles, target = random_states(self.init)

        self.states = states
        self.obstacles = obstacles
        self.target = target
        self.step_num = 1

        return self._obsevations(), self.params

    def step(self, actions):
        """Updates the states and returns observations, rewards, terminated,
        truncated and info tensors."""
        self._move_agents(actions)
        self.step_num += 1
        truncated = torch.tensor(
            (self.step_num < self.max_step)).repeat(self.batch_size)
        observations = self._obsevations()
        rewards = self._rewards()
        terminated = self._terminated()

        return observations, rewards, terminated, truncated, self.params

    def sample_actions(self):
        """Samples an action batch."""
        return self._sampler()

    def _move_agents(self, actions):
        """Moves the agents' positions according to actions."""
        self._rotate_directions(actions)
        directions = self.states[:,:,2:4]
        velocities = self.states[:,:,4:5]
        self.states[:,:,:2] += directions * velocities

    def _rotate_directions(self, actions):
        """Rotates the directions of the whole states batch."""
        directions = self.states[:,:,2:4]
        self.states[:,:,2:4] = torch.vmap(torch.vmap(
            self._rotate))(directions, actions)

    def _rotate(self, direction_vector, angle):
        """Rotates the agent's direction by the given angle."""
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)])]).to(self.device)

        return torch.matmul(rotation_matrix, direction_vector)

    def _observations(self):
        """Calculates and returns the observations tensor."""
        return torch.vmap(torch.vmap(
            self._single_obs))(self.states, self.obstacles, self.target)

    def _single_obs(self, state, obstacles, target):
        """Calculates and returns single agent's observation tensor."""

        raise NotImplementedError

    def _rewards(self):
        """Calculates and returns the rewards tensor."""
        return torch.vmap(torch.vmap(
            self._single_rew))(self.states, self.obstacles, self.target)

    def _single_rew(self, state, obstacles, target):
        """Calculates and returns single agent's reward tensor."""

        raise NotImplementedError

    def _terminated(self):
        """Calculates and returns the terminated tensor."""
        return torch.vmap(
            self._single_env_term)(self.states, self.obstacles, self.target)

    def _single_env_term(self, state, obstacles, target):
        """Calculates and returns single env's terminated tensor."""

        raise NotImplementedError
