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
        self.num_obstacles = params['num_obstacles'] # NOTE: ADD THIS TO main.py (args & params)
        self.max_step = params['max_step']
        self.init = params['init']
        self._sampler = action_sampler(params['sampler'])
        self._others_inds = torch.tensor(
            [[i for i in range(self.num_agents) if i != j]
              for j in range(self.num_agents)]).to(self.device) # NOTE: [[]] if self.num_agents == 1

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
        target_angle = torch.stack([self._get_angles(self.states[:,i,:2],
            self.target, self.states[:,i,2:4])  # NOTE: CHECK THIS
            for i in range(self.num_agents)], dim=1)

        target_distance = torch.stack([self._get_angles(self.states[:,i,:2],
            self.target) for i in range(self.num_agents)], dim=1)  # NOTE: CHECK THIS

        obstacles_angles = torch.stack([
            torch.stack([self._get_angles(self.states[:,i,:2],
            self.obstacles[:,j,:], self.states[:,i,2:4])  # NOTE: CHECK THIS
            for i in range(self.num_agents)], dim=1)
            for j in range(self.num_obstacles)], dim=2)  # NOTE: CHECK THIS

        obstacles_distances = torch.stack([
            torch.stack([self._get_angles(self.states[:,i,:2],
            self.obstacles[:,j,:]) for i in range(self.num_agents)], dim=1)  # NOTE: CHECK THIS
            for j in range(self.num_obstacles)], dim=2) # NOTE: CHECK THIS

        others_angles = torch.stack([self._get_angles(self.states[:,i,:2],
            torch.index_select(self.states, 1, self._others_inds[i])[:,:,:2], # NOTE: CHECK THIS
            self.states[:,i,2:4]) for i in range(self.num_agents)], dim=1) # NOTE: CHECK THIS!

        others_distances = torch.stack([self._get_distances(self.states[:,i,:2],
            torch.index_select(self.states, 1, self._others_inds[i])[:,i,:2]) # NOTE: CHECK THIS
            for i in range(self.num_agents)], dim=1) # NOTE: CHECK THIS!

        # others_directions = ...

        # others_speeds = ... # NOTE: ADD THESE LATER, ONLY IF THEY ARE MADE DYNAMIC

        return torch.cat([target_angle, target_distance, obstacles_angles,
            obstacles_distances, others_angles, others_distances], dim=2)  # NOTE: CHECK THIS

            # NOTE LATER ADD others_directions TO STACKING LIST ABOVE (and perhaps speeds too)

    def _get_distances(self, own_pos_batch, others_pos_batch): # NOTE: FOR SINGLE AGENT BATCH
        """Returns batch of distances between own and others positions."""

        return torch.cdist(torch.unsqueeze(own_pos_batch, 1), others_pos_batch)

    def _get_angles(own_pos_batch, others_pos_batch, direction_batch): # NOTE: FOR SINGLE AGENT BATCH
        """Returns a batch of oriented angle differences from the direction."""
        difference_batch = others_pos_batch - torch.unsqueeze(own_pos_batch, dim=1)
        normalized_batch = torch.nn.functional.normalize(difference_batch, dim=2)
        dot_batch = torch.einsum('bj,bij->bi', direction_batch, normalized_batch)
        dir_projections = torch.einsum('bi,bj->bij', dot_batch, direction_batch)
        orthogonal_comps = normalized_batch - dir_projections
        signs = torch.where(orthogonal_comps[:,:,0] > 0, -1., 1.)

        return signs * torch.acos(dot_batch)

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
