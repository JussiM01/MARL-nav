import math
import numpy
import torch

from marlnav.utils import random_states, action_sampler, Observations


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

        # Counters for truncation and termination
        self._step_num = torch.ones([self.batch_size]).to(self.device)
        self._steps_left = (self.max_step -1)*torch.ones([self.batch_size]).to(
            self.device)
        self._terminated = torch.zeros([self.batch_size]).to(self.device)

        # Reward weight factors
        self._collision_factor = params['collision_factor']
        self._distance_factor = params['distance_factor']
        self._heading_factor = params['heading_factor']
        self._target_factor = params['target_factor']

        # Geometric attributes
        self._ob_coll_dist = 10.
        self._ag_coll_dist = 5.
        self._agents_min_d = 10.
        self._agents_max_d = 25.
        self._max_at_prop_d = 2 # NOTE: IS THIS NEEDED ?
        self._max_angle_diff = math.pi/8
        self._target_radius = 25.
        self._cap_distance = 0.1


    def reset(self): # NOTE: REFACTOR THIS!
        """Resets the agents' and env states, returns observations and info."""
        states, obstacles, target = random_states(self.init)

        self.states = self._terminated_update(self.states, states)
        self.obstacles = self._terminated_update(self.obstacles, obstacles)
        self.target = self._terminated_update(self.target, target)
        self._step_num = self._terminated_update(
            self._step_num, torch.ones([self.batch_size]).to(self.device))

        return self._obsevations(), self.params


    def _terminated_update(self, old_vars, new_vars):

        return (1 - self._terminated) * old_vars + self._terminated * new_vars


    def step(self, actions):
        """Updates the states and returns observations, rewards, terminated,
        truncated and info tensors."""
        self._move_agents(actions)
        self._step_num += torch.ones([self.batch_size]).to(self.device)
        truncated = (self._step_num < self.max_step)
        observations = self._observations()
        rewards, terminated = self._rews_and_terms(observations) # NOTE: DO RESET AFTER THIS?

        # return (torch.cat(observations, dim=2), rewards, terminated, truncated,
        #         self.params)
        return observations, rewards, terminated, truncated, self.params # NOTE: CAT OBSERVATIONS LATER & ADD INFO PARAMS

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
            self.target, self.states[:,i,2:4])
            for i in range(self.num_agents)], dim=1)

        target_distance = torch.cat([self._get_distances(self.states[:,i,:2],
            self.target) for i in range(self.num_agents)], dim=1)

        obstacles_angles = torch.cat([
            torch.stack([self._get_angles(self.states[:,i,:2],
            self.obstacles[:,j:j+1,:], self.states[:,i,2:4])
            for i in range(self.num_agents)], dim=1)
            for j in range(self.num_obstacles)], dim=2)

        obstacles_distances = torch.cat([
            torch.cat([self._get_distances(self.states[:,i,:2],
            self.obstacles[:,j:j+1,:]) for i in range(self.num_agents)], dim=1)
            for j in range(self.num_obstacles)], dim=2)

        others_angles = torch.stack([self._get_angles(self.states[:,i,:2],
            torch.index_select(self.states, 1, self._others_inds[i])[:,:,:2],
            self.states[:,i,2:4]) for i in range(self.num_agents)], dim=1)

        others_distances = torch.cat([self._get_distances(self.states[:,i,:2],
            torch.index_select(self.states, 1, self._others_inds[i])[:,:,:2])
            for i in range(self.num_agents)], dim=1)

        # others_directions = ... (OR SHOULD THIS BE DIRECTION DIFFERENCES?
        #                          AND BELOW VELOCITY DIFFERENCES? (if the speeds are made dynamic))

        # others_speeds = ... # NOTE: ADD THESE LATER, ONLY IF THEY ARE MADE DYNAMIC

        target_angle = torch.where(
            target_distance < self._cap_distance, 0., target_angle)
        obstacles_angles = torch.where(
            obstacles_distances < self._cap_distance, 0., obstacles_angles)
        others_angles = torch.where(
            others_distances < self._cap_distance, 0., others_angles)

        return Observations(target_angle, target_distance, obstacles_angles,
            obstacles_distances, others_angles, others_distances)

            # NOTE LATER ADD others_directions TO STACKING LIST ABOVE (and perhaps speeds too)

    def _rews_and_terms(self, observations): # NOTE: REFACTOR TO USE HELPER METHODS

        obstacle_collisions = self._collision_loss(
            observations.obstacles_distances, self._ob_coll_dist)
        agent_collisions = self._collision_loss(
            observations.others_distances, self._ag_coll_dist)
        in_target_area = torch.where(
            observations.target_distance < self._target_radius, 1., 0.)
        distance_scores = self._distance_reward(
            observations.others_distances, self._agents_min_d,
            self._agents_max_d, self._max_at_prop_d) # NOTE: DOES THE METHOD REALLY NEED THE LAST PARAMETER ?
        heading_scores = self._heading_reward(
            observations.target_angle, self._max_angle_diff)

        collisions = torch.clamp(obstacle_collisions + agent_collisions, max=1)
        all_in_target, _ = torch.min(in_target_area, dim=1)
        self._steps_left -= torch.squeeze(
            torch.where(all_in_target > 0, 1, 0), dim=1)
        terminated = (self._steps_left == 0)
        self._terminated = torch.where(terminated, 1, 0)

        coll_loss = self._collision_factor * collisions
        distance_rew = self._distance_factor * distance_scores
        heading_rew = self._heading_factor * heading_scores
        target_rew = self._target_factor * all_in_target.expand(
            size=(self.batch_size, self.num_agents))
        reward = target_rew + heading_rew + distance_rew -coll_loss

        return reward, terminated

    def _collision_loss(self, distances, collision_dist): # INPUT SHAPE: (batch_size, num_agents, ...)
        """Returns a tensor of ones (collisions) and zeros (no collisions)."""
        collisions = torch.where(distances < collision_dist, 1., 0.) # ... = num_objects or (num_agents-1)
        detections, _ = torch.max(collisions, dim=2)

        return detections

    def _distance_reward(self, distances, min_dist, max_dist, max_value):
        """Returns normalized rewards for staying within a proper distance."""
        above_min = torch.where(min_dist < distances, 1., 0.)
        below_max = torch.where(distances < max_dist, 1., 0.)
        detections = above_min * below_max
        capped_sums = torch.clamp(torch.sum(detections, dim=2), max=max_value) # MAX DETECTIONS TO CARE ABOUT
                                                                                # MAYBE NO NEED?
        return torch.div(capped_sums, max_value) # SCALED BY THE MAX VALUE, THIS IS NEEDED (might dominate other
                                                                            # rewards for large number of agents)

    def _heading_reward(self, heading_diffs, max_angle_diff): # INPUT SHAPE: (batch_size, num_agents, 1)
        """Returns rewards for keeping the heading near target direction."""
        abs_diffs = torch.squeeze(torch.abs(heading_diffs), dim=2)

        return torch.where(abs_diffs < max_angle_diff, 1., 0.)

    def _get_distances(self, own_pos_batch, others_pos_batch): # NOTE: FOR SINGLE AGENT BATCH
        """Returns batch of distances between own and others positions."""

        return torch.cdist(torch.unsqueeze(own_pos_batch, 1), others_pos_batch)

    def _get_angles(self, own_pos_batch, others_pos_batch, direction_batch): # NOTE: FOR SINGLE AGENT BATCH
        """Returns a batch of oriented angle differences from the direction."""
        difference_batch = others_pos_batch - torch.unsqueeze(own_pos_batch, dim=1)
        normalized_batch = torch.nn.functional.normalize(difference_batch, dim=2)
        dot_batch = torch.clamp(torch.einsum('bj,bij->bi', direction_batch,
            normalized_batch), -1 + 1e-8, 1 - 1e-8)
        dir_projections = torch.einsum('bi,bj->bij', dot_batch, direction_batch)
        orthogonal_comps = normalized_batch - dir_projections
        signs = torch.where(orthogonal_comps[:,:,0] > 0, -1., 1.)

        return signs * torch.acos(dot_batch)
