import math
import numpy
import torch

from marlnav.utils import init_sampler, action_sampler, Observations


class Env(object):
    """Parallelized model for the agents' states and the environment."""

    def __init__(self, params):

        self.params = params
        self.device = params['device']
        self.num_parallel = params['num_parallel']
        self.num_agents = params['num_agents']
        self.num_obstacles = params['num_obstacles'] # NOTE: ADD THIS TO main.py (args & params)
        self.max_step = params['max_step']
        self.episode_len = params['episode_len']
        self._init_sampler = init_sampler(params['init'])
        self._sampler = action_sampler(params['sampler'])
        self._others_inds = torch.tensor(
            [[i for i in range(self.num_agents) if i != j]
              for j in range(self.num_agents)]).to(self.device) # NOTE: [[]] if self.num_agents == 1

        states, obstacles, target = self._init_sampler() # NOTE: REFACTOR THIS (and _reinit)!

        self.states = states
        self.obstacles = obstacles
        self.target = target

        self.min_speed = params['min_speed']
        self.max_speed = params['max_speed']
        self.min_accel = params['min_accel']
        self.max_accel = params['max_accel']

        # Counters for truncation and termination
        self._step_num = torch.zeros([self.num_parallel]).to(self.device)
        self._terminates = torch.tensor(self.num_parallel*[False]).to(self.device)
        self._reinit_mask = torch.zeros([self.num_parallel]).to(self.device)

        # Reward weight factors
        self._risk_factor = params['risk_factor']
        self._distance_factor = params['distance_factor']
        self._heading_factor = params['heading_factor']
        self._target_factor = params['target_factor']
        self._soft_factor = params['soft_factor']

        # Geometric attributes
        self._ob_risk_dist = 60.
        self._ag_risk_dist = 15.
        self._ob_coll_dist = 50.
        self._ag_coll_dist = 5.
        self._agents_min_d = 30.
        self._agents_max_d = 50.
        self._max_at_prop_d = 2 # NOTE: IS THIS NEEDED ?
        self._max_angle_diff = math.pi/8
        self._target_radius = 30.
        self._cap_distance = 0.1

    def reset(self):
        """Resets the agents' and env states, returns observations and info."""
        self._reinit_mask = torch.ones([self.num_parallel]).to(self.device)

        return self._obsevations(), self.params

    def _reinit(self):
        """Reinits the env's for terminated and truncated indeces."""
        states, obstacles, target = self._init_sampler()

        self.states = self._reinit_update(self.states, states)
        self.obstacles = self._reinit_update(self.obstacles, obstacles)
        self.target = self._reinit_update(self.target, target)
        self._step_num = self._reinit_update(
            self._step_num, torch.zeros([self.num_parallel]).to(self.device))

    def _reinit_update(self, old_vars, new_vars):
        """Changes new values based on reinit_mask rows (1 new, 0 old)."""

        return (torch.einsum('b,b...->b...', (1 - self._reinit_mask), old_vars)
            + torch.einsum('b,b...->b...', self._reinit_mask, new_vars))

    def step(self, actions):
        """Updates the states and returns observations, rewards, terminated,
        truncated and info tensors."""
        self._move_agents(actions)
        self._step_num += torch.ones([self.num_parallel]).to(self.device)
        truncated = (self._step_num > self.episode_len -1)
        observations = self._observations()
        rewards, terminated = self._rews_and_terms(observations)

        is_finished = torch.logical_or(truncated, terminated)
        self._reinit_mask = torch.where(is_finished, 1, 0)
        self._reinit() # reinit envs for terminated parallel indeces and use
        observations = self._observations() # observations from reinited states

        # return (torch.cat(observations, dim=2), rewards, terminated, truncated)
        return observations, rewards, terminated, truncated # NOTE: CAT OBSERVATIONS LATER & ADD INFO PARAMS

    def sample_actions(self):
        """Samples an action tensor."""
        return self._sampler()

    def _move_agents(self, actions):
        """Moves the agents' positions according to actions."""
        self._rotate_directions(actions[:,:,0])
        directions = self.states[:,:,2:4]
        accelerations = torch.clamp(
            actions[:,:,-1:], min=self.min_accel, max=self.max_accel)
        speeds = torch.clamp(self.states[:,:,4:5] + accelerations,
            min=self.min_speed, max=self.max_speed)
        self.states[:,:,4:5] = speeds
        self.states[:,:,:2] += directions * speeds

    def _rotate_directions(self, angles):
        """Rotates the directions of the whole states tensor."""
        directions = self.states[:,:,2:4]
        self.states[:,:,2:4] = torch.vmap(torch.vmap(
            self._rotate))(directions, angles)

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

        obstacle_risks = self._in_area_detect(
            observations.obstacles_distances, self._ob_risk_dist)
        agent_risks = self._in_area_detect(
            observations.others_distances, self._ag_risk_dist)
        obstacle_collisions = self._in_area_detect(
            observations.obstacles_distances, self._ob_coll_dist)
        agent_collisions = self._in_area_detect(
            observations.others_distances, self._ag_coll_dist)
        in_target_area = torch.where(
            observations.target_distance < self._target_radius, 1., 0.)
        distance_scores = self._distance_reward(
            observations.others_distances, self._agents_min_d,
            self._agents_max_d, self._max_at_prop_d) # NOTE: DOES THE METHOD REALLY NEED THE LAST PARAMETER ?
        heading_scores = self._heading_reward(
            observations.target_angle, self._max_angle_diff)
        soft_score = self._soft_reward(observations.target_distance)

        risks = torch.clamp(obstacle_risks + agent_risks, max=1)
        collisions = torch.clamp(obstacle_collisions + agent_collisions, max=1)
        atleast_1_coll, _ = torch.max(collisions, dim=1)
        all_in_target, _ = torch.min(in_target_area, dim=1)

        terminated = atleast_1_coll > 0
        terminated = torch.logical_or(terminated, self._terminates)

        # Set envs where agents have reached the target to terminate in the next
        # step (since cummulative reward will be zeroed at the terminal step)
        to_terminate = torch.squeeze(all_in_target) > 0
        self._terminates = torch.logical_and(~self._terminates, to_terminate)
        # Only previously False indeces are set to True based on 'to_terminate'
        # so that the reinit is done only ones after the target is reached.

        risk_loss = self._risk_factor * risks
        distance_rew = self._distance_factor * distance_scores
        heading_rew = self._heading_factor * heading_scores
        target_rew = self._target_factor * all_in_target.expand(
            size=(self.num_parallel, self.num_agents))
        soft_rew = self._soft_factor * soft_score
        reward = target_rew + heading_rew + distance_rew + soft_rew -risk_loss

        return torch.mean(reward, dim=1), terminated
        # return reward, terminated # NOTE: USE THIS FOR DEBUGGING/TESTING NEW REWARDS

    def _in_area_detect(self, distances, radius): # INPUT SHAPE: (num_parallel, num_agents, ...)
        """Returns a tensor of ones (in area) and zeros (outside)."""
        detections = torch.where(distances < radius, 1., 0.) # ... = num_objects or (num_agents-1)
        detections, _ = torch.max(detections, dim=2)

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

    def _heading_reward(self, heading_diffs, max_angle_diff): # INPUT SHAPE: (num_parallel, num_agents, 1)
        """Returns rewards for keeping the heading near target direction."""
        abs_diffs = torch.squeeze(torch.abs(heading_diffs), dim=2)

        return torch.where(abs_diffs < max_angle_diff, 1., 0.)

    def _soft_reward(self, distances_to_target):
        """Returns soft reward for closeness to the target area."""
        scaled_distances = distances_to_target/self._target_radius
        scaled_reward = torch.squeeze(2./(1. + scaled_distances**2), dim=2)

        return torch.clamp(scaled_reward, max=1.)

    def _get_distances(self, own_pos_array, others_pos_array): # NOTE: FOR SINGLE AGENT PARALLEL ARRAY
        """Returns tensor of distances between own and others positions."""

        return torch.cdist(torch.unsqueeze(own_pos_array, 1), others_pos_array)

    def _get_angles(self, own_pos_array, others_pos_array, direction_array): # NOTE: FOR SINGLE AGENT PARALLEL ARRAY
        """Returns a tensor of oriented angle differences from the direction."""
        difference_array = others_pos_array - torch.unsqueeze(own_pos_array, dim=1)
        normalized_array = torch.nn.functional.normalize(difference_array, dim=2)
        dot_array = torch.clamp(torch.einsum('bj,bij->bi', direction_array,
            normalized_array), -1 + 1e-8, 1 - 1e-8)
        dir_projections = torch.einsum('bi,bj->bij', dot_array, direction_array)
        orthogonal_comps = normalized_array - dir_projections
        signs = torch.where(orthogonal_comps[:,:,0] > 0, -1., 1.)

        return signs * torch.acos(dot_array)
