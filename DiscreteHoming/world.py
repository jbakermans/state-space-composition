#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:47:21 2020

@author: jacobb
"""

import json
import numpy as np
import copy
from scipy.sparse.csgraph import shortest_path


# Functions for generating data that TEM trains on: sequences of [state,observation,action] tuples

class World:
    def __init__(self, env=None, randomise_observations=False, randomise_policy=False, shiny=None):
        # If no environment provided: initialise completely empty environment
        if env is None:
            # Initialise empty environment
            self.init_empty()
        else:
            # If the environment is provided as a filename: load the corresponding file. If it's no filename, it's
            # assumed to be an environment dictionary
            if type(env) == str or type(env) == np.str_:
                # Filename provided, load graph from json file
                file = open(env, 'r')
                json_text = file.read()
                env = json.loads(json_text)
                file.close()
            else:
                env = copy.deepcopy(env)

            # Now env holds a dictionary that describes this world
            try:
                # Copy expected fiels to object attributes
                self.adjacency = env['adjacency']
                self.locations = env['locations']
                self.n_actions = env['n_actions']
                self.n_locations = env['n_locations']
                self.n_observations = env['n_observations']
            except (KeyError, TypeError) as e:
                # If any of the expected fields is missing: treat this as an invalid environment
                print('Invalid environment: bad dictionary\n', e)
                # Initialise empty environment
                self.init_emtpy()

        # If requested: shuffle observations from original assignments
        if randomise_observations:
            self.observations_randomise()

        # If requested: randomise policy by setting equal probability for each action
        if randomise_policy:
            self.policy_random()
            
        # Initialise shiny properties
        self.init_shiny(shiny)

    def init_emtpy(self):
        # Initialise all environment fields for an empty environment
        self.adjacency = []
        self.locations = []
        self.n_actions = 0
        self.n_locations = 0
        self.n_observations = 0
        
    def init_shiny(self, shiny=None, can_be_shiny=None):
        # Copy the shiny input
        self.shiny = copy.deepcopy(shiny)
        # If there's no shiny data provided: initialise this world as a non-shiny environement
        if self.shiny is None:
            # TEM needs to know that this is a non-shiny environment (e.g. for providing actions to generative model),
            # so set shiny to None for each location
            for location in self.locations:
                location['shiny'] = None
        # If shiny data is provided: initialise shiny properties
        else:
            # Specify default values for expected fields of shiny dictionary
            shiny_defaults = {'observations': None,
                     'gamma': 0.7,
                     'beta': 0.4,
                     'n': 1,
                     'returns_min': 0,
                     'returns_max': 0,
                     'can_be_shiny': None}
            # And copy over default values for non-existing fields
            for field in shiny_defaults:
                if field not in shiny:
                    self.shiny[field] = shiny_defaults[field]
            
            # Initially make all locations non-shiny
            for location in self.locations:
                location['shiny'] = False
                
            # Calculate all graph distances, since shiny objects aren't allowed to be too close together
            dist_matrix = shortest_path(csgraph=np.array(self.adjacency), directed=False)
            # Calculate number of actions available in each location, to determine if location is on the border
            n_actions_available = [sum([action['probability']>0 for action in location['actions']]) 
                                        for location in self.locations]
            # Only locations with the maximum actions available can be shiny (which excludes borders)
            can_be_shiny = self.shiny['can_be_shiny'] if can_be_shiny is None else can_be_shiny
            can_be_shiny = [loc_i for loc_i, n_a in enumerate(n_actions_available) if n_a == max(n_actions_available)] \
                if can_be_shiny is None else can_be_shiny
            # Initialise the list of shiny locations as empty
            self.shiny['locations'] = []
            # Then select shiny locations by adding them one-by-one, with the constraint that they can't be too
            # close to each other
            loc_dist_offset, loc_dist_iter, loc_dist_iter_max = np.max(dist_matrix) / self.shiny['n'], 0, 100
            while len(self.shiny['locations']) < self.shiny['n']:
                loc_dist_iter = loc_dist_iter + 1
                new = np.random.choice(can_be_shiny)
                too_close = [dist_matrix[new, existing] < (loc_dist_offset * max(0, 1 - loc_dist_iter/loc_dist_iter_max) 
                             + (np.max(dist_matrix)-4) / self.shiny['n']) for existing in self.shiny['locations']]
                if not any(too_close):
                    self.shiny['locations'].append(new)
                    
            # If shiny observations were not provided: set observation from shiny location's observation
            self.shiny['observations'] = [self.locations[shiny_location]['observation'] for shiny_location in
                                          self.shiny['locations']] \
                if self.shiny['observations'] is None else self.shiny['observations']                
            # Set shiny locations to be shiny, and apply the prescribed observation
            for shiny_observation, shiny_location in zip(self.shiny['observations'], self.shiny['locations']):
                self.locations[shiny_location]['shiny'] = True
                self.locations[shiny_location]['observation'] = shiny_observation
            # Make list of objects that are not shiny
            not_shiny = [observation for observation in range(self.n_observations) if
                         observation not in self.shiny['observations']]
            # Update observations so there is no non-shiny occurence of the shiny objects
            for location in self.locations:
                # Update a non-shiny location if it has a shiny object observation
                if location['id'] not in self.shiny['locations'] and location['observation'] in \
                        self.shiny['observations']:
                    # Pick new observation from non-shiny objects                    
                    location['observation'] = np.random.choice(not_shiny)
                    
            # Generate a policy towards each of the shiny objects
            self.shiny['policies'] = [self.policy_distance(shiny_location) for shiny_location in
                                      self.shiny['locations']]

    def init_start(self, start_options, goal, max_distance):
        # Calculate all graph distances, since objects should 
        dist_matrix = shortest_path(csgraph=np.array(self.adjacency), directed=False)
        # Select options that fit distance requirements from start options
        reach_options = [int(start) for start in start_options if 0 < dist_matrix[int(start), goal] <= max_distance]
        # In the (rare) case that there are no options within reach: use final start_options (walk start) instead
        reach_options = [int(start_options[-1])] if len(reach_options) == 0 else reach_options
        # And return random choice from available options
        return np.random.choice(reach_options)

    def has_self_actions(self):
        # Find out if there is an action for 'standing still' - these don't count towards direction dimensionality
        for location_i, location in enumerate(self.locations):
            for action in location['actions']:
                if action['transition'][location_i] == 1:
                    return True
        return False

    def observations_randomise(self):
        # Run through every abstract location
        for location in self.locations:
            # Pick random observation from any of the observations
            location['observation'] = np.random.randint(self.n_observations)
        return self

    def policy_random(self):
        # Run through every abstract location
        for location in self.locations:
            # Count the number of actions that can transition anywhere for this location
            count = sum([sum(action['transition']) > 0 for action in location['actions']])
            # Run through all actions at this location to update their probability
            for action in location['actions']:
                # If this action transitions anywhere: it is an avaiable action, so set its probability to 1/count
                action['probability'] = 1.0 / count if sum(action['transition']) > 0 else 0
        return self

    def policy_learned(self, reward_locations):
        # This generates a Q-learned policy towards reward locations.
        # Prepare new set of locations to hold policies towards reward locations
        new_locations, reward_locations = self.get_reward(reward_locations)
        # Initialise state-action values Q at 0
        for location in new_locations:
            for action in location['actions']:
                action['Q'] = 0
        # Do value iteration in order to find a policy toward a given location
        iters = 10 * self.n_locations
        # Run value iterations by looping through all actions iteratively
        for i in range(iters):
            # Deepcopy the current Q-values so they are the same for all updates (don't update values that you
            # later need)
            prev_locations = copy.deepcopy(new_locations)
            for location in new_locations:
                for action in location['actions']:
                    # Q-value update from value iteration of Bellman equation: Q(s,a) <- sum_across_s'(p(s,a,s')
                    # * (r(s') + gamma * max_across_a'(Q(s', a'))))
                    action['Q'] = sum([probability * ((new_location in reward_locations) + self.shiny['gamma'] * max(
                        [new_action['Q'] for new_action in prev_locations[new_location]['actions']])) for
                                       new_location, probability in enumerate(action['transition'])])
        # Calculate policy from softmax over Q-values for every state
        for location in new_locations:
            exp = np.exp(self.shiny['beta'] * np.array(
                [action['Q'] if action['probability'] > 0 else -np.inf for action in location['actions']]))
            for action, probability in zip(location['actions'], exp / sum(exp)):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action['probability'] = probability
        # Return new locations with updated policy for given reward locations
        return new_locations
    
    def policy_distance(self, reward_locations):
        # This generates a distance-based policy towards reward locations, which is much faster than Q-learning but
        # ignores policy and transition probabilities
        # Prepare new set of locations to hold policies towards reward locations
        new_locations, reward_locations = self.get_reward(reward_locations)
        # Create boolean vector of reward locations for matrix indexing
        is_reward_location = np.zeros(self.n_locations, dtype=bool)
        is_reward_location[reward_locations] = True
        # Calculate distances between all locations based on adjacency matrix - this doesn't take transition
        # probabilities into account!
        dist_matrix = shortest_path(csgraph=np.array(self.adjacency), directed=True)
        # Fill out minumum distance to any reward state for each action
        for location in new_locations:
            for action in location['actions']:
                action['d'] = np.min(dist_matrix[is_reward_location, np.array(action['transition']) > 0]) if any(
                    action['transition']) else np.inf
        # Calculate policy from softmax over negative distances for every action
        for location in new_locations:
            exp = np.exp(self.shiny['beta'] * np.array(
                [-action['d'] if action['probability'] > 0 else -np.inf for action in location['actions']]))
            for action, probability in zip(location['actions'], exp / sum(exp)):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action['probability'] = probability
        # Return new locations with updated policy for given reward locations
        return new_locations      

    def policy_distance_opposite(self, reward_locations):
        # This generates a distance-based policy towards reward locations, which is much faster than Q-learning but
        # ignores policy and transition probabilities
        # Prepare new set of locations to hold policies towards reward locations
        new_locations, reward_locations = self.get_reward(reward_locations)
        # Create boolean vector of reward locations for matrix indexing
        is_reward_location = np.zeros(self.n_locations, dtype=bool)
        is_reward_location[reward_locations] = True
        # Calculate distances between all locations based on adjacency matrix - this doesn't take transition
        # probabilities into account!
        dist_matrix = shortest_path(csgraph=np.array(self.adjacency), directed=True)
        # Fill out minumum distance to any reward state for each action
        for location in new_locations:
            for action in location['actions']:
                action['d'] = np.min(dist_matrix[is_reward_location, np.array(action['transition']) > 0]) if any(
                    action['transition']) else np.inf
        # Calculate policy from softmax over negative distances for every action
        for location in new_locations:
            exp = np.exp(self.shiny['beta'] * np.array(
                [action['d'] if action['probability'] > 0 and np.isfinite(action['d'])
                 else -np.inf for action in location['actions']]))
            for action, probability in zip(location['actions'], exp / sum(exp)):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action['probability'] = probability
        # Return new locations with updated policy for given reward locations
        return new_locations  
    
    def policy_avoid(self, avoid_locations, original_policy=None):
        # If policy isn't provided, use default policy
        original_policy = self.locations if original_policy is None else original_policy
        # Stick locations to be avoided in a list
        avoid_locations = [avoid_locations] if type(avoid_locations) is not list else avoid_locations        
        # Copy original policy to create optimal policy
        avoid_policy = copy.deepcopy(original_policy)
        # Now take the policy, and check for each action if it can lead to an avoided location
        for location in avoid_policy:
            for action in location['actions']:
                # Find if any transitions of this action lead to the avoided locations
                if any([trans > 0 and loc in avoid_locations for loc, trans in enumerate(action['transition'])]):
                    action['probability'] = 0
            # Then renormalise action probabilities
            location = self.normalise_policy(location)
        # Return the optimal policy, where the optimal action is always selected
        return avoid_policy    
    
    def policy_optimal(self, original_policy):
        # Copy original policy to create optimal policy
        optimal_policy = copy.deepcopy(original_policy)
        # Now take the policy, but only allow for the highest probability action
        for location in optimal_policy:
            # Collect probabilities for actions at current location
            probs = [action['probability'] for action in location['actions']]
            # Find which actions are the allowed actions for this location
            action_is_optimal = [action['probability'] == max(probs) for action in location['actions']]
            # Now only allow optimal actions
            for action, is_optimal in zip(location['actions'], action_is_optimal):
                action['probability'] = (1.0 / sum(action_is_optimal) if sum(action_is_optimal) > 0 else 0) \
                    if is_optimal else 0
        # Return the optimal policy, where the optimal action is always selected
        return optimal_policy
    
    def policy_opposite(self, original_policy):
        # Copy original policy to create opposite policy
        opposite_policy = copy.deepcopy(original_policy)
        # Now take the policy, but only allow for the highest probability action
        for location in opposite_policy:
            # Collect probabilities for actions at current location
            probs = np.array([action['probability'] for action in location['actions']])
            # And assign opposite probability - or 0 if action isn't available
            for action, prob in zip(location['actions'], probs):
                action['probability'] = (1 - prob) / sum(1 - probs[probs > 0]) if prob > 0 else 0
        # Return the optimal policy, where the optimal action is always selected
        return opposite_policy
    
    def normalise_policy(self, location):
        # Count total action probability to renormalise
        total_probability = sum([action['probability'] for action in location['actions']])
        # Renormalise action probabilities
        for action in location['actions']:
            action['probability'] = action['probability'] / total_probability if total_probability > 0 else action[
                'probability']
        # Return location with actions with normalied porbabilities
        return location    

    def generate_walk(self, walk_length=100, start=None, repeat_bias_factor=2):
        # If shiny hasn't been specified: there are no shiny objects, generate default policy
        if self.shiny is None:
            walk = self.walk_default([], walk_length, repeat_bias_factor=repeat_bias_factor)
        # If shiny was specified: use policy that uses shiny policy to approach shiny objects sequentially
        else:
            walk = self.walk_shiny([], walk_length, repeat_bias_factor=repeat_bias_factor)
        # Cast walk into input as expected by model
        locations, directions = self.get_model_input(walk)
        # And return list of locations and directions
        return locations, directions

    def generate_replay(self, start=None):
        # Pick current shiny object to approach
        shiny_current = np.random.randint(self.shiny['n'])
        # If no start position provided: find start position that is not the shiny position, or next to shiny
        if start is None:
            # Get distance matrix
            dist_matrix = shortest_path(csgraph=np.array(self.adjacency), directed=True)            
            # Get list of available start positions
            start_avail = [i for i in range(self.n_locations) 
                           if 2 < dist_matrix[i, self.shiny['locations'][shiny_current]] < 6]
            # Pick random location from list of available start positions
            start = start_avail[np.random.randint(len(start_avail))]
        # Replay consists of three parts: first a walk from start to shiny... 
        walk_from_start = self.walk_towards(self.shiny['locations'][shiny_current],
                                            policy=self.shiny['policies'][shiny_current], start=start)
        # ... Then a walk directly back to start
        walk_to_start = self.walk_towards(walk_from_start[0][0]['id'], 
                                          policy=self.policy_optimal(self.policy_distance(walk_from_start[0][0]['id'])), 
                                          start=walk_from_start[-1][0]['id'])
        # ... Then exploration from start, avoiding the shiny locations
        walk_explore = self.walk_default([], self.n_locations**2,
                                          policy=self.policy_avoid(self.shiny['locations'][shiny_current]), 
                                          start=walk_from_start[0][0]['id'])        
        # Cast walk into input as expected by model
        locations_from, directions_from = self.get_model_input(walk_from_start)        
        locations_to, directions_to = self.get_model_input(walk_to_start)     
        locations_explore, directions_explore = self.get_model_input(walk_explore)
        # And return model inputs
        return locations_from, directions_from, locations_to, directions_to, locations_explore, directions_explore
    
    def generate_replay_training(self, shiny, start, walk_lengths, replay_lengths, 
                                 walk_start=None, stay_at_shiny=False, rate_map=False):
        # There will be walks and replays, but currently this only supports 1 replay - easy to extend later
        # With just this single replay, walks and replays fit together as follows: 
        # 1) random walk that ends at shiny, 2) directly replay back to start, 3)-N) directed walk to shiny
        all_walks, all_replays = [], []    
        # Start by random diffusion, avoiding start, after which the shiny object is discovered for the first time        
        all_walks.append(self.walk_default([], walk_lengths[0], policy=self.policy_avoid(start), start=start))
        # Initialise shiny properties, making sure there is single shiny location: the last step of the first walk
        shiny['n'] = 1
        shiny['observations'] = None
        self.init_shiny(shiny, can_be_shiny=[all_walks[-1][-1][0]['id']])
        # If you want to stay at shiny after going there: only keep self action
        if stay_at_shiny:
            for action in self.shiny['policies'][0][self.shiny['locations'][0]]['actions']:
                action['probability'] = 1 if action['transition'][self.shiny['locations'][0]] == 1 else 0
        # Determine options for location where replay ends, and all next walks start
        walk_start = [start] if walk_start is None else walk_start
        # Select appropriate location (distance to shiny within reach of replay and other walks) from options
        # To avoid learning average direction: set max_distance to 2, was min(replay_lengths + walk_lengths) - 1
        walk_start = self.init_start(walk_start, self.shiny['locations'][0], min(replay_lengths + walk_lengths) - 1)
        # Second walk: replay directly from shiny to start location, then pad with standstill to fill partition
        all_replays.append(self.walk_pad_standstill(self.walk_towards(walk_start, 
                                          policy=self.policy_optimal(self.policy_distance(walk_start)), 
                                          start=self.shiny['locations'][0]), replay_lengths[0]))
        # All next walk: explore towards shiny from start location
        for walk_length in walk_lengths[1:]:
            # Or, if making a post-replay ratemap: random walk that avoids shiny
            all_walks.append(self.walk_default([], walk_length, 
                                               policy=self.locations if rate_map # self.policy_avoid(self.shiny['locations'][0])
                                               else self.shiny['policies'][0], start=walk_start))
        # Create model input from all walks
        walks, replays = {}, {}
        for in_dat, out_dat in zip([all_walks, all_replays], [walks, replays]):
            out_dat['position'], out_dat['direc'] = [np.concatenate(list(dat), axis=-1)
                                                     for dat in zip(*[self.get_model_input(walk) for walk in in_dat])]
        # And return model inputs
        return walks, replays    
            
    def walk_default(self, walk, walk_length, policy=None, start=None, repeat_bias_factor=2):
        # Finish the provided walk until it contains walk_length steps
        while len(walk) < walk_length:
            # Get new step of location, observation, action using specified policy
            new_location, new_observation, new_action = self.walk_step(walk, policy, start, 
                                                                       repeat_bias_factor=repeat_bias_factor)
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action]) 
        # Return the final walk
        return walk

    def walk_shiny(self, walk, walk_length, start=None, repeat_bias_factor=2):
        # Pick current shiny object to approach
        shiny_current = np.random.randint(self.shiny['n'])
        # Reset number of iterations to hang around an object once found
        shiny_returns = self.shiny['returns_min'] + np.random.randint(
            self.shiny['returns_max'] - self.shiny['returns_min'])
        # Finish the provided walk until it contains walk_length steps
        while len(walk) < walk_length:
            # Get new location based on previous action and location
            new_location = self.get_location(walk, start)
            # Check if the shiny object was found in this step
            if new_location['id'] == self.shiny['locations'][shiny_current]:
                # After shiny object is found, start counting down for hanging around
                shiny_returns -= 1
            # Check if it's time to select new object to approach
            if shiny_returns < 0:
                # If there is only one shiny object: walk randomly for some iterations before moving to shiny again
                if self.shiny['n'] == 1:
                    # Get number of steps for random walking: dependent on number of returns
                    random_walk_length = min(walk_length - len(walk) - 1, self.shiny['returns_min'] 
                                             + np.random.randint(self.shiny['returns_max'] - self.shiny['returns_min']))
                    # Create random walk from chosen location
                    random_walk = self.walk_default([], random_walk_length, start=new_location['id'])                    
                    # Extend current walk by freshly generated random walk
                    walk.extend(random_walk)
                    # Update new location from the generated random walk
                    new_location = self.get_location(walk)
                # Pick new current shiny object to approach
                shiny_current = np.random.randint(self.shiny['n'])
                # Reset number of iterations to hang around an object once found
                shiny_returns = self.shiny['returns_min'] + np.random.randint(
                    self.shiny['returns_max'] - self.shiny['returns_min'])
            # Get new observation at new location
            new_observation = self.get_observation(new_location)
            # Get new action based on policy of new location towards shiny object
            new_action = self.get_action(self.shiny['policies'][shiny_current][new_location['id']], walk)
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action])
        # Return the final walk
        return walk
    
    def walk_towards(self, goal, policy=None, start=None):
        # For unspecified policy: take the default
        policy = self.locations if policy is None else policy
        # Initialise empty walk and location 
        walk = []
        new_location = {'id': -1}
        # Walk until the goal location is reached
        while new_location['id'] != goal:
            # Get new step of location, observation, action using specified policy
            new_location, new_observation, new_action = self.walk_step(walk, policy, start)
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action])
        # Return the final walk
        return walk
    
    def walk_step(self, walk=None, policy=None, start=None, repeat_bias_factor=2):
        # For unspecified policy: take the default
        policy = self.locations if policy is None else policy
        # For unspecified walk: use empty walk
        walk = [] if walk is None else walk
        # Get new location based on previous action and location
        new_location = self.get_location(walk, start)
        # Get new observation at new location
        new_observation = self.get_observation(new_location)
        # Get new action based on policy of new location towards shiny object
        new_action = self.get_action(policy[new_location['id']], walk, repeat_bias_factor=repeat_bias_factor)    
        # Return step
        return new_location, new_observation, new_action
    
    def walk_pad_standstill(self, walk, walk_length):
        # Check if the last location from the current walk has a stand still action available
        stand_still_action = [action['id'] for action in walk[-1][0]['actions'] 
                              if action['transition'][walk[-1][0]['id']] > 0]
        if len(stand_still_action) < 1:
            print('Location ' + str(walk[-1][0]['id']) + ' has no stand-still action available for padding.')
            return walk
        else:
            stand_still_action = stand_still_action[0]
        # Change the last taken action to the standing still action
        walk[-1][2] = stand_still_action
        # Pad the provided walk with standstill actions until it contains walk_length steps
        while len(walk) < walk_length:
            # Copy previous location as new location
            new_location = walk[-1][0]
            # Get new observation at new location
            new_observation = self.get_observation(new_location)
            # New action is stand still action again
            new_action = stand_still_action
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action])
        # Return the final walk
        return walk
        
    
    def get_model_input(self, walk):
        # Find whether there is a 'stand-still' action, which doesn't count toward action dimensionality
        do_stand_still = self.has_self_actions()
        # Extract list of locations from walk
        locations = np.array([step[0]['id'] for step in walk], dtype=np.int16)
        # Extract list of actions from walk; convert to one-hot vectors, or vector of zeros for stand-still actions
        # if there are stand still actions
        actions = [np.zeros(self.n_actions - 1) if (step[2] == 0 and do_stand_still) else
                   np.eye(self.n_actions - do_stand_still)[step[2] - do_stand_still] for step in walk]
        # But James uses 'direction towards location', instead of 'action from location', so shift actions forward
        directions = np.transpose(np.array([np.zeros(self.n_actions - do_stand_still)] + actions[:-1]))
        # Return model input in expected format
        return locations, directions         

    def get_location(self, walk, start=None):
        # First step: start at random location, or at specified start location
        if len(walk) == 0:
            new_location = np.random.randint(self.n_locations) if start is None else start
        # Any other step: get new location from previous location and action
        else:
            new_location = int(
                np.flatnonzero(np.cumsum(walk[-1][0]['actions'][walk[-1][2]]['transition']) > np.random.rand())[0])
        # Return the location dictionary of the new location
        return self.locations[new_location]

    def get_observation(self, new_location):
        # Find sensory observation for new state, and store it as one-hot vector
        new_observation = np.eye(self.n_observations)[new_location['observation']]
        # Return the new observation
        return new_observation

    def get_action(self, new_location, walk, repeat_bias_factor=2):
        # Build policy from action probability of each action of provided location dictionary
        policy = np.array([action['probability'] for action in new_location['actions']])
        # Add a bias for repeating previous action to walk in straight lines, only if (this is not the first step) and
        # (the previous action was a move)
        policy[[] if len(walk) == 0 or new_location['id'] == walk[-1][0]['id'] else walk[-1][2]] *= repeat_bias_factor
        # And renormalise policy (note that for unavailable actions, the policy was 0 and remains 0, so in that case no
        # renormalisation needed)
        policy = policy / sum(policy) if sum(policy) > 0 else policy
        # Select action in new state
        new_action = int(np.flatnonzero(np.cumsum(policy) > np.random.rand())[0])
        # Return the new action
        return new_action

    def get_reward(self, reward_locations):
        # Stick reward location into a list if there is only one reward location. Use multiple reward locations
        # simultaneously for e.g. wall attraction
        reward_locations = [reward_locations] if type(reward_locations) is not list else reward_locations
        # Copy locations for updated policy towards goal
        new_locations = copy.deepcopy(self.locations)
        # Disable self-actions at reward locations because they will be very attractive
        for reward_location in reward_locations:
            # Check for each action if it's a self-action, and disable the action if it is
            for action in new_locations[reward_location]['actions']:
                if action['transition'][reward_location] == 1:
                    action['probability'] = 0
            # Renormalise action probabilities after potentially removing self-action
            new_locations[reward_location] = self.normalise_policy(new_locations[reward_location])
        return new_locations, reward_locations
    
    def get_dict(self):
        # Produce dictionary that contains all properties of this world object, and can be used to construct a new one
        env = {}
        # Copy properties to dictionary
        env['adjacency'] = self.adjacency
        env['locations'] = self.locations
        env['n_actions'] = self.n_actions
        env['n_locations'] = self.n_locations
        env['n_observations'] = self.n_observations
        # Return dictionary of world properties
        return env
        