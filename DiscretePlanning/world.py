#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:26:43 2021

@author: jbakermans
"""

import json
import numpy as np
import copy
import random
from scipy.sparse.csgraph import shortest_path


# Functions for generating data that TEM trains on: sequences of [state,observation,action] tuples

class World:
    def __init__(self, env=None, randomise_observations=False, randomise_policy=False, env_type=None):
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
                self.init_empty()

        # If requested: shuffle observations from original assignments
        if randomise_observations:
            self.observations_randomise()

        # If requested: randomise policy by setting equal probability for each action
        if randomise_policy:
            self.policy_random(self.locations)
            
        # Initialise components, like shiny locations and borders
        self.init_components(env_type)

    def init_empty(self):
        # Initialise all environment fields for an empty environment
        self.adjacency = []
        self.locations = []
        self.components = {}
        self.n_actions = 0
        self.n_locations = 0
        self.n_observations = 0
        
    def init_components(self, env_type):
        # Initialise components with base component: standard empty environment
        self.init_base()
        # First copy env_type input so it can be re-used when env is copied
        self.env_type = env_type if env_type is None else dict(env_type)
        # Then initialise world as specified by env_type: dictionary that 
        # specifies world type, which determines which components it contains
        if self.env_type is not None:
            if env_type['name'] == 'reward':
                self.init_env_reward(env_type)
            elif env_type['name'] == 'walls':
                self.init_env_walls(env_type)
            elif env_type['name'] == 's':
                self.init_env_s(env_type)
            elif env_type['name'] == 'walls_n':
                self.init_env_walls_n(env_type)
            else:
                print('Component type ' + env_type['name'] + ' not recognised.')
                
    def init_env_reward(self, env_type):
        # Use shiny mechanism to create home in fixed position for easy comparison
        self.init_shiny('reward')

    def init_env_walls(self, env_type):
        # Use shiny mechanism to create home in fixed position for easy comparison
        self.init_shiny('reward')
        # Get all shiny components, so that walls can't go on top
        all_shiny = [key for key, value in self.components.items()
                     if value['type'] == 'shiny']
        # Now create wall, which can go anywhere except shouldn't overlap with reward
        self.init_wall(
            'obstacle1', {'length': 4, 'can_be_wall':
                         [i for i in range(self.n_locations)
                          if i not in flatten([self.components[comp]['locations']
                                           for comp in all_shiny])]})
        # And let's stick in a second wall that is one step smaller
        self.init_wall(
            'obstacle2', {'length': 3, 'can_be_wall':
                         [i for i in range(self.n_locations)
                          if i not in flatten([self.components[comp]['locations']
                                           for comp in all_shiny])]})

    def init_env_walls_n(self, env_type):
        # Use shiny mechanism to create home in fixed position for easy comparison
        self.init_shiny('reward')
        # Get all shiny components, so that walls can't go on top
        all_shiny = [key for key, value in self.components.items()
                     if value['type'] == 'shiny']
        # Create obstacles as determined by n_walls field in env_type
        for i in range(env_type['n_walls']):            
            # Now create wall, which can go anywhere except shouldn't overlap with reward
            self.init_wall_fast(
                'obstacle' + str(i), 
                {'length': env_type['wall_length'] if 'wall_length' in env_type.keys()
                 else np.random.choice([2,3,4]), 
                 'can_be_wall': [i for i in range(self.n_locations)
                                 if i not in flatten([self.components[comp]['locations']
                                                      for comp in all_shiny])],
                 'centre': env_type['centre'] if 'centre' in env_type.keys()
                 else False})
        # It's now possible for locations to be unable to reach reward,
        # because they are cut off by multiple walls. Remove all actions for those
        self.set_policy(self.policy_remove_blocked(
            self.get_policy(), self.components['reward']['locations'][0]))
            
    def init_env_s(self, env_type):
        # Use shiny mechanism to create home in fixed position for easy comparison
        self.init_shiny('reward', {'can_be_shiny': 
                                              [i for i in range(13)] + 
                                              [i+28 for i in range(13)]})
        # Get all shiny components, so that walls can't go on top
        all_shiny = [key for key, value in self.components.items()
                     if value['type'] == 'shiny']
        
        # Create two walls, same length, from opposite sides of the arena
        # Let's go fully hacky: this is only going to work for 7x7 square grid
        # And even hackier: just do horizontal for now
        no_go_rows = [int(s/7) for s in flatten(
            [self.components[comp]['locations'] for comp in all_shiny])]
        l = 5
        row1 = random.choice([i+1 for i in range(5) if (i + 1) not in no_go_rows])
        no_go_rows = no_go_rows + [row1, row1 - 1, row1 + 1]
        row2 = random.choice([i+1 for i in range(5) if (i + 1) not in no_go_rows])
        loc1 = [row1*7 + i for i in range(l)]
        loc2 = [row2*7 + i + (7-l) for i in range(l)]      
        dir1 = 1
        dir2 = 1
        # Now create wall at specified locations
        self.init_wall(
            'obstacle1', {'length': l, 'locations': loc1, 'orientation': dir1})
        self.init_wall(
            'obstacle2', {'length': l, 'locations': loc2, 'orientation': dir2})
    
    def init_component(self, name, vals=None, defaults=None):
        # Copy over default values for non-existing fields
        self.components[name] = self.init_pars(vals, defaults)
        
        # Initialise location and action level dictionaries
        for location in self.locations:
            location['components'][name] = {}
            for action in location['actions']:
                action['components'][name] = {}
        # Return root level component dictionary
        return self.components[name]
    
    def init_base(self, name=None):
        # By default, base environment name is 'base'
        name = 'base' if name is None else name
        # Base component: copy original environment, as a base to add components to
        self.components = {name:{'type': 'base'}}
        # Initialise location and action level as empty component lists
        for location in self.locations:
            location['components'] = {name: {'in_comp': True}}
            for action in location['actions']:
                action['components'] = {name: {}}
        # Now set base policy to original environment policy
        # This is the only place where I can't copy policy in-place:
        # I want the base policy to be just the plain environment without objects
        # The root policy will reflect objects (e.g. walls)
        self.set_policy(self.get_policy(in_place=False), name='base')
                                
    def init_shiny(self, name, shiny_in=None):
        # Defaults for root level component dictionary
        defaults = {'type': 'shiny',
                    'locations': None,
                    'gamma': 0.7,
                    'beta': 0.4,
                    'n': 1,
                    'can_be_shiny': None}
     
        # Initialise component with default parameters, and get root dictionary
        shiny = self.init_component(name, shiny_in, defaults)
        
        # Initially make all locations non-shiny
        for location in self.locations:
            location['components'][name]['in_comp'] = False
            
        # If locations are not provided: find suitable locations for shiny objects
        if shiny['locations'] is None:
            # Find 'normal' locations that are not on border or wall
            shiny['can_be_shiny'] = self.get_internal_locations() \
                if shiny['can_be_shiny'] is None else shiny['can_be_shiny']
            # Initialise the list of shiny locations as empty
            shiny['locations'] = []
            # Calculate all graph distances, since shiny objects aren't allowed to be too close together
            dist_matrix = self.get_distance(self.adjacency)
            # Then select shiny locations by adding them one-by-one, with the constraint that they can't be too
            # close to each other
            loc_dist_offset, loc_dist_iter, loc_dist_iter_max = np.max(dist_matrix) / shiny['n'], 0, 100
            while len(shiny['locations']) < shiny['n']:
                loc_dist_iter = loc_dist_iter + 1
                new = np.random.choice(shiny['can_be_shiny'])
                too_close = [dist_matrix[new, existing] 
                             < (loc_dist_offset * max(0, 1 - loc_dist_iter/loc_dist_iter_max) 
                             + (np.max(dist_matrix)-4) / shiny['n']) for existing in shiny['locations']]
                if not any(too_close):
                    shiny['locations'].append(new)
                
        # Set shiny locations to be shiny
        for shiny_location in shiny['locations']:
            self.locations[shiny_location]['components'][name]['in_comp'] = True

        # The shiny policy is a copy of the root policy
        self.set_policy(self.get_policy(),name=name)
        # # Generate a policy towards shiny: copy base policy, then update from distances
        # self.set_policy(self.policy_distance(self.get_policy(name='base', in_place=False), 
        #                                      shiny['locations']), 
        #                 name)
        
    def init_wall(self, name, wall_in=None):
        # Defaults for root level component dictionary
        defaults = {'type': 'wall',
                    'locations': None,
                    'gamma': 0.7,
                    'beta': 0.4,
                    'length': 3,
                    'orientation': None,
                    'can_be_wall': None}
        
        # Initialise component with default parameters, and get root dictionary
        wall = self.init_component(name, wall_in, defaults)    
        
        # Initially make all locations non-wall
        for location in self.locations:
            location['components'][name]['in_comp'] = False
            
        # If locations are not provided: find suitable locations for wall objects
        if wall['locations'] is None:
            # Find 'normal' locations that are not on border or wall
            wall['can_be_wall'] = self.get_internal_locations(wall['can_be_wall'])
            # Choose an orientation if not provided. Orientation is action direction
            wall['orientation'] = np.random.randint(self.has_self_actions(), self.n_actions) \
                if wall['orientation'] is None else wall['orientation']
            # Seed wall location, then grow along action direction
            wall['locations'] = [np.random.choice(wall['can_be_wall'])]
            # Add wall locations until it has desired length
            while len(wall['locations']) < wall['length']:
                # Find transitioned location for current end location
                new = np.argmax(self.locations[wall['locations'][-1]] \
                                ['actions'][wall['orientation']]['transition'])
                # If no transition in action direction, or new location not allowed: restart
                if new == 0 or new not in wall['can_be_wall']:
                    wall['locations'] = [np.random.choice(wall['can_be_wall'])]
                else:
                    wall['locations'].append(new)                            
                    
        # Set wall locations to be part of wall
        for wall_location in wall['locations']:
            self.locations[wall_location]['components'][name]['in_comp'] = True
                        
        # Generate a policy towards shiny: copy base policy, then update from distances
        self.set_policy(self.policy_distance(self.get_policy(name='base', in_place=False), 
                                             wall['locations'], opposite=True), 
                        name)        

        # Now there is one thing specific to walls: they are not accessible,
        # so they change the real environment.
        wall_pol = self.policy_avoid(self.get_policy(), wall['locations'])
        for wall_location in wall['locations']:
            for action in wall_pol[wall_location]['actions']:
                action['probability'] = 1 \
                    if np.argmax(action['transition']) == wall_location \
                        else 0
        self.set_policy(wall_pol)
        
    def init_wall_fast(self, name, wall_in=None):
        # Quick-and-dirty wall generation: less general, but much quicker.
        # Assumes square environment on square grid.
        # Defaults for root level component dictionary
        defaults = {'type': 'wall',
                    'locations': None,
                    'gamma': 0.7,
                    'beta': 0.4,
                    'length': 3,
                    'orientation': None,
                    'can_be_wall': None}
        
        # Initialise component with default parameters, and get root dictionary
        wall = self.init_component(name, wall_in, defaults)    
        
        # Initially make all locations non-wall
        for location in self.locations:
            location['components'][name]['in_comp'] = False
            
        # Quick function to sample wall on square env of square grid    
        def sample_wall_locs(wall, side):
            # Choose orientation (vert = 0 or hor = 1)
            wall['orientation'] = np.round(np.random.rand())
            # Choose starting loc in orthogonal direction
            start_loc_orth = np.random.randint(2*wall['centre'], 
                                               side - 2*wall['centre'])
            # Choose starting loc in parallel direction
            start_loc_par = np.random.randint(1*wall['centre'],
                                              side - wall['length'] - wall['centre'])
            # Set wall locations
            if wall['orientation'] == 0:
                # Vertical wall
                return [start_loc_par * side + start_loc_orth
                        + i * side for i in range(wall['length'])]
            else:
                # Horizontal wal
                return [start_loc_orth * side + start_loc_par
                        + i for i in range(wall['length'])]

        # If locations are not provided: find suitable locations for wall objects
        if wall['locations'] is None:
            # Get side length from number of locations (assumes square graph!)
            side = int(np.sqrt(self.n_locations))
            # Sample wall
            wall['locations'] = sample_wall_locs(wall, side)
            # Repeat until all locations valid
            while (any([l not in wall['can_be_wall'] for l in wall['locations']])):
                # Sample wall again
                wall['locations'] = sample_wall_locs(wall, side)               
                                
        # Copy the root policy for this component
        self.set_policy(self.get_policy(), name)                    
                
        # Set wall locations to be part of wall
        for wall_location in wall['locations']:
            self.locations[wall_location]['components'][name]['in_comp'] = True

        # Now there is one thing specific to walls: they are not accessible,
        # so they change the real environment.
        wall_pol = self.policy_avoid(self.get_policy(), wall['locations'])
        for wall_location in wall['locations']:
            for action in wall_pol[wall_location]['actions']:
                action['probability'] = 1 \
                    if np.argmax(action['transition']) == wall_location \
                        else 0
        self.set_policy(wall_pol)        
    
    def init_pars(self, pars_in=None, defaults=None):
        # If no input parameters provided: set input to empty
        pars_in = {} if pars_in is None else pars_in
        # If no defaults provided: start with empty dictionary
        defaults = {} if defaults is None else defaults
        # Then copy over all values provided, overwriting defaults if existing
        for category in pars_in.keys():
            # If this dict value is a dict itself: copy all it contains
            # (but keep defaults for values it doesn't contain!)
            if isinstance(pars_in[category], dict):
                if category not in defaults:
                    defaults[category] = {}
                for key, val in pars_in[category].items():
                    defaults[category][key] = val
            # If this dict entry is just a value: copy that value
            else:
                defaults[category] = pars_in[category]
        # Return dictionary that combines defaults and provided parameters
        return defaults

    def init_start(self, start_options, goal, max_distance):
        # Calculate all graph distances, since objects should 
        dist_matrix = self.get_distance(self.adjacency)
        # Select options that fit distance requirements from start options
        reach_options = [int(start) for start in start_options if 0 < dist_matrix[int(start), goal] <= max_distance]
        # In the (rare) case that there are no options within reach: use final start_options (walk start) instead
        reach_options = [int(start_options[-1])] if len(reach_options) == 0 else reach_options
        # And return random choice from available options
        return np.random.choice(reach_options)

    def set_locs_on_grid(self):
        # Change the x, y coordinates of locations so they end up on a square grid
        xs = [l['x'] for l in self.locations]
        ys = [l['y'] for l in self.locations]
        # Find rows and columns: number of distinct x and y values
        rows = len(set(ys))
        cols = len(set(xs))        
        # Find max and min for x and y
        xlim = [min(xs), max(xs)]
        ylim = [min(ys), max(ys)]
        # Get steps in both directions
        dx = ((xlim[1] - xlim[0]) / (cols-1)) if cols > 1 else 0
        dy = ((ylim[1] - ylim[0]) / (rows-1)) if rows > 1 else 0
        # Then scale (subtract mean, scale, add mean) the largest direction
        if dy > dx and dx > 0:
            ys = [(y - 0.5 * (ylim[0] + ylim[1])) * dx / dy + 0.5 * (ylim[0] + ylim[1]) for y in ys]
        if dx > dy and dy > 0:
            xs = [(x - 0.5 * (xlim[0] + xlim[1])) * dy / dx + 0.5 * (xlim[0] + xlim[1]) for x in xs]            
        # Finally update all location positions
        for location, x, y in zip(self.locations, xs, ys):
            location['x'] = x
            location['y'] = y

    def has_self_actions(self):
        # Find out if there is an action for 'standing still'
        action_is_standstill = [True for _ in range(self.n_actions)]
        # Find if each action always transitions to self if available
        for location_i, location in enumerate(self.locations):
            for action_i, action in enumerate(location['actions']):
                if action['probability'] > 0 and action['transition'][location_i] == 0:
                    action_is_standstill[action_i] = False
        return any(action_is_standstill)

    def observations_randomise(self):
        # Run through every abstract location
        for location in self.locations:
            # Pick random observation from any of the observations
            location['observation'] = np.random.randint(self.n_observations)

    def policy_random(self, original_policy):
        # Set uniform policy in-place
        for location in original_policy:
            # Count the number of actions that can have > 0 probability
            count = sum([action['probability'] > 0 for action in location['actions']])
            # Run through all actions at this location to update their probability
            for action in location['actions']:
                # If this action transitions anywhere: it is an avaiable action, so set its probability to 1/count
                action['probability'] = (1.0 / count if action['probability'] > 0 else 0) \
                    if count > 0 else 0
        # Return udpated policy in case it's needed for further processing
        return original_policy

    def policy_learned(self, original_policy, reward_locations, beta=0.5, gamma=0.75):
        # This generates a Q-learned policy towards reward locations in-place
        # Make sure reward locations are in a list
        reward_locations = [reward_locations] if type(reward_locations) is not list else reward_locations
        # Initialise state-action values Q at 0
        for location in original_policy:
            for action in location['actions']:
                action['Q'] = 0
        # Do value iteration in order to find a policy toward a given location
        iters = 10 * self.n_locations
        # Run value iterations by looping through all actions iteratively
        for i in range(iters):
            # Deepcopy the current Q-values so they are the same for all updates (don't update values that you
            # later need)
            prev_locations = copy.deepcopy(original_policy)
            for location in original_policy:
                for action in location['actions']:
                    # Q-value update from value iteration of Bellman equation: Q(s,a) <- sum_across_s'(p(s,a,s')
                    # * (r(s') + gamma * max_across_a'(Q(s', a'))))
                    action['Q'] = sum([probability * ((new_location in reward_locations) + gamma * max(
                        [new_action['Q'] for new_action in prev_locations[new_location]['actions']])) for
                                       new_location, probability in enumerate(action['transition'])])
        # Calculate policy from softmax over Q-values for every state
        original_policy = self.policy_softmax(original_policy, beta)
        # Return udpated policy in case it's needed for further processing
        return original_policy
                
    
    def policy_distance(self, original_policy, reward_locations, beta=0.5, optimal=False,
                        opposite=False, adjacency=None, disable_if_worse=False):
        # This generates a distance-based policy towards reward locations, which is much faster than Q-learning but
        # ignores policy and transition probabilities. It updates policy in place
        # If adjacency is not provided, use own adjacency matrix
        adjacency = self.adjacency if adjacency is None else adjacency
        # Make sure reward locations are in a list
        reward_locations = [reward_locations] if type(reward_locations) is not list else reward_locations
        # Create boolean vector of reward locations for matrix indexing
        is_reward_location = np.zeros(self.n_locations, dtype=bool)
        is_reward_location[reward_locations] = True
        # Calculate distances between all locations based on adjacency matrix - this doesn't take transition
        # probabilities into account!
        dist_matrix = self.get_distance(adjacency)
        # Fill out minumum distance to any reward state for each action
        for location in original_policy:
            for action in location['actions']:
                action['d'] = np.min(dist_matrix[np.array(action['transition']) > 0, is_reward_location]) if any(
                    action['transition']) else np.inf
        # Calculate policy from softmax over negative distances for every action
        for location in original_policy:
            exp = np.exp(beta * np.array(
                [(1 if opposite else -1) * action['d'] 
                 if action['probability'] > 0 and np.isfinite(action['d'])
                 else -np.inf for action in location['actions']]))
            # If all actions make things worse (i.e. they all increase distance,
            # or decrease distance for opposite direction): disable them
            if disable_if_worse and (
                    all([(1 if opposite else -1) * action['d'] <= 
                         (1 if opposite else -1) * np.min(dist_matrix[is_reward_location, location['id']])
                         for action in location['actions'] if np.isfinite(action['d'])])):
                exp[:] = 0
            # For disconnected nodes (e.g. in wall) all negative distances can be 0
            for action, a_exp in zip(location['actions'], exp):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action['probability'] = ((a_exp == max(exp))/sum(exp == max(exp)) if optimal 
                                         else a_exp / sum(exp)) if sum(exp) > 0 else 0  
        # Return udpated policy in case it's needed for further processing
        return original_policy    
                    
    def policy_avoid(self, original_policy, avoid_locations):
        # Stick locations to be avoided in a list
        avoid_locations = [avoid_locations] if type(avoid_locations) is not list else avoid_locations        
        # Now take the policy, and remove policy to avoided locations in-place
        for location in original_policy:
            for action in location['actions']:
                # Find if any transitions of this action lead to the avoided locations
                if any([trans > 0 and loc in avoid_locations for loc, trans in enumerate(action['transition'])]):
                    action['probability'] = 0
            # Then renormalise action probabilities
            location = self.normalise_policy(location)
        # Return udpated policy in case it's needed for further processing
        return original_policy
                
    def policy_optimal(self, original_policy):
        # Take original policy, but only allow for the highest probability action in-place
        for location in original_policy:
            # Collect probabilities for actions at current location
            probs = [action['probability'] for action in location['actions']]
            # Find which actions are the allowed actions for this location
            action_is_optimal = [action['probability'] == max(probs) for action in location['actions']]
            # Now only allow optimal actions
            for action, is_optimal in zip(location['actions'], action_is_optimal):
                action['probability'] = (1.0 / sum(action_is_optimal) if sum(action_is_optimal) > 0 else 0) \
                    if is_optimal else 0
        # Return udpated policy in case it's needed for further processing
        return original_policy
        
    def policy_opposite(self, original_policy):
        # Take original policy, but change to 1-orginal in-place
        for location in original_policy:
            # Collect probabilities for actions at current location
            probs = np.array([action['probability'] for action in location['actions']])
            # And assign opposite probability - or 0 if action isn't available
            for action, prob in zip(location['actions'], probs):
                action['probability'] = (1 - prob) / sum(1 - probs[probs > 0]) if prob > 0 else 0
        # Return udpated policy in case it's needed for further processing
        return original_policy
    
    def policy_zero_to_self_transition(self, original_policy, zero_policy=False, change_policy=False):
        # Make sure all unavailable actions, indicated by all-zero transitions,
        # now get a stand-still transition
        for l_i, location in enumerate(original_policy):
            for action in location['actions']:
                if np.sum(action['transition']) == 0:
                    action['transition'] = np.eye(len(original_policy))[l_i]
            # Optionally: also set transitions for zero-probability policy actions to self
            if zero_policy:
                for action in location['actions']:
                    if action['probability'] == 0:
                        action['transition'] = np.eye(len(original_policy))[l_i]                
            # Optionally: when all actions are disallowed in policy, set them uniform
            if change_policy and sum([a['probability'] for a in location['actions']]) == 0:
                for action in location['actions']:
                    action['probability'] = 1/len(location['actions'])
        # Return udpated policy in case it's needed for further processing
        return original_policy
    
    def policy_remove_blocked(self, original_policy, goal):
        # Set actions from locations that can't reach reward all to 0, like walls
        # Make sure reward locations are in a list
        # Calculate distances between all locations based on adjacency matrix - this doesn't take transition
        # probabilities into account!
        goal_dist = self.get_distance(self.get_adjacency(original_policy))[:, goal]
        # Calculate policy from softmax over negative distances for every action
        for location, dist in zip(original_policy, goal_dist):
            if not np.isfinite(dist):
                for action in location['actions']:
                    action['probability'] = 0
        # Return udpated policy in case it's needed for further processing
        return original_policy    
    
    def policy_absorbing_reward(self, original_policy, reward_locations):
        # Make reward locations absorbing: all actions transition to self
        for reward_loc in reward_locations:
            location = original_policy[reward_loc]
            for action in location['actions']:
                action['transition'] = np.eye(len(original_policy))[reward_loc]
        # Return udpated policy in case it's needed for further processing
        return original_policy    

    def policy_softmax(self, original_policy, beta):
        # Calculate policy from softmax over Q-values for every state. Assume Q-value as key in actions
        for location in original_policy:
            exp = np.exp(beta * np.array(
                [action['Q'] if action['probability'] > 0 else -np.inf for action in location['actions']]))
            for action, probability in zip(location['actions'], exp / sum(exp)):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action['probability'] = probability
        # Return updated policy for further processing (but softmax is applied in-place)
        return original_policy

                    
    def normalise_policy(self, location):
        # Count total action probability to renormalise
        total_probability = sum([action['probability'] for action in location['actions']])
        # Renormalise action probabilities in-place
        for action in location['actions']:
            action['probability'] = action['probability'] / total_probability if total_probability > 0 \
                else action['probability']
        # Return udpated location in case it's needed for further processing
        return location
                                    
    def set_policy(self, new_policy, name=None):
        # Copy input policy actions to the requested component dict in actions
        for l_i, new_location in enumerate(new_policy):
            for a_i, new_action in enumerate(new_location['actions']):
                # Select action dictionary to update
                old_dict = self.locations[l_i]['actions'][a_i] if name is None \
                    else self.locations[l_i]['actions'][a_i]['components'][name]
                # Run through keys in new dictionary and update corresponding old keys
                for new_key, new_val in new_action.items():
                    old_dict[new_key] = new_val

    def get_policy(self, name=None, in_place=True):
        # Get policy (=list of locations) but with actions for requested component only
        # If name is not provided: return a 'clean' copy of the root policy, without components
        policy = []
        for location in self.locations:
            # Select all location info except actions
            curr_location = select_dict_entries(location, ['action'], include=False)
            # Then add actions: all for component, or cleaned (without components) for environment
            curr_location['actions'] = [select_dict_entries(action, ['components'], include=False)
                                        if name is None else action['components'][name]
                                        for action in location['actions']]
            # Append sub-selected location to policy
            policy.append(curr_location)
        # If you want to make changes to copy of policy instead of in-place: deepcopy
        policy = policy if in_place else copy.deepcopy(policy)
        # Return policy with selected actions
        return policy

    def get_internal_locations(self, include=None):
        # Find 'interal' locations: locations with all actions available in base environment
        include = [i for i in range(self.n_locations)] if include is None else include
        # Calculate number of actions available in each location, to determine if location is on the border
        n_actions_available = [sum([action['components']['base']['probability']>0 
                                    for action in location['actions']]) 
                                    for location in self.locations]
        # Only locations with the maximum actions available can be shiny (which excludes borders)
        return [loc_i for loc_i, n_a in enumerate(n_actions_available) 
                if n_a == max(n_actions_available) and loc_i in include]        

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
    
    def get_dict(self):
        # Produce dictionary that contains all properties of this world object, and can be used to construct a new one
        env = {}
        # Copy properties to dictionary
        env['adjacency'] = self.adjacency
        env['locations'] = self.locations
        env['components'] = self.components
        env['n_actions'] = self.n_actions
        env['n_locations'] = self.n_locations
        env['n_observations'] = self.n_observations        
        # Return dictionary of world properties
        return env
    
    def get_base_dict(self):
        # Produce dictionary that contains all properties of this world object, and can be used to construct a new one
        env = {}
        # Copy properties to dictionary
        env['adjacency'] = self.adjacency
        env['locations'] = self.get_policy(name='base')
        env['components'] = {'base': self.components['base']}
        env['n_actions'] = self.n_actions
        env['n_locations'] = self.n_locations
        env['n_observations'] = self.n_observations        
        # Return dictionary of world properties
        return env    
    
    def get_base_copy(self):
        # Create new environment with the same base as current
        new_env = World(self.get_dict())
        # Get base policy for new env, and set as base and root policy
        new_env.set_policy(self.get_policy(name='base', in_place=False))
        new_env.set_policy(self.get_policy(name='base', in_place=False), 'base')        
        # Return new environment
        return new_env

    def get_full_copy(self):
        # Create new environment with the same base as current
        new_env = World(self.get_base_dict(), env_type=self.env_type)
        # Then initialise components of base copy with own init_env parameters
        #new_env.init_components(self.env_type)
        # Return new environment
        return new_env
    
    def get_adjacency(self, policy):
        # Get adjacency matrix for given policy: 1 if there is any action that 
        # with probability > 0 takes you from location i (row) to location j (col)
        return [[1*any([action['probability'] > 0 and action['transition'][l_to] > 0 
                        for action in loc_from['actions']]) 
                 for l_to in range(len(policy))] for loc_from in policy]

    def get_transition_matrix(self, policy):
        # Get transition matrix for given policy: T[s_from, s_to] 
        # = sum_a p(a | s_from) * p(s_to | s_from, a)
        return [[sum([action['probability'] * action['transition'][l_to]
                        for action in loc_from['actions']]) 
                 for l_to in range(len(policy))] for loc_from in policy]
    
    def get_successor_representation(self, policy, gamma=0.75):
        # Get successor representation for given policy: M[s_from, s_to] 
        # = (I - gamma * T)^-1, with transition probability T[s_from, s_to] 
        return np.linalg.inv(np.eye(len(policy)) - gamma * np.array(
            self.get_transition_matrix(policy)))
    
    def get_stationary_distribution(self, policy):
        # Get stationary distribution: first left-hand eigenvector of transition
        # probability T[s_from, s_to] (transpose to get lh instead of rh ev)
        return np.linalg.eig(np.array(self.get_transition_matrix(policy)).transpose())[1][:,0]
    
    def get_distance(self, adjacency, directed=True):
        # Use shortest path function to get distance matrix.
        # Adjacency is list of lists
        return shortest_path(csgraph=np.array(adjacency), directed=directed)
        

def select_dict_entries(dict_in, entries, include=True):
    # Subselect dict entries, keeping entries if include=True or removing them if include=False
    return {key: val for key, val in dict_in.items()
            if (include and key in entries or not include and key not in entries)}

def flatten(t):
    return [item for sublist in t for item in sublist]
            