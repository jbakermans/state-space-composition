#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:32:26 2021

@author: jbakermans
"""

import json
import numpy as np
import copy
import random
from scipy.sparse.csgraph import shortest_path

class World:
    def __init__(self, environments, env_specs=None):
        # Copy environment input so we can re-initialise when needed
        self.input = {'environments': environments, 'env_specs': env_specs}
        # Environmments input is a list of environments, from high level to low level
        self.env = self.init_env(environments, env_specs)

    def init_env(self, environments, env_specs=None):
        # Create environment dictionary
        env = {}
        # Load root level (highest in hierarchy) environment
        curr_env = self.load_env(random.choice(environments[0]))
        # Set all data entries for this environment
        env['adjacency'] = curr_env['adjacency']
        env['locations'] = curr_env['locations']
        env['n_actions'] = curr_env['n_actions']
        env['n_locations'] = curr_env['n_locations']
        # Set level in hierarchy fro this environment: 0 is root (highest level)
        env['level'] = 0
        # Then add components (which includes adding component policies)
        env = self.init_components(env, env_specs)
        # Then move on to next depths. Could make this proper recursive but it's a pain
        for l_i, location in enumerate(env['locations']):
            # Create a lower-level env for each high-level location
            curr_env = self.load_env(environments[1][l_i % len(environments[1])])
            # Create new env dict
            new_env = {}
            # Set all data entries for this environment
            new_env['adjacency'] = curr_env['adjacency']
            new_env['locations'] = curr_env['locations']
            new_env['n_actions'] = curr_env['n_actions']
            new_env['n_locations'] = curr_env['n_locations']
            new_env['level'] = 1          
            # Then add components (which includes adding component policies)
            new_env = self.init_components(new_env, env_specs, higher=location)
            # Finally, add new env to location
            env['locations'][l_i]['env'] = new_env 
        # And return full environment
        return env

    def load_env(self, env):
        # If the environment is provided as a filename: load the corresponding file. 
        # If it's not a filename, it's assumed to be an environment dictionary
        if type(env) == str or type(env) == np.str_:
            # Filename provided, load graph from json file
            file = open(env, 'r')
            json_text = file.read()
            env = json.loads(json_text)
            file.close()
        else:
            env = copy.deepcopy(env)
        # Return env dictionary
        return env
        
    def init_components(self, env, env_specs, higher=None):
        # Set environment specification defaults
        spec_defaults = {'blockade': True,
                         'dummy': False}
        # And combine default with input specifications
        env_specs = self.init_pars(env_specs, spec_defaults)        
        # Initialise components with base component: standard empty environment
        env = self.init_base(env)
        # Initialise all other components: reward & doors
        if higher is None:
            # Create reward
            env = self.init_shiny(env, 'reward')
            # And create blockade if requested, and whether is just a dummy
            if env_specs['blockade']:
                env = self.init_block(env, 'blockade', {'dummy': env_specs['dummy']})
        else:
            # If there is a higher level: component for each available high-level action
            env = self.init_shinies(env, higher)
            # Also lowest level gets grid locations (for prettier plotting)
            self.set_locs_on_grid(env['locations'], req_rows=5, req_cols=5)
        # Finally, return env dictionary
        return env
    
    def init_base(self, env):
        # Base component: copy original environment, as a base to add components to
        env['components'] = {'base': {'type': 'base'}}
        # Initialise location and action level as empty component lists
        for location in env['locations']:
            location['components'] = {'base': {'in_comp': True}}
            for action in location['actions']:
                action['components'] = {'base': {}}
        # Now set base policy to original environment policy
        # Set transition for unavailable actions to self-transition
        self.set_policy(self.policy_zero_to_self_transition(self.get_policy(
            env=env, in_place=False)), env=env, name='base')
        # And get adjacency for that original policy
        env['adjacency'] = self.get_adjacency(self.get_policy(env=env, name='base'))
        # And return env equipped with base component and policy
        return env
    
    def init_shinies(self, env, higher):
        # If the current location at the higher level is reward: create reward
        if higher['components']['reward']['in_comp']:
            env = self.init_shiny(env, 'reward')
            is_shiny = env['components']['reward']['locations']
        else:
            is_shiny = []
        # Initalise locations that can be shiny: any that aren't shiny already
        can_be_shiny = [i for i in range(env['n_locations']) if i not in is_shiny]
        # Then add shiny for each higher level action
        # The NAME of the component corresponds to the high-level action
        # The ID of the component corresponds to the agent's representation
        ids = [a['id'] for a in higher['actions']]; random.shuffle(ids)
        # These are linked in memory
        for action, comp_id in zip(higher['actions'], ids):
            # Create door object
            env = self.init_shiny(env, 'door' + str(action['id']),
                                  {'can_be_shiny': can_be_shiny, 
                                   'id': comp_id})
            # Update locations available for shiny by removing current object
            can_be_shiny = [i for i in can_be_shiny if i not in
                            env['components']['door' + str(action['id'])]['locations']]
        # Return updated environment
        return env    
                                
    def init_shiny(self, env, name, shiny_in=None):
        # Defaults for root level component dictionary
        defaults = {'type': 'shiny',
                    'locations': None,
                    'n': 1,
                    'can_be_shiny': None,
                    'id': None}
     
        # Initialise component with default parameters, and get root dictionary
        env = self.init_component(env, name, shiny_in, defaults)
        # Get current shiny dictionary
        shiny = env['components'][name]
        
        # Initially make all locations non-shiny
        for location in env['locations']:
            location['components'][name]['in_comp'] = False
            
        # If locations are not provided: find suitable locations for shiny objects
        if shiny['locations'] is None:
            # Initialise which locations can be shiny, if not provided
            can_be_shiny = [i for i in range(env['n_locations'])] \
                if shiny['can_be_shiny'] is None else shiny['can_be_shiny']
            # Only include internal locations, so corners are not possible
            can_be_shiny = self.get_internal_locations(env, can_be_shiny)
            # Initialise the list of shiny locations as empty
            shiny['locations'] = []
            # Add shiny locations until 
            for _ in range(shiny['n']):
                # Sample shiny location and add to locations
                shiny['locations'].append(np.random.choice(can_be_shiny))
                # Update available locations for shiny
                can_be_shiny = [i for i in can_be_shiny if i not in shiny['locations']]
                
        # Set shiny locations to be shiny
        for shiny_location in shiny['locations']:
            env['locations'][shiny_location]['components'][name]['in_comp'] = True

        # Generate a policy towards shiny: copy base policy, then update from distances
        self.set_policy(self.policy_distance(self.policy_zero_to_self_transition(
            self.get_policy(
                env=env, name='base', in_place=False)), 
            shiny['locations'], env=env), 
            env=env, name=name)
        
        # Return updated environment
        return env
    
    def init_block(self, env, name, block_in=None):
        # Defaults for root level component dictionary
        defaults = {'type': 'block',
                    'dummy': False,
                    'locations': None,
                    'orientation': None,
                    'can_be_block': None}
        
        # Initialise component with default parameters, and get root dictionary
        env = self.init_component(env, name, block_in, defaults)    
        # Get current shiny dictionary
        block = env['components'][name]        
        
        # Initially make all locations non-block
        for location in env['locations']:
            location['components'][name]['in_comp'] = False
        
        # Build blockade component
        if block['dummy']:
            # This is a dummy blockade: is has blockade representations, but no locations
            block['locations'] = [None, None]
        else:
            # If locations are not provided: find suitable locations for block objects
            if block['locations'] is None:
                # Find 'normal' locations that are not on border or block
                block['can_be_block'] = self.get_internal_locations(
                    env, block['can_be_block'])
                # Choose an orientation if not provided. Orientation is action direction
                block['orientation'] = np.random.randint(env['n_actions']) \
                    if block['orientation'] is None else block['orientation']
                # Set seed block location
                block['locations'] = [np.random.choice(block['can_be_block'])]
                # Then set leaf block location: one step along orientation
                block['locations'].append(np.argmax(
                    env['locations'][block['locations'][-1]] \
                        ['actions'][block['orientation']]['transition']))
                        
            # Set block locations to be part of block
            for block_location in block['locations']:
                env['locations'][block_location]['components'][name]['in_comp'] = True
                            
            # Generate a policy away from either block side
            self.set_policy(self.policy_distance(self.policy_zero_to_self_transition(
                self.get_policy(
                    env=env, name='base', in_place=False)), 
                block['locations'], env=env, opposite=True), 
                env=env, name=name)
            
            # Now there is one thing specific to blocks: they cut all transitions between
            block_pol = self.get_policy(env=env)
            for b1, b2 in [block['locations'], list(reversed(block['locations']))]:
                # Cut off any actions from b1 to b2
                for action in block_pol[b1]['actions']:
                    if action['transition'][b2] > 0:
                        action['probability'] = 0
                # Then renormalise
                block_pol[b1] = self.normalise_policy(block_pol[b1])
            # And update main policy
            self.set_policy(block_pol, env=env)
            
        # Return updated environment
        return env
            
    
    def init_component(self, env, name, vals=None, defaults=None):
        # Copy over default values for non-existing fields
        env['components'][name] = self.init_pars(vals, defaults)
        
        # Initialise location and action level dictionaries
        for location in env['locations']:
            location['components'][name] = {}
            for action in location['actions']:
                action['components'][name] = {}
        # Return root level component dictionary
        return env        
    
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
        for location in original_policy:
            exp = np.exp(beta * np.array(
                [action['Q'] if action['probability'] > 0 else -np.inf for action in location['actions']]))
            for action, probability in zip(location['actions'], exp / sum(exp)):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action['probability'] = probability
        # Return udpated policy in case it's needed for further processing
        return original_policy
                
    
    def policy_distance(self, original_policy, reward_locations, 
                        beta=0.5, opposite=False, env=None, adjacency=None):
        # This generates a distance-based policy towards reward locations, which is much faster than Q-learning but
        # ignores policy and transition probabilities. It updates policy in place
        # Get environment to calculate policy for - root environment if not provided
        env = self.env if env is None else env        
        # If adjacency is not provided, use own adjacency matrix
        adjacency = env['adjacency'] if adjacency is None else adjacency
        # Make sure reward locations are in a list
        reward_locations = [reward_locations] if type(reward_locations) is not list else reward_locations
        # Create boolean vector of reward locations for matrix indexing
        is_reward_location = np.zeros(env['n_locations'], dtype=bool)
        is_reward_location[reward_locations] = True
        # Calculate distances between all locations based on adjacency matrix - this doesn't take transition
        # probabilities into account!
        dist_matrix = self.get_distance(adjacency)
        # Fill out minumum distance to any reward state for each action
        for location in original_policy:
            for action in location['actions']:
                action['d'] = np.min(dist_matrix[is_reward_location, np.array(action['transition']) > 0]) if any(
                    action['transition']) else np.inf
        # Calculate policy from softmax over negative distances for every action
        for location in original_policy:
            exp = np.exp(beta * np.array(
                [(1 if opposite else -1) * action['d'] 
                 if action['probability'] > 0 and np.isfinite(action['d'])
                 else -np.inf for action in location['actions']]))
            # For disconnected nodes (e.g. in block) all negative distances can be 0
            for action, a_exp in zip(location['actions'], exp):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action['probability'] = a_exp / sum(exp) if sum(exp) > 0 else 0  
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
    
    def policy_zero_to_self_transition(self, original_policy):
        # Make sure all unavailable actions, indicated by all-zero transitions,
        # now get a stand-still transition
        for l_i, location in enumerate(original_policy):
            for action in location['actions']:
                if np.sum(action['transition']) == 0:
                    action['transition'] = np.eye(len(original_policy))[l_i]
        # Return udpated policy in case it's needed for further processing
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
                                    
    def set_policy(self, new_policy, env=None, name=None):
        # If environment dictionary not provided: use root level environment
        old_policy = self.env['locations'] if env is None else env['locations']
        # Copy input policy actions to the requested component dict in actions
        for l_i, new_location in enumerate(new_policy):
            for a_i, new_action in enumerate(new_location['actions']):
                # Select action dictionary to update
                old_dict = old_policy[l_i]['actions'][a_i] if name is None \
                    else old_policy[l_i]['actions'][a_i]['components'][name]
                # Run through keys in new dictionary and update corresponding old keys
                for new_key, new_val in new_action.items():
                    old_dict[new_key] = new_val

    def get_policy(self, env=None, name=None, in_place=True):
        # If environment dictionary not provided: use root level environment
        env = self.env if env is None else env
        # Get policy (=list of locations) but with actions for requested component only
        # If name is not provided: return a 'clean' copy of the default policy, without components
        policy = []
        for location in env['locations']:
            # Select all location info except actions
            curr_location = select_dict_entries(location, ['action', 'env'], include=False)
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
    
    def set_locs_on_grid(self, locations, req_rows=None, req_cols=None):
        # Change the x, y coordinates of locations so they end up on a square grid
        xs = [l['x'] for l in locations]
        ys = [l['y'] for l in locations]
        # Find rows and columns: number of distinct x and y values
        rows = len(set(ys))
        cols = len(set(xs))
        # Find max and min for x and y
        xlim = [min(xs), max(xs)]
        ylim = [min(ys), max(ys)]
        # Set required nr of rows and cols, if not provided
        req_rows = max(rows, cols) if req_rows is None else req_rows
        req_cols = max(rows, cols) if req_cols is None else req_cols
        # Then scale (subtract mean, scale, add mean) to get required rows/cols
        ys = [(y - 0.5 * (ylim[0] + ylim[1])) * rows / req_rows
              + 0.5 * (ylim[0] + ylim[1]) for y in ys]
        xs = [(x - 0.5 * (xlim[0] + xlim[1])) * cols / req_cols
              + 0.5 * (xlim[0] + xlim[1]) for x in xs]            
        # Finally update all location positions
        for location, x, y in zip(locations, xs, ys):
            location['x'] = x
            location['y'] = y    

    def get_internal_locations(self, env=None, include=None):
        # If environment dictionary not provided: use root level environment
        env = self.env if env is None else env        
        # Find 'interal' locations: locations with all actions available in base environment
        include = [i for i in range(env['n_locations'])] if include is None else include
        # Calculate number of actions available in each location, to determine if location is on the border
        n_actions_available = [sum([action['components']['base']['probability']>0 
                                    for action in location['actions']]) 
                                    for location in env['locations']]
        # Only locations with the maximum actions available can be shiny (which excludes borders)
        return [loc_i for loc_i, n_a in enumerate(n_actions_available) 
                if n_a > (max(n_actions_available) - 4) and loc_i in include]        
    
    def get_components(self):
        # Find all components on high level
        comp_high = [k for k in self.env['components'].keys()]
        comp_high.sort()
        # Find all components on low level
        comp_low = list(set(flatten([[k for k in loc['env']['components'].keys()]
                                     for loc in self.env['locations']])))
        comp_low.sort()        
        # Return list of components on both levels
        return [comp_high, comp_low]
        
    def get_copy(self):
        # Create copy with identical setup
        return World(environments=self.input['environments'],
                     env_specs=self.input['env_specs'])
    
    def get_adjacency(self, policy):
        # Get adjacency matrix for given policy: 1 if there is any action that 
        # with probability > 0 takes you from location i (row) to location j (col)
        return [[1*any([action['probability'] > 0 and action['transition'][l_to] > 0 
                        for action in loc_from['actions']]) 
                 for l_to in range(len(policy))] for loc_from in policy]
    
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
            