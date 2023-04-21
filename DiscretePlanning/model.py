#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:58:14 2021

@author: jbakermans
"""

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import random
from collections import namedtuple, deque
from datetime import datetime

# Base model with standard functionality for composing policies
class Model():    

    def __init__(self, env, pars=None):
        # Initialise environment: set home, retrieve distances
        self.env = self.init_env(env)
        # Initialise parameters, copying over all fields and setting missing defaults
        self.pars = self.init_pars(pars)
        
        # Get representations for each location from components
        self.reps = self.init_representations(self.env['env'])
        # Update environment representations
        self.env['env'] = self.update_rep(self.env['env'], self.reps)
        
        # Build policy from representations
        self.pol = self.init_policy(self.env['env'], self.reps)
        
    def reset(self, env, pars=None):
        # Reset a model to restart fresh (e.g. with new params, or different env)
        # This simple calls __init__, but has a separate name for clarity
        self.__init__(env, pars)
        
    def simulate(self):
        # Simulate exploration: move around environment, exploring components
        # explore = self.sim_explore()
        # Actually, we know what we get at the end of exploration: 
        # Full representation at reward location
        explore = [self.exp_final()]        
        # Copy final step of explore walk
        final_step = dict(explore[-1])
        # Start testing phase
        self.pars['test'] = True
        # Main simulations are going to change what happens after exploration
        if self.pars['experiment'] == 1:
            # 1. Continue by walking, both with or without latent learning
            # Clear memory
            self.pars['memory'] = [[] for _ in range(self.env['env'].n_locations)]
            # Set current representation to only observed representation
            final_step['representation'] = self.rep_observe(final_step['location'])
            # And continue walk
            test = self.sim_explore([final_step])
        elif self.pars['experiment'] == 2:
            # 2. Continue by replay, both for Bellman backups and memory updates
            test = self.sim_replay(final_step)
        # And return both
        return explore, test
        
    def sim_explore(self, explore=None):
        # Start with empty list that will contain info about each explore step
        explore = [self.exp_step([])] if explore is None else explore
        # Exploration stage: explore environment until completing representation
        while (self.pars['test'] and len(explore) < self.pars['test_max_steps']) \
            or (not self.pars['test'] and
                (any([any([r is None for r in rep]) 
                     for rep in explore[-1]['representation'].values()])
                 and len(explore) < self.pars['explore_max_steps'])):
            # Run step
            curr_step = self.exp_step(explore)
            # Evaluate step
            curr_step = self.exp_evaluate(curr_step)
            # Store all information about current step in list
            explore.append(curr_step)  
            # Display progress
            if self.pars['print']:
                print('Finished explore', len(explore), '/', self.pars['explore_max_steps'], 
                      'at', curr_step['location'], 
                      'reps at', curr_step['representation'])
        return explore
    
    def sim_replay(self, start_step):
        # Initialise replay weights if not set yet
        self.pars['replay_q_weights'] = np.zeros(
            (self.env['env'].n_locations, 
             self.env['env'].n_actions)) \
            if self.pars['replay_q_weights'] is None else self.pars['replay_q_weights']
        # Start with empty list that will contain lists of replays
        replay = []
        # Set k-variable thate counts replay steps to 0 to indicate start of replay
        start_step['k'] = 0
        # Each replay consists of a list of replay steps, which are dicts with step info
        for j in range(self.pars['replay_n']):
            # Start replay at current representation
            curr_replay = []
            curr_step = start_step
            # Replay until reaching max steps, or terminal state
            while curr_step['k'] < self.pars['replay_max_steps'] \
                and curr_step['location'] not in self.pars['replay_terminal']:
                    # Take replay step
                    curr_step = self.replay_step(curr_step)
                    # Evaluate step
                    curr_step = self.replay_evaluate(curr_step)                
                    # Add this replay step to current replay list
                    curr_replay.append(curr_step)
            # Display progress
            if self.pars['print']:            
                print('Finished replay', j, '/', self.pars['replay_n'], 
                      'in', curr_step['k'], 'steps at', curr_step['location'])
            # After finishing replay iteration, add this replay to list of replays
            replay.append(curr_replay)
        # And return replay list
        return replay                    

    def exp_step(self, explore):
        # Find new location in this step
        if len(explore) == 0:
            # First step: start from home
            curr_loc, curr_rep = self.exp_init()
        else:
            # All other steps: get new location and representation from transition
            curr_loc, curr_rep = self.exp_transition(explore[-1])
        # Add representation to memory
        self.rep_encode(curr_loc, curr_rep)
        # Select action for current location
        curr_action = self.get_sample(self.loc_policy(curr_loc, self.pars['explore_pol']))
        # Return summary for this step
        curr_step = {'i': len(explore), 'location': curr_loc,
                     'representation': curr_rep, 'action': curr_action}
        return curr_step         
    
    def exp_init(self):
        # Set initial location and representation for explore walk: any free location      
        curr_loc = random.choice([location['id'] for location in self.env['env'].locations
                                  if location['id'] not in 
                                  flatten([comp['locations']
                                       for comp in self.env['env'].components.values()
                                       if 'locations' in comp.keys()])])
        # Get representation for initial location
        curr_rep = self.rep_observe(curr_loc)
        return curr_loc, curr_rep
    
    def exp_final(self):
        # Set final location and representation for explore walk: reward loc,
        # with all representations set
        curr_loc = random.choice(self.env['reward_locs'])
        # Get representation for this location
        curr_rep = self.rep_observe(curr_loc)
        # But update all representations to correct representation
        curr_rep = {key: [curr_loc for _ in val] for key, val in curr_rep.items()}
        # Choose random action 
        curr_action = self.get_sample(self.loc_policy(curr_loc, self.pars['explore_pol']))
        # And return full step dictionary
        return {'i': 0, 'location': curr_loc, 'representation': curr_rep, 
                'action': curr_action}
        
    def exp_transition(self, prev_step):
        # Collect previous step
        prev_loc = prev_step['location']
        prev_action = prev_step['action']
        prev_rep = prev_step['representation']
        # Transition both location and representation
        curr_loc = self.loc_transition(prev_loc, prev_action)
        curr_rep = self.rep_update(curr_loc, prev_rep, prev_action)
        # And return both
        return curr_loc, curr_rep
    
    def replay_step(self, prev_step):        
        if prev_step['k'] == 0:
            # In the first step of replay: copy location and representation from last real step
            curr_loc, curr_rep = prev_step['location'], prev_step['representation']      
        else:
            # Transition location and representation
            curr_loc, curr_rep = self.replay_transition(prev_step)
        # Add representation to memory
        self.rep_encode(curr_loc, curr_rep)            
        # Select action for current location
        curr_action = self.get_sample(self.loc_policy(curr_loc, self.pars['replay_pol']))
        # Set current step 
        curr_step = {'i': prev_step['i'], 'k': prev_step['k'] + 1, 'location': curr_loc,
                     'representation': curr_rep, 'action': curr_action}
        # And return all information about this replay step
        return curr_step
                
    def replay_transition(self, prev_step):
        # Like explore transition, except no observation - get location from base
        prev_action = prev_step['action']
        prev_rep = prev_step['representation']
        prev_loc = prev_rep['base'][0]
        # Transition new location: base represenation
        # Get new representation by sampling from combined transition and retrieval pdf
        # Might change this, but for now: only use transitioned representation during 
        # replay, ignore the representation probability retrieved from memory
        curr_loc = self.loc_transition(prev_loc, prev_action, comp='base', noise='base')
        curr_rep = self.rep_sample(curr_loc, prev_rep, prev_action, w=1)
        # Update replay q-weights, depending on forward or reverse replay
        if self.pars['replay_reverse']:
            # Get reversed action, in opposite direction of this step
            reversed_action = self.get_opposite_action(
                self.env['env'].locations[prev_loc],
                self.env['env'].locations[curr_loc])
            # Then do update with that reversed action, swapping around curr and prev
            self.pars['replay_q_weights'][:, reversed_action] = self.replay_weight_update(
                prev_loc, prev_rep, curr_rep, reversed_action)
        else:
            # Normal forward q-update
            self.pars['replay_q_weights'][:, prev_step['action']] = self.replay_weight_update(
                curr_loc, curr_rep, prev_step['representation'], prev_step['action'])        
        # Return both
        return curr_loc, curr_rep       
    
    def replay_weight_update(self, curr_loc, curr_rep, prev_rep, prev_action):
        # TD-learn: if the animal thinks it's home, set reward
        reward = 1*(curr_loc in self.env['reward_locs'])
        # Build state from rep
        # curr_state = np.array(self.rep_to_state(curr_rep))
        # prev_state = np.array(self.rep_to_state(prev_rep))
        curr_state = np.eye(self.env['env'].n_locations)[curr_loc]
        prev_state = np.eye(self.env['env'].n_locations)[prev_rep['base'][0]]
        # Calculate TD delta: r + gamma * max_a Q(curr_rep, a) - Q(prev_rep, prev_action)
        delta = reward + self.pars['gamma'] * \
            np.max(np.matmul(curr_state, self.pars['replay_q_weights'])) \
            - np.matmul(prev_state, self.pars['replay_q_weights'][:,prev_action])
        # Return Q weight update using TD rule: w = w + alpha * delta * dQ/dw
        return self.pars['replay_q_weights'][:, prev_action] \
            + self.pars['alpha'] * delta * prev_state        
    
    # def exp_evaluate(self, curr_step):
    #     # Increase number of visits to current location
    #     self.env['env'].locations[curr_step['location']]['visits'] += 1
    #     # Add visit information to current step
    #     curr_step['visits'] = self.env['env'].locations[curr_step['location']]['visits']
    #     # Find if best action probability has >0 optimum probability
    #     p = [action['probability'] for action 
    #          in self.pol[curr_step['location']]['actions']]
    #     curr_step['pol_corr'] = self.pars['optimal_pol'][curr_step['location']][
    #         'actions'][p.index(max(p))]['probability'] > 0
    #     # Return updated step
    #     return curr_step
        
    # def replay_evaluate(self, curr_step):
    #     # Increase number of visits to current location
    #     self.env['env'].locations[curr_step['location']]['visits'] += 1
    #     # Add visit information to current step
    #     curr_step['visits'] = self.env['env'].locations[curr_step['location']]['visits']
    #     # Find if max action Q-val has >0 optimum probability
    #     q = [action['Q'] for action 
    #          in self.env['env'].locations[curr_step['location']]['actions']]
    #     curr_step['q_corr'] = self.pars['optimal_pol'][curr_step['location']][
    #         'actions'][q.index(max(q))]['probability'] > 0       
    #     # Return updated step
    #     return curr_step      
    
    def exp_evaluate(self, curr_step):
        # Increase number of visits to current location
        self.env['env'].locations[curr_step['location']]['visits'] += 1
        # Add visit information to current step
        curr_step['visits'] = self.env['env'].locations[curr_step['location']]['visits']

        # When the state rep is incomplete, see if incomplete policy gets you to goal
        if any([any([v is None for v in val]) 
                for val in curr_step['representation'].values()]):
            curr_step['curr_rep_corr'] = \
                self.pars['part_rep_opt'][curr_step['location']]
        else:
            # Complete rep: use optimal policy provided by full representation
            curr_step['curr_rep_corr'] = \
                self.pars['full_rep_opt'][curr_step['location']]

        # Find whether optimal policy provided by full representation gets to goal
        curr_step['full_rep_corr'] = \
            self.pars['full_rep_opt'][curr_step['location']]

        # Return updated step
        return curr_step
        
    def replay_evaluate(self, curr_step):
        # Increase number of visits to current location
        self.env['env'].locations[curr_step['location']]['visits'] += 1
        # Add visit information to current step
        curr_step['visits'] = self.env['env'].locations[curr_step['location']]['visits']      

        # Find whether optimal policy provided by full representation gets to goal
        curr_step['replay_rep_corr'] = \
            self.pars['full_rep_opt'][curr_step['location']]      
        
        # Find whether following learned q-values gets to goal
        # Find if most likely action is optimal action (>0 prob in optimal pol)
        curr_step['replay_bm_corr'] = self.is_qmap_optimal(curr_step['location'])   
        
        # Return updated step
        return curr_step             
    
    def loc_transition(self, prev_loc, prev_action, comp=None, noise=None):
        # Transition probabilities usually don't depend on component, unless 
        # e.g. you want to use base component to ignore objects
        a = self.env['env'].locations[prev_loc]['actions'][prev_action]
        # Get new location from environment transition of previous action
        curr_loc = self.get_sample(a['transition'] if comp is None
                                   else a['components'][comp]['transition'])
        # Add noise and sample if noise component was provided as input
        return curr_loc if noise is None else self.get_sample(
            self.pars['transition_confusion'][noise][curr_loc])
            
    def loc_policy(self, curr_loc, policy):
        # Get policy from location (i.e. as described in environment object)
        curr_pol = np.array([action['probability'] 
                             for action in policy[curr_loc]['actions']])
        # Sometimes (e.g.) for walls there are no actions available. 
        # Return random one in that case
        return curr_pol if sum(curr_pol) > 0 \
            else [1.0/len(curr_pol) for _ in curr_pol]

    def rep_update(self, curr_loc, prev_rep, prev_action):
        # Get new representation by sampling from combined transition and retrieval pdf
        curr_rep = self.rep_sample(curr_loc, prev_rep, prev_action)
        # Fix any representations that can directly be observed from the enfironment
        curr_rep = self.rep_fix(curr_rep, self.rep_observe(curr_loc))
        # And return new representations
        return curr_rep
    
    def rep_encode(self, curr_loc, curr_rep):
        # Add representation to memory
        self.pars['memory'][curr_loc].append(curr_rep)
    
    def rep_sample(self, curr_loc, prev_rep, prev_action, w=None):
        # path_int_weight specifies how much to weight the path integrated 
        # representation probability versus the memory retrieved probability
        # Use value from parameters if not specified
        w = self.pars['path_int_weight'] if w is None else w
        # Get probability from retrieval - but if w is 1, don't retrieve at all
        # If I retrieve anyway and I don't have path integrated probability,
        # I would *still* use the retrieved probability even if w is 1.
        # So instead, when w is 1, path integrate instead.
        p_retrieve = self.rep_prob_retrieve(curr_loc) if w < 1 \
            else self.rep_prob_transition(prev_rep, prev_action)
        # Get probability from transition (path integration)
        # Similarly, don't actually path integrate if w is 0 - or when w is 1,
        # path integration isn't necessary because p_retrieve is already path integrated
        p_transition = p_retrieve if (w == 0 or w == 1) else \
            self.rep_prob_transition(prev_rep, prev_action)
        # Weight probabilities and sample
        curr_rep = {}
        for name, rep in prev_rep.items():
            # For base represenation: use location that was used for memory retrieval
            if name == 'base':
                curr_rep[name] = [curr_loc]
            else:
                curr_rep[name] = []
                # Run through each location of this representation
                for r_i in range(len(rep)):
                    # Get not-None probabilities
                    p_all = [p for p in [p_transition[name][r_i], p_retrieve[name][r_i]]
                             if p is not None]
                    # Weight probabibilities if both are not None
                    p_weighted = None if len(p_all) == 0 else \
                        p_all[0] if len(p_all) == 1 else \
                            [w * p_trans + (1 - w) * p_ret 
                             for p_trans, p_ret in zip(*p_all)]
                    # Sample from combined probability
                    curr_rep[name].append(
                        None if p_weighted is None else self.get_sample(p_weighted))
        return curr_rep    
    
    def rep_prob_transition(self, prev_rep, prev_action):
        # Transition all representations that have been observed
        curr_prob = {}
        # Get transition probability, including transition noise, for each
        # vector representation
        for name, rep in prev_rep.items():
            # Find whether this component has been observed
            curr_prob[name] = [None if r is None else
                               self.pars['transition_confusion'][name][
                                   self.loc_transition(r, prev_action, comp='base')]
                               for r in rep]
        return curr_prob        
    
    def rep_prob_retrieve(self, curr_loc):
        # Get probability distributions over all representations at memory indexed
        # by current location. Probability ~ # of occurences of representation in memory
        curr_prob = {name: [[0 for _ in range(self.env['env'].n_locations)] for _ in rep]
                     for name, rep in self.rep_observe().items()}
        # Add representations in each memory to probabilities
        for mem in self.pars['memory'][curr_loc]:
            # Go through the different representations of this memory
            for name, rep in mem.items():
                # Add memory to to probability
                for r_i, r in enumerate(rep):
                    if r is not None:
                        curr_prob[name][r_i][r] += 1
        # Calculate representation probabilities from counts - if there are any
        curr_prob = {name: [[p / sum(prob) for p in prob] if sum(prob) > 0 else None
                            for prob in rep]
                     for name, rep in curr_prob.items()}
        return curr_prob    
    
    def rep_observe(self, curr_loc=None):
        # Get observed representation-location for given true-location
        # If true location not provided, return empty representation
        curr_rep = {}
        for name, component in self.env['env'].components.items():
            if component['type'] == 'shiny':
                # Shiny representation: observe if currently at shiny
                curr_rep[name] = [curr_loc] \
                    if curr_loc is not None and curr_loc in component['locations'] \
                    else [None]
            elif component['type'] == 'wall':
                # Wall representation: observe wall edge within
                # one step distance (which is the closest you can get)
                curr_rep[name] = [
                    curr_loc 
                    if curr_loc is not None and self.env['dist'][curr_loc, loc] < 2
                    else None for loc in 
                    [component['locations'][0], component['locations'][-1]]]
            elif component['type'] == 'base':
                # Base representation: observe true location
                curr_rep[name] = [curr_loc if curr_loc is not None else None]
        # Return observed representation
        return curr_rep

    def rep_fix(self, curr_rep, observed_rep):
        # Update representation from observed representation, where available
        fixed_rep = {}
        for name, rep in curr_rep.items():
            # Copy over observed representation, if there is one
            fixed_rep[name] = [r_c if r_o is None else r_o
                               for r_o, r_c in zip(observed_rep[name], rep)]
        return fixed_rep
                
    def rep_to_state(self, curr_rep):
        # Collect representation vector for each representation location
        state = []
        for comp, rep in sorted(curr_rep.items()):
            loc_len = len(rep)
            rep_len = len(self.reps[comp][0])  
            rep_element_len = len(self.reps[comp][0][0])
            for loc_i, curr_loc in enumerate(rep):                
                state.append(
                    flatten([[0 for _ in range(rep_element_len)] 
                             for _ in range(int(rep_len/loc_len))] if curr_loc is None \
                             else self.reps[comp][curr_loc][
                                 (loc_i * int(rep_len/loc_len)):
                                     ((loc_i + 1) * int(rep_len/loc_len))]))
        return flatten(state)
    
    def init_pars(self, pars_in):
        # Get default dictionary
        pars_dict = self.init_defaults();
        # If any parameters were provided: combine defaults and provided values
        if pars_in is not None:
            # Then copy over all values provided, overwriting defaults if existing
            for category in pars_in.keys():
                # If this dict value is a dict itself: copy all it contains
                # (but keep defaults for values it doesn't contain!)
                if isinstance(pars_in[category], dict):
                    for key, val in pars_in[category].items():
                        pars_dict[category][key] = val
                # If this dict entry is just a value: copy that value
                else:
                    pars_dict[category] = pars_in[category]
        # Return dictionary that combines defaults and provided parameters
        return pars_dict
                
    def init_defaults(self):
        # Create dictionary for all parameters
        pars_dict = {}
        # Set default transition noise for all components
        pars_dict['transition_noise'] = {key: 0 for 
                                         key in self.env['env'].components.keys()}
        # Set confusion matrix: which locations can be confused?
        pars_dict['transition_confusion'] = {
            key: self.get_mat_confusion(val, self.env['env'].get_policy('base', in_place=False))
            for key, val in pars_dict['transition_noise'].items()}
        pars_dict['path_int_weight'] = 0.2
        pars_dict['explore_max_steps'] = 10
        pars_dict['test_max_steps'] = 20        
        pars_dict['replay_n'] = 20
        pars_dict['replay_max_steps'] = 15
        pars_dict['replay_q_weights'] = None
        pars_dict['replay_reverse'] = False
        pars_dict['replay_terminal'] = []
        pars_dict['print'] = True
        pars_dict['memory'] = [[] for _ in range(self.env['env'].n_locations)]
        pars_dict['explore_pol'] = self.env['env'].policy_random(
            self.env['env'].get_policy(in_place=False))        
        pars_dict['replay_pol'] = self.env['env'].policy_random(
            self.env['env'].get_policy(in_place=False))        
        # self.env['env'].policy_distance(
        #     self.env['env'].get_policy(in_place=False), self.env['reward_locs'],
        #     opposite=True)
        pars_dict['optimal_pol'] = self.get_policy_optimal()
        pars_dict['optimal_dist'] = np.min(self.env['env'].get_distance(
            self.env['env'].get_adjacency(pars_dict['optimal_pol']))[
                :, self.env['env'].components['reward']['locations']], axis=-1)
        pars_dict['partial_pol'] = self.get_policy_optimal(name='base')
        pars_dict['full_rep_opt'] = [self.is_policy_optimal(l, pol=pars_dict['optimal_pol'],
                                                            dist=pars_dict['optimal_dist'][l])
                                     for l in range(self.env['env'].n_locations)]
        pars_dict['part_rep_opt'] = [self.is_policy_optimal(l, pol=pars_dict['partial_pol'],
                                                            dist=pars_dict['optimal_dist'][l])
                                     for l in range(self.env['env'].n_locations)]        
        pars_dict['rep_control'] = False
        pars_dict['experiment'] = 1
        pars_dict['test'] = False
        pars_dict['alpha'] = 0.8
        pars_dict['gamma'] = 0.7
        # Optionally use fast representations to speed things up in square environments
        pars_dict['use_fast_reps'] = False
        # Return default dictionar
        return pars_dict        
                
    def init_env(self, env):
        # Create dictionary of environment variables
        env_dict = {}
        # Create environment from json file in envs directory
        env_dict['env'] = env
        # To indicate an action isn't available, the environment may have all-zero
        # transition probabilities for that action. That's problematic if the model
        # tries to take unavailable actions, so set them all to self-transitions
        env_dict['env'].set_policy(
            env_dict['env'].policy_zero_to_self_transition(env_dict['env'].get_policy()))
        for comp in env_dict['env'].components.keys():
            env_dict['env'].set_policy(
                env_dict['env'].policy_zero_to_self_transition(
                    env_dict['env'].get_policy(comp)), comp)            
        # Get where rewards are in current environment: any shiny location
        env_dict['reward_locs'] = self.get_reward_locs(env_dict['env'])
        # Get distance between all locations in base policy
        env_dict['dist'] = env_dict['env'].get_distance(
            env_dict['env'].get_adjacency(env_dict['env'].get_policy('base',in_place=False)))
        # Start with zero visits to each location
        for location in env_dict['env'].locations:
            location['visits'] = 0
        # Return environment dictionary
        return env_dict
    
    def init_representations(self, env):
        # Create representation at every location for each component
        rep_dict = {}
        # Run through environment components
        for name, component in env.components.items():
            if self.pars['rep_control']:
                rep_dict[name] = self.representations_to_locations(env, [0])
            else:
                if component['type'] == 'shiny':
                    rep_dict[name] = self.get_representation_shiny(env, name)                
                elif component['type'] == 'wall':
                    rep_dict[name] = self.get_representation_wall(env, name)
                elif component['type'] == 'base':
                    rep_dict[name] = self.get_representation_base(env, name)                
                else:
                    print('Type ' + component['type'] + 
                          ' of component ' + name + 'not recognised.')
            # Print progress, because representations take time to compute
            if self.pars['print']:
                print('Finished representation ' + name)
        return rep_dict
    
    def init_policy(self, env, reps):
        # By default, simply use the policy provided by environment
        return env.get_policy(in_place=False)   

    def update_pol(self, env, pol):
        # Update base policy
        env.set_policy(pol)
        return env
    
    def update_rep(self, env, rep):
        # Add representations to locations in environment
        for l_i, location in enumerate(env.locations):
            # Add separate component representations
            for comp in rep.keys():
                location['components'][comp]['representation'] = \
                    rep[comp][l_i]
            # Add concatenated location representation
            location['representation'] = flatten([flatten([r for r in rep[comp][l_i]]) 
                                              for comp in sorted(rep.keys())])
        return env
        
    def get_representation_shiny(self, env, name):
        # Create representation towards shiny object
        return self.representations_to_locations(env, env.components[name]['locations'])
    
    def get_representation_wall(self, env, name):
        # Create representation towards start of wall
        wall_start = self.representations_to_locations(env, [env.components[name]['locations'][0]])
        # Create representation towards end of wall
        wall_end = self.representations_to_locations(env, [env.components[name]['locations'][-1]])
        # Combine representations into wall representation
        return [start + end for start, end in zip(wall_start, wall_end)]

    def get_representation_base(self, env, name):
        # Create representation towards object in top-left corner
        # return self.representations_to_locations(env, [0])
        # Actually, save some time by not having a base representation
        return [[[] for _ in range(env.n_actions)] for _ in range(env.n_locations)]
    
    def representations_to_locations(self, env, locs):
        if self.pars['use_fast_reps']: return self.representations_to_locations_fast(env, locs)
        # Find representation for every location in env
        # Get number of neurons for this component, given by longest sequence of each action
        # Then distance along each action can be expressed by one-hot vector for that action
        # Add one additional neuron (the first) for when action is not on path        
        # Add another neuron in front for when action is in opposite direction to goal
        neurons = [n + 2 for n in self.max_actions(env)]
        # Initialise empty representations for each location
        representations = [[[0 for _ in range(n)] for n in neurons] 
                           for _ in range(env.n_locations)]
        # Get optimal policy towards current requested object.
        # Careful: this relies on adjacency, which SHOULD NOT reflect walls
        policy = env.policy_optimal(env.policy_distance(env.get_policy('base',False), locs))
        # Now find representation for each location, by finding number of actions
        # of each type towards shiny under optimal policy
        for l_i, location in enumerate(env.locations):
            # Find actions on path from current location to goal
            actions_to, final_loc = self.actions_on_path(policy, location, locs)
            # Build policy from final location back to current location
            policy_return = env.policy_optimal(env.policy_distance(env.get_policy('base',False), 
                                                                   [l_i]))
            # And find actions along opposite path, from goal to current location
            actions_from, _ = self.actions_on_path(policy_return, final_loc, [l_i])
            # Now count the number of times each action is taken
            action_number = [sum([a == a_i for a in actions_to]) for a_i in range(env.n_actions)]
            # Then set one-hot representation for that action distance
            for a_i, (a_n, a_r) in enumerate(zip(action_number, representations[l_i])):
                if a_i in actions_from or neurons[a_i] == 2:
                    # If the action is part of return actions: note as opposite
                    a_r[0] = 1
                else:
                    # Else: just count the numer of actions to goal
                    a_r[a_n + 1] = 1
        return representations
    
    def representations_to_locations_fast(self, env, locs):
        # This is a hacky-but-much-faster version of representations_to_locations
        # It assumes a bunch of things:
        # 1. Square environments, so max times an action can be taken = sqrt(locs)
        # 2. Square grids, so there are 4 actions, in order: N, E, S, W
        # 3. The x, y coordinates of locations follow the grid
        # If these are true, then this will produce the same reps, but much faster
        neurons = [int(np.sqrt(env.n_locations)) + 2 for _ in range(env.n_actions)]
        # Initialise empty representations for each location
        representations = [[[0 for _ in range(n)] for n in neurons] 
                           for _ in range(env.n_locations)]
        # Now find representation for each location, by finding number of actions
        # of each type towards shiny under optimal policy
        for l_i, location in enumerate(env.locations):
            # Find nearest target location
            loc = locs[np.argmin([abs(env.locations[l]['x'] - location['x']) + 
                                  abs(env.locations[l]['y'] - location['y']) for l in locs])]
            # Find nr of N, E, S, W from location to target: just coordinate difference
            action_number = [- env.locations[loc]['y'] + location['y'],
                             env.locations[loc]['x'] - location['x'],
                             env.locations[loc]['y'] - location['y'],
                             - env.locations[loc]['x'] + location['x']];
            # Convert coordinate number to integer steps
            action_number = [int(np.round(n * np.sqrt(env.n_locations))) 
                             for n in action_number]
            # Then set one-hot representation for that action distance
            for a_i, (a_n, a_r) in enumerate(zip(action_number, representations[l_i])):
                if a_n < 0:
                    # If the target is in opposite action direction: first neuron
                    a_r[0] = 1
                else:
                    # Else: just count the numer of actions to goal
                    a_r[a_n + 1] = 1
        return representations    
    
    def max_actions(self, env):
        # Find longest possible sequence of single action
        longest_action_sequence = []        
        for action in range(env.n_actions):
            # Find how far you can get for each location
            curr_longest = 0
            # If this is the first action, and environment has stand-still: keep 0
            if not (action == 0 and env.has_self_actions()):
                # Location iterator var is changed in for loop - ok in Python, not in C
                for location in env.locations:
                    # Start at 1: not taking the action also counts
                    curr_length = 1
                    # Get action from base component
                    curr_action = location['actions'][action]['components']['base']
                    # Keep trying to take current action until action not available
                    while curr_action['probability'] > 0 \
                        and sum(curr_action['transition']) > 0:
                        curr_length += 1
                        location = env.locations[
                            np.argmax(curr_action['transition'])]
                        curr_action = location['actions'][action]['components']['base']                        
                    # After action not being available: update longest path
                    curr_longest = max(curr_longest, curr_length)
            # Now that we've found the longest sequence for current action: append
            longest_action_sequence.append(curr_longest)
        # Return longest sequence of actions possible for each action
        return longest_action_sequence
    
    def actions_on_path(self, policy, location, goals):
        # Keep list of actions on path to goals
        actions = []
        # Follow policy until at shiny
        while location['id'] not in goals:
            # Find highest probability action for this location in policy
            action = np.argmax([a['probability'] 
                                     for a in policy[location['id']]['actions']])
            # Get transition given that action
            location = policy[np.argmax(location['actions'][action]['transition'])]
            # Store action taken in action list
            actions.append(action)
        # Return action list
        return actions, location

    def get_opposite_action(self, loc_from, loc_to, action=None):
        # If action provided: assume grid world, with N-E-S-W actions
        if action is not None:
            return [2,3,0,1][action]
        else:
            # Action not provided, find probabilities of going in opposite direction
            ps = [action['transition'][loc_from['id']] for action in loc_to['actions']]
            # Randomly pick action from the ones with the highest probabilities
            return random.choice([i for i, p in enumerate(ps) if p == max(ps)])
    
    def get_mat_confusion(self, noise, policy):
        # Create confusion matrix between locations, where each location can be
        # confused for its neighbours with a probability given by noise
        # Confusion matrix: C_ij is probability that REAL i gets REPLACED BY j
        # Get distance between locations from adjacency from policy
        dist = self.env['env'].get_distance(self.env['env'].get_adjacency(policy))
        # Set confusion to {1-noise: correct, noise: neighbour}
        loc_confusion = noise * (dist == 1) \
            / np.reshape(np.sum(dist==1,1),[dist.shape[0],1]) \
                + (1 - noise) * np.eye(dist.shape[0])
        return loc_confusion    
    
    def get_reward_locs(self, env=None):
        # Find all rewarded locations in environment from shiny components
        # Use own environment if environment not provided
        env = self.env['env'] if env is None else env
        return flatten([comp['locations'] for comp in env.components.values()
                    if comp['type'] == 'shiny'])
        
    def get_policy_optimal(self, env=None, name=None):
        # Use default env if env not supplied
        env = self.env['env'] if env is None else env        
        # Calculate optimal policy in this environment
        opt_pol = env.get_policy(in_place=False, name=name)
        opt_pol = env.policy_optimal(env.policy_distance(
                opt_pol, self.get_reward_locs(env), 
                adjacency=env.get_adjacency(opt_pol)))
        return opt_pol
    
    def get_sample(self, probability):
        # Sample from array containing probability distribution
        return np.random.choice(len(probability), p=probability)    
    
    def is_policy_optimal(self, loc, env=None, pol=None, dist=None):
        # Use default env if env not supplied
        env = self.env['env'] if env is None else env
        # Use optimal pol if pol not supplied
        pol = self.pars['optimal_policy'] if pol is None else pol
        # Get distance along optimal policy if not provided
        dist = self.pars['optimal_dist'][loc] if dist is None else dist
        # Get goals for this environment
        goals = self.get_reward_locs(env)
        # Track path taken
        path = [loc]
        # Run around until at goal, or hitting same location twice
        while path[-1] not in goals \
            and path[-1] not in path[:-2]:
            # Find actions with greatest probability
            p = [action['probability'] for action 
                 in pol[path[-1]]['actions']]
            # Sample randomly from those actions
            a = np.random.choice([i for i, p_ in enumerate(p) if p_ == max(p)])
            # Transition to next state for that action, or to self if action unavailable
            path.append(self.get_sample(
                env.locations[path[-1]]['actions'][a]['transition'])
                if env.locations[path[-1]]['actions'][a]['probability'] > 0
                else path[-1])
        # Return whether final step is goal location, and path length is optimal
        return path[-1] in goals and len(path) == (dist + 1)
    
    def is_qmap_optimal(self, loc, env=None, qs=None, dist=None):
        # Use default env if env not supplied
        env = self.env['env'] if env is None else env
        # Grab q-values from parameters if not supplied
        qs = self.pars['replay_q_weights'] if qs is None else qs
        # Get distance along optimal policy if not provided
        dist = self.pars['optimal_dist'][loc] if dist is None else dist
        # Get goals for this environment
        goals = self.get_reward_locs(env)
        # Track path taken
        path = [loc]
        # Run around until at goal, or hitting same location twice
        while path[-1] not in goals \
            and path[-1] not in path[:-2]:
            # Find actions with greatest q-value
            q = np.matmul(np.eye(env.n_locations)[path[-1]], qs)
            # Sample randomly from those actions            
            a = np.random.choice([i for i, q_ in enumerate(q) if q_ == max(q)])
            # Transition to next state for that action, or to self if action unavailable
            path.append(self.get_sample(
                env.locations[path[-1]]['actions'][a]['transition'])
                if env.locations[path[-1]]['actions'][a]['probability'] > 0
                else path[-1])
        # Return whether final step is goal location, and path length is optimal
        return path[-1] in goals and len(path) == (dist + 1)  
    
class CalcPolicy(Model):
    
    def init_policy(self, env, reps):
        # Create representation at every location for each component
        pol_dict = {}
        # Run through environment components
        for name, representation in reps.items():
            if env.components[name]['type'] == 'shiny':
                pol_dict[name] = self.get_policy_shiny(env, representation)                
            elif env.components[name]['type'] == 'wall':
                pol_dict[name] = self.get_policy_wall(env, representation)                
            elif env.components[name]['type'] == 'base':
                pol_dict[name] = self.get_policy_base(env, representation)                
            else:
                print('Type ' + env.components[name]['type'] + 
                      ' of component ' + name + 'not recognised.')
        # Update environment for new policies
        env = self.policy_update(env, pol_dict)
        # Calculate full policy by combining components
        env = self.policy_compose(env, reps)
        return env.get_policy()  
    
    def get_policy_shiny(self, env, rep):
        policy = []
        # Run through locations and their representations
        for location, representation in zip(env.locations, rep):
            # Policy towards shiny: follow actions that are allowed in base
            curr_pol = [np.argmax(a_r) > 1 and 
                        location['actions'][a_i]['components']['base']['probability'] > 0
                        for a_i, a_r in enumerate(representation)]
            # Normalise policy and add to policy list
            policy.append(self.policy_normalise(curr_pol))            
        return policy

    def get_policy_wall(self, env, rep):
        policy = []
        # Run through locations and their representations
        for location, representation in zip(env.locations, rep):
            # Find where location is relative to wall: find distances to both ends
            action_distances = [[np.argmax(representation[a_i]) - 1,
                    np.argmax(representation[a_i + env.n_actions]) - 1]
                    for a_i in range(env.n_actions)]
            # Find action where distance to both wall ends is larger than 0
            action_both_nonzero = [d[0] > 0 and d[1] > 0 for d in action_distances]
            # Find action where one wall end is same, other is opposite direction
            action_one_opposite = [(d[0] > -1 and d[1] == -1) or (d[0] == -1 and d[1] > -1)
                                  for d in action_distances]      
            if sum(action_one_opposite) > 0:
                # If there's an action towards one wall end, opposite to other: blocked along long side
                # Find non-zero distances for actions parallel to wall (one d==-1, one d > -1)
                min_dist = min([max(d) for d in action_distances
                                if (d[0] == -1 and d[1] > -1) or (d[0] > -1 and d[1] == -1)])
                # Policy follows shortest way around wall
                curr_pol = [((d[0] == min_dist and d[1] == -1) or (d[0] == -1 and d[1] == min_dist))
                            and location['actions'][a_i]['components']['base']['probability'] > 0
                            for a_i, d in enumerate(action_distances)]
                print('Location', location['id'], 'blocked')            
            elif sum(action_both_nonzero) == 1:
                # If there is one action with both non-zero distances: blocked along short side
                # Take action that doesn't go towards or opposite of wall
                curr_pol = [d[0] == 0 and d[1] == 0
                            and location['actions'][a_i]['components']['base']['probability'] > 0
                            for a_i, d in enumerate(action_distances)]
            elif sum(action_both_nonzero) == 2:
                # If there are two action with both non-zero distances: free
                # Policy is random over allowed actions
                curr_pol = [action['components']['base']['probability'] > 0
                            for action in location['actions']]                
            # Normalise policy and add to policy list
            policy.append(self.policy_normalise(curr_pol))            
        return policy
    
    def get_policy_base(self, env, rep):
        policy = []
        # Run through locations and their representations
        for location, representation in zip(env.locations, rep):
            # Random policy: any action allowed in base
            curr_pol = [location['actions'][a_i]['components']['base']['probability'] > 0
                        for a_i, a_r in enumerate(representation)]
            # Normalise policy and add to policy list
            policy.append(self.policy_normalise(curr_pol))            
        return policy               
    
    def policy_update(self, env, pol):
        # Run through environment components
        for name, policies in pol.items():
            # Get current policy in environment
            policy = env.get_policy(name=name, in_place=True)
            # Update policies
            for l_i, location in enumerate(policy):
                for a_i, action in enumerate(location['actions']):
                    action['probability'] = policies[l_i][a_i]
            # And set policy in environment
            env.set_policy(policy, name=name)
        # Return environment for further use
        return env
            
    def policy_normalise(self, pol):
        # Get sum of policy
        pol_sum = sum(pol)
        # Normalise by sum, or set policy as one-hot of first action if sum is 0
        return [p / pol_sum for p in pol] if pol_sum > 0 \
            else [1.0 * (i == 0) for i in range(len(pol))]        
            
    def policy_compose(self, env, rep):
        # Full policy is going to consist of weighting across components
        # Weights depend can depend on representation: e.g. only include obstacle
        # if it's blocking a reward
        is_shiny = [key for key, val in env.components.items()
                    if val['type'] == 'shiny']
        is_wall = [key for key, val in env.components.items()
                    if val['type'] == 'wall']
        # Calculate weighting for each location
        for l_i, location in enumerate(env.locations):
            # Start with 0 weights for each component
            curr_weights = {key: 0 for key in env.components.keys()}
            # Select currently relevant shiny: closest shiny
            if len(is_shiny) > 0:
                # Get distances for each shiny component
                shiny_dist = [sum([np.argmax(r[1:]) for r in rep[name][l_i]]) 
                              for name in is_shiny]
                # Get shiny at smallest distance
                curr_shiny = [n for n, d in zip(is_shiny, shiny_dist)
                              if d == min(shiny_dist)][0]
            else: 
                # No shiny object
                curr_shiny = None
            # Find walls that are blocking current shiny
            if len(is_wall) > 0 and curr_shiny is not None:
                # Get shiny representation
                curr_shiny_rep = rep[curr_shiny][l_i]
                # Get shiny distances
                curr_shiny_dist = [np.argmax(r) - 1 for r in curr_shiny_rep]
                # Get wall distance, or -1 if not blocking
                wall_dist = []
                for curr_wall in is_wall:
                    if location['components'][curr_wall]['in_comp']:
                        # Ignore this wall if current location is on wall
                        wall_dist.append(-1)
                    else:
                        # Get current wall representation
                        curr_wall_rep = rep[curr_wall][l_i]
                        # Get current wall distances
                        curr_wall_dist = [[np.argmax(curr_wall_rep[a_i]) - 1,
                                np.argmax(curr_wall_rep[a_i + env.n_actions]) - 1]
                                for a_i in range(env.n_actions)]
                        # Find action towards wall
                        a_to_wall = [i for i, d in enumerate(curr_wall_dist) 
                                     if d[0] > 0 and d[1] > 0 and d[0] == d[1]]
                        # If there are no actions with d>0 to both ends:
                        # wall is oriented with short end towards location
                        a_to_wall = [i for i, d in enumerate(curr_wall_dist) 
                                     if d[0] > 0 and d[1] > 0][0] if len(a_to_wall) < 1 \
                            else a_to_wall[0]
                        
                        # If the distance to shiny is smaller than the minimum to wall:
                        # definitely not blocked. Also not blocked if any distance to
                        # shiny is larger than the minimum non-zero distance to wall
                        if curr_shiny_dist[a_to_wall] <= max(curr_wall_dist[a_to_wall]) \
                            or not any([(w_d[0] < 1 and w_d[1] > -1) 
                                        or (w_d[0] > -1 and w_d[1] < 1)
                                        and (max(w_d) - s_d >= 0)
                                        for a_i, (s_d, w_d)
                                        in enumerate(zip(curr_shiny_dist, curr_wall_dist))
                                        if not a_i == a_to_wall]):
                            wall_dist.append(-1)
                        else:                           
                            wall_dist.append(min(curr_wall_dist[a_to_wall]))
                # Set wall to nearest blocking wall, if there is any - else, None
                if any([d > -1 for d in wall_dist]):
                    curr_wall = [n for n, d in zip(is_wall, wall_dist) 
                                 if d == min([w_d for w_d in wall_dist if w_d > -1])][0]                    
                else:
                    curr_wall = None
            else:
                # No shiny object, so don't care about walls
                curr_wall = None
            # Now calculate weights: 1 for shiny, or base if no shiny
            if curr_shiny is None:
                curr_weights['base'] = 1
            else:
                curr_weights[curr_shiny] = 1
            # And for wall: 1/dist for nearest blocking wall
            if curr_wall is not None:
                curr_weights[curr_wall] = 4 / min([w_d for w_d in wall_dist if w_d > -1])
            # Normalise weights and assign to location
            norm = sum(curr_weights.values())
            for key, val in curr_weights.items():
                location['components'][key]['weight'] = val / norm                
            # Finally: update policy of environment
            policy = [sum([action['components'][comp]['probability'] 
                           * location['components'][comp]['weight'] 
                           for comp in curr_weights.keys()])
                      for action in location['actions']]
            # Only allow available actions and renormalise
            policy = self.policy_normalise([policy[a_i] if action['probability'] > 0
                                            else 0 for a_i, action 
                                            in enumerate(location['actions'])])
            # And finally finally assign normalised probability to root actions
            for a_i, action in enumerate(location['actions']):
                action['probability'] = policy[a_i]
        # Return environment with updated composed policy
        return env       
    
    
class LearnPolicy(Model):
    
    # Add some additional parameters to the defaults
    def init_defaults(self):
        # Inherit parent defaults
        pars_dict = Model.init_defaults(self)
        # Now add a field for the DQN we are going to train
        pars_dict['DQN'] = {}
        # Number of environments to train simultaneously
        pars_dict['DQN']['envs'] = 25
        # Policy inverse temperature
        pars_dict['DQN']['beta'] = 1
        # Temporal discount factor
        pars_dict['DQN']['gamma'] = 0.99
        # Initial epsilon for epsilon-greedy policy
        pars_dict['DQN']['eps_start'] = 0.9
        # Final epsilon for epsilon-greedy policy
        pars_dict['DQN']['eps_end'] = 0.05
        # Steps to decay epsilon for
        pars_dict['DQN']['eps_decay'] = 200
        # Steps after which to update target weights
        pars_dict['DQN']['target_update'] = 1
        # Number of training episodes
        pars_dict['DQN']['episodes'] = 1
        # Number of training steps per episode
        pars_dict['DQN']['steps'] = 200
        # Accuracy threshold to stop training
        pars_dict['DQN']['acc_threshold'] = None      
        # Hidden layer dimension for DQN
        pars_dict['DQN']['hidden_dim'] = None        
        # Number of transitions to sample from memory for traning
        pars_dict['DQN']['mem_sample'] = 5
        # Number of transitions in memory
        pars_dict['DQN']['mem_size'] = 100 * pars_dict['DQN']['mem_sample']
        # Number of maximum steps to try when evaluating performance
        pars_dict['DQN']['max_eval_steps'] = 100
        # Save model parameters after training - set to None to not save
        pars_dict['DQN']['save'] = './trained/' \
            + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pt'      
        # Load model parameters instead of training - set to None to not load
        pars_dict['DQN']['load'] = None #'./trained/20211112_170701.pt'
        return pars_dict
    
    # Use learned 
    def exp_evaluate(self, curr_step):
        # Increase number of visits to current location
        self.env['env'].locations[curr_step['location']]['visits'] += 1
        # Add visit information to current step
        curr_step['visits'] = self.env['env'].locations[curr_step['location']]['visits']
        
        # Set policy net to evaluation mode
        self.policy_net.eval()        
        
        # Get state for current representation
        state = self.rep_to_state(curr_step['representation'])        
        # Find location policy for current representation
        q = self.policy_net(
            torch.tensor(state, dtype=torch.float).view(1,-1)).detach().numpy()
        # Or alternatively: when the state rep is incomplete, take random qs
        if any([any([v is None for v in val]) 
                for val in curr_step['representation'].values()]):
            q = np.random.rand(*q.shape)
        p = self.learn_get_policy_location(
            self.env['env'].locations[curr_step['location']], q, 
            self.pars['DQN']['beta'])
        # Find if most likely action is optimal action (>0 prob in optimal pol)
        curr_step['curr_rep_corr'] = self.pars['optimal_pol'][curr_step['location']][
            'actions'][p.index(max(p))]['probability'] > 0        
        
        # Get state for full representation
        state = self.rep_to_state(
            {key: [curr_step['representation']['base'][0] for _ in val] 
             for key, val in curr_step['representation'].items()})
        # Find location policy for current representation
        q = self.policy_net(
            torch.tensor(state, dtype=torch.float).view(1,-1)).detach().numpy()
        p = self.learn_get_policy_location(
            self.env['env'].locations[curr_step['location']], q, 
            self.pars['DQN']['beta'])
        # Find if most likely action is optimal action (>0 prob in optimal pol)
        curr_step['full_rep_corr'] = self.pars['optimal_pol'][curr_step['location']][
            'actions'][p.index(max(p))]['probability'] > 0          

        # Return updated step
        return curr_step
        
    def replay_evaluate(self, curr_step):
        # Increase number of visits to current location
        self.env['env'].locations[curr_step['location']]['visits'] += 1
        # Add visit information to current step
        curr_step['visits'] = self.env['env'].locations[curr_step['location']]['visits']     
        
        # Set policy net to evaluation mode
        self.policy_net.eval()        

        # Get state for full representation
        state = self.rep_to_state(
            {key: [curr_step['representation']['base'][0] for _ in val] 
             for key, val in curr_step['representation'].items()})
        # Find location policy for current representation
        q = self.policy_net(
            torch.tensor(state, dtype=torch.float).view(1,-1)).detach().numpy()
        p = self.learn_get_policy_location(
            self.env['env'].locations[curr_step['location']], q, 
            self.pars['DQN']['beta'])
        # Find if most likely action is optimal action (>0 prob in optimal pol)
        curr_step['replay_rep_corr'] = self.pars['optimal_pol'][curr_step['location']][
            'actions'][p.index(max(p))]['probability'] > 0          
        
        # Find location policy for current representation
        q = np.matmul(np.eye(self.env['env'].n_locations)[curr_step['location']],
                      self.pars['replay_q_weights'])
        p = self.learn_get_policy_location(
            self.env['env'].locations[curr_step['location']], q, 
            self.pars['DQN']['beta'])
        # Find if most likely action is optimal action (>0 prob in optimal pol)
        curr_step['replay_bm_corr'] = self.pars['optimal_pol'][curr_step['location']][
            'actions'][p.index(max(p))]['probability'] > 0               
        
        # Return updated step
        return curr_step          
    
    def init_policy(self, env, reps):        
        # Try loading policy net
        self.policy_net = self.load_policy_net()
        # If loading wasn't successful: learn policy net
        self.policy_net = self.learn_policy_net() if self.policy_net is None \
            else self.policy_net
        # Get learned policy
        learned_pol = self.learn_get_policy_learned(self.policy_net, env)
        # Use internal representation and environment 
        return learned_pol
    
    def load_policy_net(self):
        # If loading disabled: return None
        if self.pars['DQN']['load'] is None:
            return None
        else:
            try:
                # Get network dimensions
                d_in, d_out = self.learn_init_network()
                # Get wall representation parameters
                wall_length, wall_start = self.learn_init_network_walls()
                # Initialise network
                policy_net = WallDQN(d_in, d_out, wall_length, wall_start,
                                     hidden_dim=self.pars['DQN']['hidden_dim'])
                # Load network parameters
                policy_net.load_state_dict(torch.load(self.pars['DQN']['load']))                
            except FileNotFoundError:
                # File doesn't exist: return none
                return None
        return policy_net
    
    def learn_policy_net(self):
        # Get network dimensions
        d_in, d_out = self.learn_init_network()
        
        # Initialise networks
        policy_net = DQN(d_in, d_out, hidden_dim=self.pars['DQN']['hidden_dim'])
        target_net = DQN(d_in, d_out, hidden_dim=self.pars['DQN']['hidden_dim'])
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        
        # Initialise optimiser
        optimiser = torch.optim.RMSprop(policy_net.parameters())
        # Create a tensor board to stay updated on training progress. 
        # Start tensorboard with tensorboard --logdir=runs
        writer = SummaryWriter(self.pars['DQN']['save'].replace('.pt','_tb'))
        
        # Initialise memory of transitions to sample from
        memory = ReplayMemory(self.pars['DQN']['mem_size'])
        
        # Run training loop
        for i_episode in range(self.pars['DQN']['episodes']):
            # Get all variables to start episode
            envs, reward_locs, location, state = self.learn_init_episode(
                self.env['env'], self.pars['DQN']['envs'])
            # Now run around through environments
            for t in range(self.pars['DQN']['steps']):
                # Calculate current number of training steps
                steps = i_episode * self.pars['DQN']['steps'] + t
                
                # Do transition: choose action, find new state, add to memory
                location, state, memory = self.learn_do_transition(
                    state, steps, policy_net, envs, location, reward_locs, memory)                
                
                # Perform one step of the optimization (on the policy network)
                if steps > self.pars['DQN']['mem_sample']:
                    self.learn_optimise_step(policy_net, target_net, optimiser, memory)
                    
                # Display progress
                if t % 100 == 0:
                    curr_optimality = np.array(
                        self.learn_evaluate_performance(policy_net, envs))
                    curr_arrived = np.logical_and(
                        curr_optimality > -1, 
                        curr_optimality < self.pars['DQN']['max_eval_steps'])
                    curr_failed = np.logical_and(
                        curr_optimality > -1, 
                        curr_optimality == self.pars['DQN']['max_eval_steps'])
                    curr_d = np.mean(curr_optimality[curr_arrived])
                    curr_o = np.sum(curr_arrived) / (np.sum(curr_arrived) 
                                                     + np.sum(curr_failed))
                    print('-- Step', t, '/', self.pars['DQN']['steps'],
                          'optimilaty', curr_o,
                          'mean dist', curr_d,
                          'reward', self.learn_evaluate_reward(memory))
                    writer.add_scalar('Optimality', curr_o, steps)   
                    writer.add_scalar('Distance', curr_d, steps)                       
                    writer.add_scalar('Reward',  
                                      self.learn_evaluate_reward(memory), steps)                
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.pars['DQN']['target_update'] == 0:
                target_net.load_state_dict(policy_net.state_dict())
            # Save policy net at the end of each episode
            if self.pars['DQN']['save'] is not None:
                torch.save(policy_net.state_dict(), self.pars['DQN']['save'])            
            # Display progress
            print('- Finished episode', i_episode, '/', self.pars['DQN']['episodes'])
        # Return trained policy network
        return policy_net
    
    def learn_init_network(self):
        # Find dimensions of input and output of network
        # Input: dimension of state by concatenating all representations
        d_in = sum([sum([len(rep) for rep in comp[0]]) 
                        for comp in self.reps.values()])                        
        # Output: number of actions in environemnt
        d_out = self.env['env'].n_actions
        # Return both
        return d_in, d_out 
    
    def learn_init_network_walls(self):
        # Find length of wall representation, and start index 
        # of each wall in the full representation vector
        wall_length = 0
        # Find where each component starts, and add to array if wall
        wall_start = []
        # Start index that counts position in representation vector at 0
        i = 0
        for key in sorted(self.reps.keys()):
            # Get length of this representation from location 0
            curr_length = sum([len(rep) for rep in self.reps[key][0]])         
            # Check if this is a wall representation
            if self.env['env'].components[key]['type'] == 'wall':
                # If so: current index should be added to wall indices
                wall_start.append(i)
                # And wall length can be set to current length
                wall_length = curr_length
            # Then move the index along by the length of the current representation
            i += curr_length
        # Return the length and the index of wall representations
        return wall_length, wall_start
    
    def learn_init_episode(self, base_env, n_envs):
        # Create environments
        envs = self.learn_get_training_envs(base_env, n_envs)
        # Find reward locations in each environment
        reward_locs = [self.get_reward_locs(env) for env in envs]
        # Initialise locations: random location for each environment
        location = self.learn_get_location(envs)
        # And get state for these locations
        state = self.learn_get_state(location)
        # Return all variables for starting an episode
        return envs, reward_locs, location, state
        
    def learn_do_transition(self, state, steps, policy_net, envs, 
                            location, reward_locs, memory):
        # Select and perform an action
        action = self.learn_get_action(state, steps, policy_net)
        next_location = self.learn_get_transition(envs, location, action)
        reward = self.learn_get_reward(next_location, reward_locs)
        next_state = self.learn_get_state(next_location)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        location = next_location
        state = next_state              
        # Return updated location, state, and memory          
        return location, state, memory
    
    def learn_optimise_step(self, policy_net, target_net, optimiser, memory):
        # Sample transitions from memory
        transitions = memory.sample(self.pars['DQN']['mem_sample'])
        # Concatenate all samples to get one big batch transition
        batch = Transition(*[torch.cat(x) for x in zip(*transitions)])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(batch.state).gather(1, batch.action)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        next_state_values = target_net(batch.next_state).max(1)[0].view(-1,1).detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.pars['DQN']['gamma']) \
            + batch.reward
    
        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
    
        # Optimize the model
        optimiser.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimiser.step()
        # Return model
        return optimiser, policy_net
    
    def learn_get_training_envs(self, real_env, n_envs):
        # Get training environment from real environment for each batch
        envs = []
        for i in range(n_envs):
            print('Creating training env', i)
            # Create new environment from base of real environment
            new_env = real_env.get_full_copy()
            # Check if it's valid: are there any actions available?
            while all([sum([action['probability'] for action in loc['actions']]) == 0
                       for loc in new_env.locations]):
                # Print that resampling is necessary
                print('No valid locations. Resample environment')
                # When reward is closed in by walls: resample environment
                new_env = real_env.get_full_copy()
            # Initialise new environment in same way as real environment
            new_env_dict = self.init_env(new_env)
            # Then only keep environment object from the dictionary
            new_env = new_env_dict['env']
            # Create representations for this environment
            new_reps = self.init_representations(new_env)
            # Add represenations to new environment
            new_env = self.update_rep(new_env, new_reps)
            # Find shiny locations in new environment
            reward_locs = self.get_reward_locs(new_env)
            # Add optimal policy to environment to evaluate network performance
            new_env.pol = new_env.get_policy(in_place=False)
            new_env.pol = new_env.policy_optimal(new_env.policy_distance(
                new_env.pol, reward_locs, adjacency=new_env.get_adjacency(new_env.pol)))
            # Finally add environment to list of training environments
            envs.append(new_env)
        return envs
    
    def learn_get_action(self, state, steps_done, policy_net):
        # Find threshold for epsilon-greedy policy, decaying over time
        eps_threshold = self.pars['DQN']['eps_end'] \
            + (self.pars['DQN']['eps_start'] - self.pars['DQN']['eps_end']) \
                * np.exp(-1. * steps_done /  self.pars['DQN']['eps_decay'])
        # Decide for each environment whether it will use random or policy action
        do_random = torch.tensor(np.random.rand(self.pars['DQN']['envs'], 1)
                                 > eps_threshold, dtype=torch.bool)
        # Select action according to policy
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.            
            action_pol = policy_net(state).max(1)[1].view(-1, 1)
        # Select random action
        action_rand = torch.tensor(np.random.randint(0, self.env['env'].n_actions,
                                                     (self.pars['DQN']['envs'], 1)),
                                   dtype=torch.int64)
        # And return random or policy action according to boolean decision tensor
        return do_random * action_rand + torch.logical_not(do_random) * action_pol
        
    def learn_get_location(self, envs):
        # Randomly select location from each environment
        return [random.choice(env.locations) for env in envs]
    
    def learn_get_transition(self, envs, locs, actions):
        # Randomly sample new location from transition probability of each action
        return [self.get_sample(loc['actions'][int(action)]['transition'])
                for env, loc, action in zip(envs, locs, actions)]   
        
    def learn_get_state(self, locs):
        # Build state tensor: representations of each location
        return torch.stack([torch.tensor(loc['representation'], dtype=torch.float)
                            for loc in locs])
    
    def learn_get_reward(self, locs, reward_locs):
        # If any shiny component is true: location is rewarded
        return torch.stack([torch.tensor(loc['id'] in reward, dtype=torch.float) 
                            for loc, reward in zip(locs, reward_locs)]).view(-1, 1)
    
    def learn_get_policy(self, source_pol, q_vals, beta=1):
        # Run through actions of source policy and choose actions with highest q-val
        for loc, qs in zip(source_pol, q_vals):
            # Get probability for current location
            p = self.learn_get_policy_location(loc, qs, beta)
            # And assign probabilities to actions in source policy
            for action, probability in zip(loc['actions'], p):
                action['probability'] = probability
        # Return updated source policy
        return source_pol
    
    def learn_get_policy_location(self, loc, qs, beta):
        # Get actions to be taken here: highest q and available in source
        p = [np.exp(beta*q) * (a['probability'] > 0)
             for q, a in zip(qs.flatten(), loc['actions'])]
        # Normalise probability 
        p = [curr_p / sum(p) for curr_p in p] if sum(p) > 0 else [0 for _ in p]       
        # Return policy for this location
        return p
    
    def learn_get_policy_learned(self, policy_net, env):
        # Set policy net to evaluation mode
        policy_net.eval()
        # Do forward pass through policy net to get Q-values
        q_vals = policy_net(self.learn_get_state(env.locations)).detach().numpy()
        # Build policy from environment root policy and Q-values
        learned_pol = self.learn_get_policy(env.get_policy(in_place=False), q_vals,
                                            beta=self.pars['DQN']['beta'])
        # Return learned policy
        return learned_pol
    
    def learn_evaluate_performance(self, policy_net, envs):
        # Find for each environment in what percentage of locations the learned
        # policy is optimal
        optimality = []
        # We'll use the policy net to get a policy, but we dont want to optimise
        with torch.no_grad():
            # Run through training environments
            for env in envs:
                # Get learned policy for current environment
                learned_pol = self.learn_get_policy_learned(policy_net, env)
                # Calculate optimality: learned optimal action is optimal at each location
                optimality.append(self.learn_get_optimality(learned_pol, env))
            # Switch policy net back to train mode
            policy_net.train()
        # Return optimality
        return optimality    
    
    def learn_evaluate_policy(self, learned_pol, env=None):
        # Use default env if env not supplied
        env = self.env['env'] if env is None else env
        # Get optimality for learned policy
        return self.learn_get_optimality(learned_pol, env)    
    
    def learn_evaluate_reward(self, memory):
        # Find average reward earned per environment per transitions in memory
        return np.mean([np.mean(mem.detach().numpy()) for mem in memory.get_all(3)])
    
    def learn_get_optimality_old(self, learned_pol, opt_pol):
        # Keep track if learned optimal action is optimal at each location
        is_opt = []
        # Run through locations in learned and optimal policy
        for loc_learned, loc_opt in zip(learned_pol, opt_pol):
            # Find if this location has an optimal action
            if any([action['probability'] > 0
                    for action in loc_opt['actions']]):
                # Get learned action probabilities
                p = [action['probability'] for action in loc_learned['actions']]
                # Find if best action probability has >0 optimum probability
                is_opt.append(loc_opt['actions'][
                    p.index(max(p))]['probability'] > 0)
        # Calculate optimality percentage
        return sum(is_opt)/len(is_opt)
        
    def learn_get_optimality(self, learned_pol, env):
        # Get distances to reward loc under optimal policy
        opt_dist = env.get_distance(
            env.get_adjacency(
                env.get_policy(in_place=False)))
        goals = self.get_reward_locs(env)
        opt_dist = np.array(opt_dist)[:, goals[0]]
        # Keep track of distance under learned policy
        learned_dist = []
        # Run through locations in learned and optimal policy
        for loc_learned, d in zip(learned_pol, opt_dist):
            # Only get learned dist this is not reward or wall loc
            if d > 0 and np.isfinite(d):
                # Greedily follow available actions until hitting shiny or
                # running out of steps
                curr_loc = loc_learned['id']
                curr_d = 0
                while curr_loc not in goals \
                    and curr_d < self.pars['DQN']['max_eval_steps']:
                    # Take best action that is available
                    p = [action['probability'] for action 
                         in learned_pol[curr_loc]['actions']]
                    a = p.index(max(p))
                    curr_loc = self.get_sample(
                        env.locations[curr_loc]['actions'][a]['transition'])
                    curr_d += 1
                # Add current d to learned distance
                learned_dist.append(curr_d - d 
                                    if curr_d < self.pars['DQN']['max_eval_steps'] 
                                    else self.pars['DQN']['max_eval_steps'])
            else:
                learned_dist.append(-1)
        # Calculate optimality percentage
        return learned_dist
    

class FitPolicy(LearnPolicy):
        
    # This model directly learns policy in a supervised manner, instead of via 
    # reinforcement learning. It keeps action selection and environment transitions
    # similar to the LearnPolicy model for better comparability
    
    def learn_policy_net(self):
        # Get network dimensions
        d_in, d_out = self.learn_init_network()
        # Get wall representation parameters
        wall_length, wall_start = self.learn_init_network_walls()
        
        # Initialise networks
        #policy_net = DQN(d_in, d_out, hidden_dim=self.pars['DQN']['hidden_dim'])
        policy_net = WallDQN(d_in, d_out, wall_length, wall_start, 
                             hidden_dim=self.pars['DQN']['hidden_dim'])
        target_net = WallDQN(d_in, d_out, wall_length, wall_start,
                             hidden_dim=self.pars['DQN']['hidden_dim'])
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        
        # Initialise optimiser
        optimiser = torch.optim.Adam(policy_net.parameters())
        # Create a tensor board to stay updated on training progress. 
        # Start tensorboard with tensorboard --logdir=runs
        writer = SummaryWriter(self.pars['DQN']['save'].replace('.pt','_tb'))                
        
        # Initialise memory of transitions to sample from
        memory = ReplayMemory(self.pars['DQN']['mem_size'])
        
        # Keep track of generisation accuracy: performance at start of each episode
        generalisation = np.zeros(self.pars['DQN']['episodes']);
        
        # Run training loop
        for i_episode in range(self.pars['DQN']['episodes']):
            # Get all variables to start episode
            envs, reward_locs, location, state = self.learn_init_episode(
                self.env['env'], self.pars['DQN']['envs'])
            # Now run around through environments
            for t in range(self.pars['DQN']['steps']):
                # Calculate current number of training steps
                steps = i_episode * self.pars['DQN']['steps'] + t
                
                # Display progress
                if t % 100 == 0:
                    curr_optimality = np.array(
                        self.learn_evaluate_performance(policy_net, envs))
                    curr_arrived = np.logical_and(
                        curr_optimality > -1, 
                        curr_optimality < self.pars['DQN']['max_eval_steps'])
                    curr_failed = np.logical_and(
                        curr_optimality > -1, 
                        curr_optimality == self.pars['DQN']['max_eval_steps'])
                    curr_d = np.mean(curr_optimality[curr_arrived]) \
                        if any(flatten(curr_arrived)) else 100
                    curr_o = np.sum(curr_arrived) / (np.sum(curr_arrived) 
                                                     + np.sum(curr_failed))
                    print('-- Step', t, '/', self.pars['DQN']['steps'],
                          'optimilaty', curr_o,
                          'mean dist', curr_d,
                          'reward', self.learn_evaluate_reward(memory))
                    #print('test', curr_optimality)                    
                    writer.add_scalar('Optimality', curr_o, steps)   
                    writer.add_scalar('Distance', curr_d, steps)                       
                    writer.add_scalar('Reward',  
                                      self.learn_evaluate_reward(memory), steps)
                    # The very first step measures generalisation: accuracy on new envs
                    if t == 0:
                        # Write initial accuracy to file
                        generalisation[i_episode] = curr_o
                        # If average of last 5 episodes exceeds training threshold: break and return
                        if self.pars['DQN']['acc_threshold'] is not None and \
                            np.mean(generalisation[max(0, i_episode-2):(i_episode+1)]) \
                                >= self.pars['DQN']['acc_threshold']:
                            # Save policy net and performance at the end of each episode
                            if self.pars['DQN']['save'] is not None:
                                torch.save(policy_net.state_dict(), self.pars['DQN']['save'])
                                with open(self.pars['DQN']['save'].replace('.pt','.npy'), "wb") as f:
                                    np.save(f, self.learn_evaluate_performance(policy_net, envs))
                                with open(self.pars['DQN']['save'].replace('.pt','_init.npy'), "wb") as f:
                                    np.save(f, generalisation)
                            print('Finished after reaching threshold in episode', i_episode,
                                  'with accuracy', curr_o)
                            return policy_net

                # Do transition: choose action, find new state, add to memory
                location, state, memory = self.learn_do_transition(
                    state, steps, policy_net, envs, location, reward_locs, memory)                
                    
                # Get action probabilities and learned values for new locatoin
                next_policy = torch.tensor([[action['probability'] 
                                             for action in env.pol[loc['id']]['actions']]
                                            for loc, env in zip(location, envs)], 
                                           dtype=torch.float)
                next_values = policy_net(state)                    
                
                # Learn action values from correct policy 
                self.learn_optimise_step(policy_net, optimiser, next_policy, next_values,
                                         debug={'writer': writer, 'steps': steps})                                    
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.pars['DQN']['target_update'] == 0:
                target_net.load_state_dict(policy_net.state_dict())
            # Save policy net and performance at the end of each episode
            if self.pars['DQN']['save'] is not None:
                torch.save(policy_net.state_dict(), self.pars['DQN']['save'])
                with open(self.pars['DQN']['save'].replace('.pt','.npy'), "wb") as f:
                    np.save(f, self.learn_evaluate_performance(policy_net, envs))
                with open(self.pars['DQN']['save'].replace('.pt','_init.npy'), "wb") as f:
                    np.save(f, generalisation)                    
            # Display progress
            print('- Finished episode', i_episode, '/', self.pars['DQN']['episodes'])
        # Return trained policy network
        return policy_net    

    def learn_optimise_step(self, policy_net, optimiser, policy, values, debug=None):
        # Compute cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(values, policy)
        if debug is not None: 
            debug['writer'].add_scalar('Loss', loss, debug['steps'])
            
        # Optimize the model
        optimiser.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        optimiser.step()
        # Return model
        return optimiser, policy_net
    
    def learn_get_transition_list(self, envs, locs, actions):
        # Move through list of locations instead of following action
        # Careful: this can bias policy to policy in top-left,
        # because those will be sampled more often. Better: sample randomly
        new_locs = []
        for loc, env in zip(locs, envs):
            # Select new location: simply next one in list
            new_loc = env.locations[loc['id'] + 1 
                                    if loc['id'] < env.n_locations - 2 else 0]
            # But don't want to select wall locations, so skip those
            while sum([action['probability'] for action in new_loc['actions']]) == 0:
                new_loc = env.locations[new_loc['id'] + 1 
                                        if new_loc['id'] < env.n_locations - 2 else 0]
            # Append new location to output
            new_locs.append(new_loc)
        return new_locs
    
    def learn_get_transition(self, envs, locs, actions):
        # Move through list of locations instead of following action
        new_locs = []
        for loc, env in zip(locs, envs):
            # Randomly select new location
            new_loc = env.locations[np.random.randint(env.n_locations)]
            # But don't want to select wall locations, so skip those
            while sum([action['probability'] for action in new_loc['actions']]) == 0:
                new_loc = env.locations[np.random.randint(env.n_locations)]
            # Append new location to output
            new_locs.append(new_loc)
        return new_locs      

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def get_all(self, item):
        return [mem[item] for mem in self.memory]

    def __len__(self):
        return len(self.memory)    
    
class DQN(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super(DQN, self).__init__()
        # Set hidden layer to max of in and out if not provided
        hidden_dim = [int(np.max([in_dim,out_dim]))] \
            if hidden_dim is None else hidden_dim      
        # Create list of all weights
        self.w = torch.nn.ModuleList(
            [torch.nn.Linear(prev_dim, next_dim) for prev_dim, next_dim 
             in zip([in_dim] + hidden_dim, hidden_dim + [out_dim])])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor N_batch x N_actions.
    def forward(self, x):
        for w_i, curr_w in enumerate(self.w):
            # Apply current weights
            x = curr_w(x)
            # All except for output layer: apply non-linearity
            if w_i < len(self.w) - 1:
                x = torch.nn.functional.relu(x)
        return x
    
class WallDQN(torch.nn.Module):
    
    def __init__(self, in_dim, out_dim, wall_dim, wall_start, hidden_dim=None):
        super(WallDQN, self).__init__()
        # Create wall MLP
        self.wall_net = DQN(wall_dim, wall_dim, [wall_dim * 2])
        # Create policy MLP
        self.pol_net = DQN(in_dim, out_dim, hidden_dim)
        # Wall dim is dimension of single wall representation
        self.wall_dim = wall_dim
        # Wall start is list of indices of where each wall representation starts
        self.wall_start = wall_start

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor N_batch x N_actions.
    def forward(self, x):
        # First apply wall net to each wall representation
        for i in self.wall_start:
            x[...,i:(i+self.wall_dim)] = \
                self.wall_net(x[...,i:(i+self.wall_dim)].clone())
        # Then apply policy net to resulting representation
        x = self.pol_net(x)
        return x    
    
def flatten(t):
    return [item for sublist in t for item in sublist]
    

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))     