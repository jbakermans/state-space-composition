#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:53:47 2021

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
        self.reps = self.init_representations(self.env['dict'])
        # Update environment representations
        self.env['dict'] = self.update_rep(self.env['dict'], self.reps)
        
        # Build policy from representations
        self.pol = self.init_policy(self.env['obj'], self.reps)
        
    def reset(self, env, pars=None):
        # Reset a model to restart fresh (e.g. with new params, or different env)
        # This simple calls __init__, but has a separate name for clarity
        self.__init__(env, pars)
        
    def simulate(self):
        # Simulate exploration: move around environment, exploring components
        explore = self.sim_explore()
        # Actually, we know what we get at the end of exploration: 
        # Full representation at reward location
        #explore = [self.exp_final()]        
        # Copy final step of explore walk
        final_step = dict(explore[-1])
        # Main simulations are going to change what happens after exploration
        if self.pars['experiment'] == 1:
            # 1. Continue by walking, both with or without latent learning
            # Clear memory
            self.pars['memory'] = [[] for _ in range(self.env['obj'].n_locations)]
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
        while len(explore) < self.pars['explore_max_steps']:
            # Run step
            curr_step = self.exp_step(explore)
            # Store all information about current step in list
            explore.append(curr_step)  
            # Display progress
            if self.pars['print']:
                print('Finished explore', len(explore), '/', self.pars['explore_max_steps'], 
                      'at', curr_step['location'], 
                      'reps at', curr_step['representation'])
        return explore
    
    def sim_replay(self, start_step, level=None):
        # Start with empty list that will contain lists of replays
        replay = []
        # Set k-variable thate counts replay steps to 0 to indicate start of replay
        start_step['k'] = 0
        # Indicate what level this replay is on
        start_step['level'] = random.choice([0,1]) if level is None else level
        # Each replay consists of a list of replay steps, which are dicts with step info
        for j in range(self.pars['replay_n']):
            # Start replay at current representation
            curr_replay = []
            curr_step = start_step
            # Replay until reaching max steps
            while curr_step['k'] < self.pars['replay_max_steps']:
                # Take replay step
                curr_step = self.replay_step(curr_step)
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
        self.rep_encode(curr_loc, curr_rep, 
                        explore[-1]['location'] if len(explore) > 0 else None)
        # Select action for current location
        curr_action = self.get_sample(self.loc_policy(
            curr_loc[1], self.env['dict']['locations'][curr_loc[0]]['env']['locations']))
        # Return summary for this step
        curr_step = {'i': len(explore), 'location': curr_loc,
                     'representation': curr_rep, 'action': curr_action}
        return curr_step         
    
    def exp_init(self):
        # Set initial location and representation for explore walk: any free location
        loc_high = random.choice([i for i in range(self.env['dict']['n_locations'])])
        loc_low = random.choice(
            [i for i in range(self.env['dict']['locations'][loc_high]['env']['n_locations'])])
        # Location is now list of hierarchical locs
        curr_loc = [loc_high, loc_low]
        # Get representation for initial location
        curr_rep = self.rep_observe(curr_loc)
        return curr_loc, curr_rep
    
    def exp_final(self):
        # Set final location and representation for explore walk: reward loc,
        # with all representations set
        curr_loc, curr_rep = self.exp_init()
        # But update all representations to correct representation
        curr_rep = [{key: [None if key == 'reward' else l for _ in val] 
                     for key, val in r.items()}
                    for l, r in zip(curr_loc, curr_rep)]
        # Choose random action 
        curr_action = self.get_sample(self.loc_policy(
            curr_loc[1], self.env['dict']['locations'][curr_loc[0]]['env']['locations']))
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
            # In case of lower-level replay, we need both high and low level location
            curr_loc = [prev_step['location'][prev_step['level']]] \
                if prev_step['level'] == 0 else prev_step['location']
            curr_rep = [prev_step['representation'][prev_step['level']]]
        else:
            # Transition location and representation
            curr_loc, curr_rep = self.replay_transition(prev_step)
        # Add representation to memory
        self.rep_encode(curr_loc, curr_rep)
        # Select action for current location
        curr_action = self.get_sample(self.loc_policy(
            curr_loc[prev_step['level']], 
            self.env['dict']['locations'] if prev_step['level'] == 0 else
            self.env['dict']['locations'][curr_loc[0]]['env']['locations']))
        # Set current step 
        curr_step = {'i': prev_step['i'], 'k': prev_step['k'] + 1, 'location': curr_loc,
                     'representation': curr_rep, 'action': curr_action, 
                     'level': prev_step['level']}
        # And return all information about this replay step
        return curr_step
                
    def replay_transition(self, prev_step):
        # Like explore transition, except no observation - get location from base
        prev_action = prev_step['action']
        prev_rep = prev_step['representation']
        prev_loc = prev_step['location']
        # Get env dictionary depending on whether this is high-level or low-level replay
        env_dict = self.env['dict'] if prev_step['level'] == 0 \
            else self.env['dict']['locations'][prev_loc[0]]['env']
        # Get transition confusion matrix, again depending on replay level
        conf_mat = self.pars['transition_confusion'][prev_step['level']] \
            if prev_step['level'] == 0 \
            else self.pars['transition_confusion'][prev_step['level']][prev_loc[0]]
        # Transition representation
        curr_rep = self.replay_update(prev_rep[0], prev_action, env_dict, conf_mat)
        # Update location from base representation
        curr_loc = curr_rep[0]['base'] if prev_step['level'] == 0 \
            else [prev_loc[0], curr_rep[0]['base'][0]]
        # Return both
        return curr_loc, curr_rep       
    
    def replay_update(self, prev_rep, prev_action, env_dict, conf_mat):
        # Update each representation in replay by simply sampling a noisy transition
        # This is a separate transition for the high or the low level
        curr_rep = {}
        for name, rep in prev_rep.items():
            # First sample transition, then sample noise on top of that transition
            curr_rep[name] = [None if r is None else
                self.get_sample(conf_mat[name][
                    self.get_sample(env_dict['locations'][r]['actions'][prev_action]
                                    ['components']['base']['transition'])])
                for r in rep]
        # Return representation dictionary in list for consistency
        return [curr_rep]
    
    def replay_weight_update(self, curr_loc, curr_rep, prev_rep, prev_action):
        # TD-learn: if the animal thinks it's home, set reward
        reward = 1*(curr_loc in self.env['reward_locs'])
        # Build state from rep
        # curr_state = np.array(self.rep_to_state(curr_rep))
        # prev_state = np.array(self.rep_to_state(prev_rep))
        curr_state = np.eye(self.env['obj'].n_locations)[curr_loc]
        prev_state = np.eye(self.env['obj'].n_locations)[prev_rep['base'][0]]
        # Calculate TD delta: r + gamma * max_a Q(curr_rep, a) - Q(prev_rep, prev_action)
        delta = reward + self.pars['gamma'] * \
            np.max(np.matmul(curr_state, self.pars['replay_q_weights'])) \
            - np.matmul(self.pars['replay_q_weights'][:,prev_action], prev_state)
        # Return Q weight update using TD rule: w = w + alpha * delta * dQ/dw
        return self.pars['replay_q_weights'][:, prev_action] \
            + self.pars['alpha'] * delta * prev_state        
    
    def exp_evaluate(self, curr_step):
        # Increase number of visits to current location
        self.env['obj'].locations[curr_step['location']]['visits'] += 1
        # Add visit information to current step
        curr_step['visits'] = self.env['obj'].locations[curr_step['location']]['visits']
        # Find if best action probability has >0 optimum probability
        p = [action['probability'] for action 
             in self.pol[curr_step['location']]['actions']]
        curr_step['pol_corr'] = self.pars['optimal_pol'][curr_step['location']][
            'actions'][p.index(max(p))]['probability'] > 0
        # Return updated step
        return curr_step
        
    def replay_evaluate(self, curr_step):
        # Increase number of visits to current location
        self.env['obj'].locations[curr_step['location']]['visits'] += 1
        # Add visit information to current step
        curr_step['visits'] = self.env['obj'].locations[curr_step['location']]['visits']
        # Find if max action Q-val has >0 optimum probability
        q = [action['Q'] for action 
             in self.env['obj'].locations[curr_step['location']]['actions']]
        curr_step['q_corr'] = self.pars['optimal_pol'][curr_step['location']][
            'actions'][q.index(max(q))]['probability'] > 0       
        # Return updated step
        return curr_step        
    
    def loc_transition(self, prev_loc, prev_action, comp=None, noise=None):
        # Transition probabilities usually don't depend on component, unless 
        # e.g. you want to use base component to ignore objects
        # Get low-level and high-level previous location dictionary
        loc_high = self.env['dict']['locations'][prev_loc[0]]
        loc_low = loc_high['env']['locations'][prev_loc[1]]
        # Copy high-level location from previous step
        curr_high = prev_loc[0]
        # Get new location from environment transition of previous action
        curr_low = self.get_sample(
            loc_low['actions'][prev_action]['transition'] if comp is None
            else loc_low['actions'][prev_action]['components'][comp]['transition'])
        # Transition to same location while at a door: high-level room transition
        if curr_low == prev_loc[1]:
            # Find high level action
            a_high = self.get_high_action(loc_low)
            # If this is actually at a door: do transition
            if len(a_high) > 0:
                # There should only be one value in a_high: can be at one door at a time
                a_high = a_high[0]
                # If this action is allowed (not next to blockade): transition
                if loc_high['actions'][a_high]['probability'] > 0:                        
                    # Sample high-level location transition
                    curr_high = self.get_sample(loc_high['actions'][a_high]['transition'])
                    # And - only true for loop, but ok for now - move low loc to other door
                    # In future better to define where a door takes you in environment
                    curr_low =  self.env['dict']['locations'][curr_high]['env']\
                        ['components']['door' + str(1 - a_high)]['locations'][0]        
        # Add noise to low-level loc and sample if noise was provided as input
        # In future better to have noise on both levels, but may get weird
        curr_low = curr_low if noise is None else \
            self.get_sample(self.pars['transition_confusion'][1][noise][curr_low])
        # And return full transitioned location at high and low level
        return [curr_high, curr_low]
            
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
    
    def rep_encode(self, curr_loc, curr_rep, prev_loc=None):
        # If there's only a single level representation provided:
        if len(curr_rep) < 2:
            # Only update memory for provided level
            if len(curr_loc) < 2:
                # Only high-level loc provided, so set high-level memory
                self.pars['M_high'][curr_loc[0]].append(curr_rep[0])
            else:
                # Both levels location provided, so set low-level memory
                self.pars['M_low'][curr_loc[0]][curr_loc[1]].append(curr_rep[0])
        else:                
            # Add representation on both levels to memory
            self.pars['M_high'][curr_loc[0]].append(curr_rep[0])
            self.pars['M_low'][curr_loc[0]][curr_loc[1]].append(curr_rep[1])
        # When transitioning between high-level locations, make between level memory
        if prev_loc is not None and prev_loc[0] != curr_loc[0]:
            # Get low-level environment at previous location
            env_low = self.env['dict']['locations'][prev_loc[0]]['env']
            # Get action id of door at low-level previous location
            action = self.get_high_action(env_low['locations'][prev_loc[1]])[0]
            # Get door id: which door did we go through
            door = env_low['components']['door' + str(action)]['id']
            # Then append door id to action entry of between-level memory
            self.pars['M_between'][prev_loc[0]][action].append(door)
    
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
        representations = []
        for level, reps in enumerate(prev_rep):
            curr_rep = {}            
            for name, rep in reps.items():
                # For base represenation: use location that was used for memory retrieval
                if name == 'base':
                    curr_rep[name] = [curr_loc[level]]
                else:
                    curr_rep[name] = []
                    # Run through each location of this representation
                    for r_i in range(len(rep)):
                        # Get not-None probabilities
                        p_all = [p for p in [p_transition[level][name][r_i], 
                                             p_retrieve[level][name][r_i]]
                                 if p is not None]
                        # Weight probabibilities if both are not None
                        p_weighted = None if len(p_all) == 0 else \
                            p_all[0] if len(p_all) == 1 else \
                                [w * p_trans + (1 - w) * p_ret 
                                 for p_trans, p_ret in zip(*p_all)]
                        # Sample from combined probability
                        curr_rep[name].append(
                            None if p_weighted is None else self.get_sample(p_weighted))
            # Add representation dict for current level to representation list
            representations.append(curr_rep)
        return representations    
    
    def rep_prob_transition(self, prev_rep, prev_action):
        # Transition all representations that have been observed
        probabilities = []
        # Start with empty dictionary for probabilities at each level
        curr_prob = {}
        # First do high-level transition: use high-level rep, and low-level base
        # to transition from
        for name, rep in prev_rep[0].items():
            # Find whether this component has been observed
            curr_prob[name] = [
                None if r is None else
                self.pars['transition_confusion'][0][name][
                    self.loc_transition([r, prev_rep[1]['base'][0]], 
                                        prev_action, comp='base')[0]]
                for r in rep]
        # Append high-level dict to probabilities
        probabilities.append(curr_prob)
        # Then low-level transition: use low-level rep, and high-level base
        # to transition from
        curr_prob = {}        
        for name, rep in prev_rep[1].items():
            curr_prob[name] = []
            for r in rep:
                if r is None:
                    # If there was no previous representation: don't path integrate
                    curr_prob[name].append(None)
                else:
                    # Transition low level rep using high-level base
                    curr_rep = self.loc_transition([prev_rep[0]['base'][0], r], 
                                                   prev_action, comp='base')
                    # If this step moved to a new room: don't path integrate
                    curr_prob[name].append(None if curr_rep[0] != prev_rep[0]['base'][0]
                                           else self.pars['transition_confusion'][1]\
                                               [prev_rep[0]['base'][0]][name][curr_rep[1]])
        # Append low-level dict to probabilities
        probabilities.append(curr_prob)            
        return probabilities        
    
    def rep_prob_retrieve(self, curr_loc):
        # Get probability distributions over all representations at memory indexed
        # by current location. Probability ~ # of occurences of representation in memory
        # Start with empty (all-zero) probability distribution
        probabilities = [{
            name: [[0 for _ in range(locs)] for _ in rep]
            for name, rep in curr_rep.items()}
            for locs, curr_rep in zip(
                    [self.env['dict']['n_locations'],
                     self.env['dict']['locations'][curr_loc[0]]['env']['n_locations']],
                    self.rep_empty())]
        # Add representations in each memory to probabilities
        for prob, mems in zip(probabilities, 
                              [self.pars['M_high'][curr_loc[0]], 
                               self.pars['M_low'][curr_loc[0]][curr_loc[1]]]):
            for mem in mems:
                # Go through the different representations of this memory
                for name, rep in mem.items():
                    # Add memory to to probability
                    for r_i, r in enumerate(rep):
                        if r is not None:
                            prob[name][r_i][r] += 1
        # Calculate representation probabilities from counts - if there are any
        probabilities = [{name: [[p / sum(prob) for p in prob] if sum(prob) > 0 else None
                            for prob in rep]
                     for name, rep in curr_prob.items()} for curr_prob in probabilities]
        return probabilities
    
    def rep_empty(self):
        # Create empty representation, where all represented locations are set to None
        # Start at high level: find for each component how many locations it has
        rep_high = {}
        for comp in self.env['comp'][0]:
            # Add component if it wasn't in the representation already
            if comp in self.env['dict']['components'] and comp not in rep_high.keys():
                # Add None for each location of this component, if it has locations
                if 'locations' in self.env['dict']['components'][comp].keys():                        
                    rep_high[comp] = [
                        None for _ in self.env['dict']['components'][comp]['locations']]
                else:
                    rep_high[comp] = [None]                
        # Then to do the same for lower level
        rep_low = {}
        for comp in self.env['comp'][1]:
            for loc in self.env['dict']['locations']:
                if comp in loc['env']['components'] and comp not in rep_low.keys():
                    if 'locations' in loc['env']['components'][comp].keys():                        
                        rep_low[comp] = [
                            None for _ in loc['env']['components'][comp]['locations']]
                    else:
                        rep_low[comp] = [None]
        # Return list of high and low level representations
        return [rep_high, rep_low]
    
    def rep_observe(self, curr_loc):
        # Get observed representation-location for given true-location
        # Start with empty representation
        reps = self.rep_empty()
        # Get components for current high-level and low-level environments
        comps = [
            self.env['dict']['components'], 
            self.env['dict']['locations'][curr_loc[0]]['env']['components']]
        # Representations at all levels can only be observed at the lowest level
        for loc, comp, rep in zip(curr_loc, comps, reps):
            for key, val in comp.items():
                # Base representation: simply current location
                if key == 'base': 
                    rep[key] = [loc for _ in rep[key]]                                    
                # Reward representation: observed when low-level at reward
                if key == 'reward' and key in comps[1].keys():
                    if curr_loc[1] in comps[1]['reward']['locations']:
                        rep[key] = [loc for _ in rep[key]]
                # Blockade representation: observed when low-level at blocked door
                if key == 'blockade' and curr_loc[0] in val['locations']:
                    # Find which action is blocked
                    blocked = [a['id'] 
                               for a in self.env['dict']['locations'][loc]['actions']
                               if a['probability'] == 0][0]
                    # If we are at blocked door: observe current blockade
                    if curr_loc[1] in comps[1]['door' + str(blocked)]['locations']:
                        rep[key] = [loc for _ in rep[key]]
                # Door representation: observed by id when at door
                if 'door' in key:
                    if loc in val['locations']: 
                        rep['door' + str(val['id'])] = [
                            loc for _ in rep['door' + str(val['id'])]]
        # Return observed representation
        return reps

    def rep_fix(self, curr_rep, observed_rep):
        # Update representation from observed representation, where available
        representations = []
        for c_rep, o_rep in zip(curr_rep, observed_rep):
            fixed_rep = {}
            for name, rep in c_rep.items():
                # Copy over observed representation, if there is one
                fixed_rep[name] = [r_c if r_o is None else r_o
                                   for r_o, r_c in zip(o_rep[name], rep)]
            representations.append(fixed_rep)
        return representations
                
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
    
    def rep_to_rep_pol(self, curr_rep):
        # There is a difference between *observed* and *policy* representations of doors
        # A door is *observed* by its ID (e.g. 0), and from then on its representation 
        # is carried around under the corresponding name (e.g. door0).
        # On the other hand, for the *policy* each door is named by its high-level action
        # (e.g. the door for action 1 is called door 1)
        # The correspondence between the two is revealed when following a high-level action
        # and stored in the between-level memory: M[high-level room][*policy*] = *observed*
        pol_rep = {}
        # Find all memories for current high-level location, for all doors
        for name, rep in curr_rep[1].items():
            if 'door' in name:
                # Get action number for this door
                door_pol = int(name[-1])
                # Get memories for this action
                door_memory = self.pars['M_between'][curr_rep[0]['base'][0]][door_pol]
                if len(door_memory) > 0:
                    # If there are memories available: just choose any of them
                    pol_rep['door' + str(door_pol)] = curr_rep[1][
                        'door' + str(random.choice(door_memory))]
                else:
                    # If there are no memories: set to None
                    pol_rep['door' + str(door_pol)] = [
                        None for _ in curr_rep[1]['door' + str(door_pol)]]
            else:
                # For all other representations: just copy
                pol_rep[name] = rep
        # Return the original high-level representation, and re-ordered low-level rep
        return [curr_rep[0], pol_rep]        
    
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
        # Set default transition noise for all components on low level
        pars_dict['transition_noise'] = [{key: 0 for key in comp} 
                                         for comp in self.env['comp']]
        # Set confusion matrix: which locations can be confused?
        pars_dict['transition_confusion'] = \
            self.get_all_confusion(pars_dict['transition_noise'])
        pars_dict['path_int_weight'] = 0.2
        pars_dict['explore_max_steps'] = 100
        pars_dict['test_max_steps'] = 20        
        pars_dict['replay_n'] = 20
        pars_dict['replay_max_steps'] = 15
        pars_dict['replay_q_weights'] = None 
        pars_dict['print'] = True
        pars_dict['M_high'] = [[] for _ in range(self.env['dict']['n_locations'])]
        pars_dict['M_between'] = [[[] for _ in range(self.env['dict']['n_actions'])]
                                  for _ in range(self.env['dict']['n_locations'])]
        pars_dict['M_low'] = [[[] for _ in range(loc['env']['n_locations'])]
                              for loc in self.env['dict']['locations']]
        pars_dict['rep_control'] = False
        pars_dict['experiment'] = 2
        pars_dict['alpha'] = 0.8
        pars_dict['gamma'] = 0.7
        # Return default dictionar
        return pars_dict        
                
    def init_env(self, env):
        # Create dictionary of environment variables
        env_dict = {}
        # Get environment object
        env_dict['obj'] = env
        # And directly assign environment dict for easy access
        env_dict['dict'] = env.env
        # Get where rewards are in current environment: any shiny location
        env_dict['reward_locs'] = self.get_reward_locs(env_dict['dict'])
        # Get distance between all locations in base policy
        env_dict['dist'] = self.get_loc_dists(env_dict['obj'])
        # Get all components in the current environment - defines order in representation
        env_dict['comp'] = env_dict['obj'].get_components()
        # Start with zero visits to each location
        for loc_high in env_dict['dict']['locations']:
            loc_high['visits'] = 0
            for loc_low in loc_high['env']['locations']:
                loc_low['visits'] = 0
        # Get maximum number of actions in each direction - defines representation size
        env_dict['max_actions'] = self.get_max_actions(env_dict['dict'])
        # Return environment dictionary
        return env_dict
    
    def init_representations(self, env_dict):
        # Get representation at each location for root env
        reps = [self.get_representations(env_dict)]
        # Then append representation for all locations for low-level envs
        reps.append([self.get_representations(location['env'])
            for location in env_dict['locations']])
        # And return
        return reps    
    
    def init_policy(self, env_obj, reps):
        # Return full optimal policy
        return self.get_full_policy_optimal()
    
    def update_pol(self, env, pol):
        # Update base policy
        env.set_policy(pol)
        return env
    
    def update_rep(self, env_dict, rep):
        # Get representation components and sizes FROM SELF: for consistency across envs
        comps = self.env['comp']
        sizes = self.env['max_actions']
        # Rep contains high and low-level reps
        rep_high, rep_low = rep  
        # Add representations to locations in environment
        for l_i, loc_high in enumerate(env_dict['locations']):
            # Add separate component representations
            for comp in rep_high.keys():
                loc_high['components'][comp]['representation'] = \
                    rep_high[comp][l_i]
            # Get concatenated representation: all components after each other
            rep_full = []
            # Add zeros for component if it doesn't exist in this particular env
            for comp in comps[0]:
                rep_full.append(flatten([r for r in rep_high[comp][l_i]] 
                                        if comp in rep_high.keys()
                                        else [[0] + [0 for _ in range(s)] 
                                              for s in sizes[0]]))
            # Finally, build full representation by concatenating all components
            loc_high['representation'] = flatten(rep_full)
            # Then do the exact same thing but at the low-level env at this loc
            for l_j, loc_low in enumerate(loc_high['env']['locations']):
                # Get current rep
                curr_rep = rep_low[l_i]
                # Add separate component representations
                for comp in curr_rep.keys():
                    loc_low['components'][comp]['representation'] = \
                        curr_rep[comp][l_j]
                # Get concatenated representation: all components after each other
                rep_full = []
                # Add zeros for component if it doesn't exist in this particular env
                # Used to have bug here! Was l_i instead of l_j - gives same rep everywhere
                for comp in comps[1]:
                    rep_full.append(flatten([r for r in curr_rep[comp][l_j]] 
                                            if comp in curr_rep.keys()
                                            else [[0] + [0 for _ in range(s)] 
                                                  for s in sizes[1]]))
                # Finally, build full representation by concatenating all components                        
                loc_low['representation'] = flatten(rep_full)
        return env_dict
        
    def get_representations(self, env_dict):
        # Create representation at every location for each component
        rep_dict = {}
        # Run through environment components
        for name, component in env_dict['components'].items():
            if component['type'] == 'shiny':
                rep_dict[name] = self.get_representation_shiny(env_dict, name)
            elif component['type'] == 'block':
                rep_dict[name] = self.get_representation_block(env_dict, name)                                
            elif component['type'] == 'base':
                rep_dict[name] = self.get_representation_base(env_dict, name)                
            else:
                print('Type ' + component['type'] + 
                      ' of component ' + name + 'not recognised.')
            # Print progress, because representations take time to compute
            print('Finished representation ' + name)
        return rep_dict    
    
    def get_representation_shiny(self, env_dict, name):
        # Create representation towards shiny object
        return self.representations_to_locations(
            env_dict, env_dict['components'][name]['locations'])
    
    def get_representation_block(self, env_dict, name):
        # Create representation towards start of block
        block_start = self.representations_to_locations(
            env_dict, [env_dict['components'][name]['locations'][0]])
        # Create representation towards end of block
        block_end = self.representations_to_locations(
            env_dict, [env_dict['components'][name]['locations'][1]])
        # Combine representations into wall representation
        return [start + end for start, end in zip(block_start, block_end)]    
    
    def get_representation_base(self, env_dict, name):
        # Save some time by not having a base representation
        return [[[] for _ in range(env_dict['n_actions'])] 
                for _ in range(env_dict['n_locations'])]
    
    def representations_to_locations(self, env_dict, locs, env_obj=None):
        # Find representation for every location in env
        # Get environment object because we'll need some policy building functionality
        env_obj = self.env['obj'] if env_obj is None else env_obj
        # Get number of neurons for this component, given by longest sequence of each action
        # Then distance along each action can be expressed by one-hot vector for that action
        # Add one additional neuron (the first) for when action is not on path        
        neurons = [n + 1 for n in self.env['max_actions'][env_dict['level']]]
        # Initialise empty representations for each location
        representations = [[[0 for _ in range(n)] for n in neurons] 
                            for _ in range(env_dict['n_locations'])]
        # Then calculate representations if provided locations are not None
        if not any([l is None for l in locs]):
            # Get base policy, which is going to be used for action distances
            policy = env_obj.get_policy(env=env_dict, name='base')
            # Get distance from each location to locs according to base policy
            dist = env_obj.get_distance(env_obj.get_adjacency(policy))[:, locs]
            # Now find representation for each location: distance to object along each action
            for l_i, location in enumerate(env_dict['locations']):
                # Then for each action: repeat until distance increases
                for a_i, a_r in enumerate(representations[l_i]):
                    # Copy location as start for transitioning
                    loc = location
                    # Create list of all distances along chain of current action
                    d = [int(dist[l_i])]
                    # Then repeat action as often as possible, and keep track of distance
                    for i in range(neurons[a_i] - 1):
                        # Transition along current action (changing iterator OK in Python)
                        loc = policy[np.argmax(loc['actions'][a_i]['transition'])]
                        # Add distance to distance array
                        d.append(int(dist[loc['id']]))
                    # Representation is given by location of minimum distance along action
                    a_r[d.index(min(d))] = 1
        return representations
    
    def max_actions(self, env_dict):
        # Find longest possible sequence of single action
        longest_action_sequence = []        
        for action in range(env_dict['n_actions']):
            # Find how far you can get for each location
            curr_longest = 0
            # Location iterator var is changed in for loop - ok in Python, not in C
            for location in env_dict['locations']:
                # Start at 1: not taking the action also counts
                curr_walk = [location['id']]
                # Get action from base component
                curr_action = location['actions'][action]['components']['base']
                # Keep trying to take current action until action not available
                while curr_action['probability'] > 0 \
                    and sum(curr_action['transition']) > 0:
                    location = env_dict['locations'][
                        np.argmax(curr_action['transition'])]
                    curr_action = location['actions'][action]['components']['base']
                    if location['id'] in curr_walk:
                        # If at location where you have been before (e.g. in loops): stop
                        break
                    else:
                        # In other cases: add current location to walk
                        curr_walk.append(location['id'])
                # After action not being available: update longest path
                curr_longest = max(curr_longest, len(curr_walk))
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

    def get_all_confusion(self, noise):
        # Get confusion between all locations in base policy for root env
        conf = [{key: self.get_mat_confusion(
            noise[0][key], self.env['obj'].get_policy(env=self.env['dict'], name='base'))
            for key in self.env['dict']['components'].keys()}]
        # Then append confusion between all locations for low-level envs
        conf.append([{key: self.get_mat_confusion(
            noise[1][key], self.env['obj'].get_policy(env=location['env'], name='base'))
            for key in location['env']['components'].keys()}
            for location in self.env['dict']['locations']])
        # And return
        return conf 
    
    def get_mat_confusion(self, noise, policy):
        # Create confusion matrix between locations, where each location can be
        # confused for its neighbours with a probability given by noise
        # Confusion matrix: C_ij is probability that REAL i gets REPLACED BY j
        # Get distance between locations from adjacency from policy
        dist = self.env['obj'].get_distance(self.env['obj'].get_adjacency(policy))
        # Set confusion to {1-noise: correct, noise: neighbour}
        loc_confusion = noise * (dist == 1) \
            / np.reshape(np.sum(dist==1,1),[dist.shape[0],1]) \
                + (1 - noise) * np.eye(dist.shape[0])
        return loc_confusion    
    
    def get_loc_dists(self, env=None):
        # Find distances between locations in base policy on all levels
        # Use own environment if environment not provided
        env = self.env['obj'] if env is None else env
        # Get distance between all locations in base policy for root env
        dists = [env.get_distance(
            env.get_adjacency(
                env.get_policy(env=env.env, name='base')))]
        # Then append distance between all locations for low-level envs
        dists.append([env.get_distance(
            env.get_adjacency(
                env.get_policy(env=location['env'], name='base')))
            for location in env.env['locations']])
        # And return
        return dists
    
    def get_max_actions(self, env_dict=None):
        # Use own environment if environment not provided
        env_dict = self.env['dict'] if env_dict is None else env_dict
        # Get max actions (which gives representation size) on high level
        max_actions = [self.max_actions(env_dict)]
        # And append actions on low level
        max_actions.append(list(np.max(np.stack([
            self.max_actions(loc['env']) for loc in env_dict['locations']]), axis=0)))
        # Return max actions
        return max_actions

    def get_reward_locs(self, env_dict=None):
        # Find all rewarded locations in environment from shiny components
        # Use own environment if environment not provided
        env_dict = self.env['dict'] if env_dict is None else env_dict
        # Find reward location on each level
        reward_locs = env_dict['components']['reward']['locations']
        reward_locs = [
            [r, env_dict['locations'][r]['env']['components']['reward']['locations']]
            for r in reward_locs]
        return reward_locs        
        
    def get_high_action(self, loc_low):
        # Find high-level action by getting door you have at low-level location
        return [int(key[-1]) for key, val in loc_low['components'].items()
                  if 'door' in key and val['in_comp']]
    
    def get_low_policy_optimal(self, loc_high, env=None, pol_high=None):
        # Use default env if env not supplied
        env_obj = self.env['obj'] if env is None else env
        # Get high-level policy at high-level loc, if not provided
        pol_high = self.get_high_policy_optimal(env=env_obj.env)[loc_high['id']] \
            if pol_high is None else pol_high
        # Get dict for low-level environment at high-level location
        env_dict = loc_high['env']
        # Get reward location on all levels
        reward_locs = self.get_reward_locs(env_obj.env)
        r_high = [r[0] for r in reward_locs]
        # Get low level policy depending on high-level location
        if loc_high['id'] in r_high:
            # If current high-level location is at reward: get low-level reward policy
            opt_pol = env_obj.policy_optimal(env_obj.get_policy(
                env=env_dict, name='reward', in_place=False))
        else:
            # Get optimal high-level actions
            opt_high = [i for i, a in enumerate(pol_high['actions']) 
                        if a['probability'] > 0]
            # Get locations for these high-level actions
            loc_low = flatten([env_dict['components']['door' + str(i)]['locations']
                               for i in opt_high])
            # Calculate optimal policy to these locations
            opt_pol = env_obj.policy_optimal(env_obj.policy_distance(env_obj.get_policy(
                env=env_dict, name='base',in_place=False), loc_low, env=env_dict))
        return opt_pol    
    
    def get_high_policy_optimal(self, env=None):
        # Use default env if env not supplied
        env_obj = self.env['obj'] if env is None else env
        # And use high-level environment for that object
        env_dict = env_obj.env
        # Get reward location on high level levels
        r_high = [r[0] for r in self.get_reward_locs(env_obj.env)]
        # Get default environment policy
        env_pol = env_obj.get_policy(env=env_dict, in_place=False)
        # Calculate optimal policy to high-level location
        opt_pol = env_obj.policy_optimal(env_obj.policy_distance(
            env_pol, r_high, env=env_dict,
            adjacency=env_obj.get_adjacency(env_pol)))
        return opt_pol       
    
    def get_full_policy_optimal(self, env=None):
        # Set env object if not provided
        env_obj = self.env['obj'] if env is None else env
        # Get high-level optimal policy
        pol_high = self.get_high_policy_optimal(env=env_obj)
        # Add low-level policy to each location
        for l_i, loc in enumerate(env_obj.env['locations']):
            # Get low-level policy
            pol_low = self.get_low_policy_optimal(loc, env=env_obj, pol_high=pol_high[l_i])
            # And assign to high-level policy location
            pol_high[l_i]['env'] = {'locations': pol_low}        
        return pol_high
    
    def get_sample(self, probability):
        # Sample from array containing probability distribution
        return np.random.choice(len(probability), p=probability)    
    
class FitPolicy(Model):
    
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
        pars_dict['DQN']['episodes'] = 10
        # Number of training steps per episode
        pars_dict['DQN']['steps'] = 500
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
        pars_dict['DQN']['load'] = './trained/20211221_141051.pt'
        return pars_dict
    
    # Use learned 
    def exp_evaluate(self, curr_step):
        # Increase number of visits to current location
        self.env['obj'].locations[curr_step['location']]['visits'] += 1
        # Add visit information to current step
        curr_step['visits'] = self.env['obj'].locations[curr_step['location']]['visits']
        
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
            self.env['obj'].locations[curr_step['location']], q, 
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
            self.env['obj'].locations[curr_step['location']], q, 
            self.pars['DQN']['beta'])
        # Find if most likely action is optimal action (>0 prob in optimal pol)
        curr_step['full_rep_corr'] = self.pars['optimal_pol'][curr_step['location']][
            'actions'][p.index(max(p))]['probability'] > 0          

        # Return updated step
        return curr_step
        
    def replay_evaluate(self, curr_step):
        # Increase number of visits to current location
        self.env['obj'].locations[curr_step['location']]['visits'] += 1
        # Add visit information to current step
        curr_step['visits'] = self.env['obj'].locations[curr_step['location']]['visits']     
        
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
            self.env['obj'].locations[curr_step['location']], q, 
            self.pars['DQN']['beta'])
        # Find if most likely action is optimal action (>0 prob in optimal pol)
        curr_step['replay_rep_corr'] = self.pars['optimal_pol'][curr_step['location']][
            'actions'][p.index(max(p))]['probability'] > 0          
        
        # Find location policy for current representation
        q = np.matmul(np.eye(self.env['obj'].n_locations)[curr_step['location']],
                      self.pars['replay_q_weights'])
        p = self.learn_get_policy_location(
            self.env['obj'].locations[curr_step['location']], q, 
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
                # Initialise network
                policy_net = DQN(d_in, d_out)
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
        policy_net = DQN(d_in, d_out)
        
        # Initialise optimiser
        optimiser = torch.optim.Adam(policy_net.parameters(), lr = 1e-3)
        # Create a tensor board to stay updated on training progress. 
        # Start tensorboard with tensorboard --logdir=runs
        writer = SummaryWriter(self.pars['DQN']['save'].replace('.pt','_tb'))                
                
        # Run training loop
        for i_episode in range(self.pars['DQN']['episodes']):
            # Get all variables to start episode
            envs, location, state = self.learn_init_episode(
                self.env['obj'], self.pars['DQN']['envs'])
            # Now run around through environments
            for t in range(self.pars['DQN']['steps']):
                # Calculate current number of training steps
                steps = i_episode * self.pars['DQN']['steps'] + t
                
                # Do transition: choose action, find new state, add to memory
                location, state = self.learn_do_transition(
                    state, steps, policy_net, envs, location)                                
                
                # Get action probabilities and learned values for new locatoin
                next_policy = torch.tensor(
                    [[action['probability'] 
                      for action in env.pol[
                              loc[0]['id']]['env']['locations'][loc[1]['id']]['actions']]
                     for loc, env in zip(location, envs)], dtype=torch.float)
                next_values = policy_net(state)                    
        
                # Learn action values from correct policy 
                self.learn_optimise_step(policy_net, optimiser, next_policy, next_values)                
                    
                # Display progress
                if t % 100 == 0:
                    curr_optimality = np.array(
                        self.learn_evaluate_performance(policy_net, envs))
                    print('-- Step', t, '/', self.pars['DQN']['steps'],
                          'optimilaty', np.mean(curr_optimality))
                    writer.add_scalar('Optimality', np.mean(curr_optimality), steps)
                    # import plot; import matplotlib.pyplot as plt
                    # learned = self.learn_get_policy_learned(policy_net, envs[0])
                    # opt = envs[0].pol
                    # plt.figure();
                    # ax = plt.subplot(1,2,1)
                    # plot.plot_env({'locations': opt}, ax=ax)
                    # ax.set_title('optimal')
                    # ax = plt.subplot(1,2,2)
                    # plot.plot_env({'locations': learned}, ax=ax)
                    # ax.set_title('learned')
            # Save policy net at the end of each episode
            if self.pars['DQN']['save'] is not None:
                torch.save(policy_net.state_dict(), self.pars['DQN']['save'])
            # Display progress
            print('- Finished episode', i_episode, '/', self.pars['DQN']['episodes'])
        # Return trained policy network
        return policy_net    
    
    def learn_init_network(self):
        # Find dimensions of input and output of network
        # Input: very lazy but does the trick, just first location reps
        d_in = len(self.env['dict']['locations'][0]['representation']) + \
            len(self.env['dict']['locations'][0]['env']['locations'][0]['representation'])
        # Output: number of actions in environemnt
        d_out = self.env['dict']['locations'][0]['env']['n_actions']
        # Return both
        return d_in, d_out 
    
    def learn_init_episode(self, base_env, n_envs):
        # Create environments
        envs = self.learn_get_training_envs(base_env, n_envs)
        # Initialise locations: random location for each environment
        location = self.learn_get_location(envs)
        # And get state for these locations
        state = self.learn_get_state(location)
        # Return all variables for starting an episode
        return envs, location, state
        
    def learn_do_transition(self, state, steps, policy_net, envs, 
                            location):
        # Select and perform an action
        # location = self.learn_get_transition(envs, location)
        location = self.learn_get_location(envs)
        state = self.learn_get_state(location)
        # Return updated location, state, and memory          
        return location, state
    
    def learn_optimise_step(self, policy_net, optimiser, policy, values):
        # Compute cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(values, policy)
    
        # Optimize the model
        optimiser.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        optimiser.step()
        # Return model
        return optimiser, policy_net
    
    def learn_get_training_envs(self, real_env, n_envs):
        # Get training environment from real environment for each batch
        envs = []
        for i in range(n_envs):
            print('Creating training env', i)
            # Create new environment from base of real environment
            new_env = real_env.get_copy()
            # Now add blockade with 50% chance, to learn policy with and without
            if random.random() > 0.5:
                new_env.init_block(new_env.env, 'blockade')
            # Create representations for this environment
            new_reps = self.init_representations(new_env.env)
            # Add represenations to new environment
            new_env.env = self.update_rep(new_env.env, new_reps)
            # Add optimal policy to environment to evaluate network performance
            new_env.pol = self.get_full_policy_optimal(new_env)
            # Finally add environment to list of training environments
            envs.append(new_env)
        return envs
            
    def learn_get_location(self, envs):
        # Randomly select high-level and low-level location from each environment
        new_locs = []
        for env in envs:
            # Choose new high level location
            new_loc = [random.choice(env.env['locations'])]
            #new_loc = [env.env['locations'][0]]
            # Choose new low-level location within that high-level
            new_loc.append(random.choice(new_loc[0]['env']['locations']))
            #new_loc.append(new_loc[0]['env']['locations'][0])
            # And add pair of new locations to list of location
            new_locs.append(new_loc)
        return new_locs
    
    def learn_get_transition(self, envs, locs):
        # Move through list of locations instead of following action
        new_locs = []
        for loc, env in zip(locs, envs):
            if (loc[1]['id'] + 1) < loc[0]['env']['n_locations']:
                # Keep high-level location, increment low-level location
                new_loc = [loc[0]]
                new_loc.append(new_loc[0]['env']['locations'][loc[1]['id'] + 1])
            else:
                if (loc[0]['id'] + 1) < env.env['n_locations']:
                    # Increment high-level location, reset low-level location
                    new_loc = [env.env['locations'][loc[0]['id'] + 1]]
                    new_loc.append(new_loc[0]['env']['locations'][0])
                else:
                    # Reset high-level location, reset low-level location
                    new_loc = [env.env['locations'][0]]
                    new_loc.append(new_loc[0]['env']['locations'][0])
            # Append new location to output
            new_locs.append(new_loc)
        return new_locs       
    
    def learn_get_state(self, locs):
        # Build state tensor: representations of each location
        return torch.stack([torch.cat([torch.tensor(l['representation'], dtype=torch.float) 
                                       for l in loc])
                            for loc in locs])
        
    def learn_get_policy(self, locs, q_vals, env_obj, beta=1):
        # Get high-level base policy from base
        pol_high = env_obj.get_policy(name='base', in_place=False)
        # Create low-level policy from base
        for l_i, loc_high in enumerate(env_obj.env['locations']):
            pol_high[l_i]['env'] = {'locations': 
                                    env_obj.get_policy(env=loc_high['env'], name='base', 
                                                      in_place=False)}
        # Now update probabilities in low-level locations according to input
        for q, (loc_high, loc_low) in zip(q_vals, locs):
            # Get location to update
            curr_loc = pol_high[loc_high['id']]['env']['locations'][loc_low['id']]
            # Get probability for current location
            p = self.learn_get_policy_location(curr_loc, q, beta)
            # And assign probabilities to actions in source policy
            for action, probability in zip(curr_loc['actions'], p):
                action['probability'] = probability
        # Return updated source policy
        return pol_high
    
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
        # Create big list with all high-, low-level location pairs
        locs = []
        for high_loc in env.env['locations']:
            for low_loc in high_loc['env']['locations']:
                locs.append([high_loc, low_loc])
        # Do forward pass through policy net to get Q-values
        q_vals = policy_net(self.learn_get_state(locs)).detach().numpy()
        # Build policy from environment root policy and Q-values
        learned_pol = self.learn_get_policy(locs, q_vals, env,
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
                optimality.append(self.learn_get_optimality_old(learned_pol, env))
            # Switch policy net back to train mode
            policy_net.train()
        # Return optimality
        return optimality    
    
    def learn_evaluate_policy(self, learned_pol, env=None):
        # Use default env if env not supplied
        env = self.env['obj'] if env is None else env
        # Get optimality for learned policy
        return self.learn_get_optimality(learned_pol, env)    
    
    def learn_evaluate_reward(self, memory):
        # Find average reward earned per environment per transitions in memory
        return np.mean([np.mean(mem.detach().numpy()) for mem in memory.get_all(3)])
    
    def learn_get_optimality_old(self, learned_pol, env):
        # Get optimal policy for provided env
        opt_pol = self.get_full_policy_optimal(env)
        # Keep track if learned optimal action is optimal at each location
        is_opt = []
        # Run through high-level locations
        for loc_high_learned, loc_high_opt in zip(learned_pol, opt_pol):
            # Run through locations in learned and optimal policy
            for loc_learned, loc_opt in zip(loc_high_learned['env']['locations'], 
                                            loc_high_opt['env']['locations']):
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
        hidden_dim = 2 * int(np.max([in_dim,out_dim])) \
            if hidden_dim is None else hidden_dim        
        # Linear layer 1: from input to hidden 1
        self.w1 = torch.nn.Linear(in_dim, hidden_dim)
        # Linear layer 2: from hidden 1 to hidden 2
        #self.w2 = torch.nn.Linear(hidden_dim, hidden_dim)        
        # Linear layer 3: from hidden 2 to out
        self.w2 = torch.nn.Linear(hidden_dim, out_dim)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor N_batch x N_actions.
    def forward(self, x):
        # Pass through first layer
        x = self.w1(x)
        # Apply non-linearity
        x = torch.nn.functional.relu(x)
        # Pass through second layer
        x = self.w2(x)
        # Apply non-linearity
        #x = torch.nn.functional.relu(x)
        # Pass through third layer
        #x = self.w3(x)
        return x
    
    
def flatten(t):
    return [item for sublist in t for item in sublist]
    

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))     