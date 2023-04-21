#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:49:54 2021

@author: jbakermans
"""
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import random
from collections import namedtuple, deque
from datetime import datetime
from scipy.stats import multivariate_normal
import shutil
from shapely.geometry import LineString, Point

# Base model with standard functionality for composing policies
class Model():    

    def __init__(self, env, pars=None):
        # Initialise environment: set home, retrieve distances
        self.env = self.init_env(env)
        # Initialise parameters, copying over all fields and setting missing defaults
        self.pars = self.init_pars(pars)
        
        # Get representations for each location from components
        self.reps = self.init_representations(self.env['env'])
        
        # Build policy from representations
        self.pol = self.init_policy(self.env['env'], self.reps)
        
    def reset(self, env, pars=None):
        # Reset a model to restart fresh (e.g. with new params, or different env)
        # This simple calls __init__, but has a separate name for clarity
        self.__init__(env, pars)
        
    def simulate(self):
        # Full representation at reward location
        explore = [self.exp_final()]        
        # Copy final step of explore walk
        final_step = dict(explore[-1])
        # Main simulations are going to change what happens after exploration
        if self.pars['experiment'] == 1:
            # 1. Continue by walking, both with or without latent learning
            # Clear memory
            self.pars['memory'] = []
            # Set current representation to only observed representation
            final_step['representation'] = self.rep_observe(final_step['location'])
            # And continue walk
            test = self.sim_explore(explore=[final_step])
        elif self.pars['experiment'] == 2:
            # 2. Continue by replay, both for Bellman backups and memory updates
            test = self.sim_replay(final_step)
        elif self.pars['experiment'] == 3:
            # 3. Find way back to start, with path integration only
            self.pars['path_int_weight'] = 1
            self.pars['replay_interval'] = -1
            # Start exploration from reward
            explore = self.sim_explore(explore=[final_step])
            # Start escape from final explore step
            final_step = dict(explore[-1])
            # But set the action to zero, so escape starts at end of explore
            final_step['action'] = [0, 0]
            # Then go home from final step
            test = self.sim_explore(explore=[final_step], go_to_goal=True)
        elif self.pars['experiment'] == 4:
            # 4. Find way back to start, by learning q-weights in replay
            self.pars['path_int_weight'] = 1
            # Start exploration from reward
            explore = self.sim_explore(explore=[final_step])
            # Start escape from final explore step
            final_step = dict(explore[-1])
            # But set the action to zero, so escape starts at end of explore
            final_step['action'] = [0, 0]            
            # Then go home from final step
            test = self.sim_explore(explore=[final_step], go_to_goal=True)
        elif self.pars['experiment'] == 5:
            # 5. Find way back to start, by using replay to build error-correctin mem
            self.pars['path_int_weight'] = 0.5
            # Start exploration from reward
            explore = self.sim_explore(explore=[final_step])
            # Start escape from final explore step
            final_step = dict(explore[-1])
            # But set the action to zero, so escape starts at end of explore
            final_step['action'] = [0, 0]            
            # Then go home from final step
            test = self.sim_explore(explore=[final_step], go_to_goal=True)            
        # And return both
        return explore, test
    
    def sim_explore(self, explore=None, go_to_goal=False):
        # Start with empty list that will contain info about each explore step
        explore = [self.exp_step([])] if explore is None else explore
        # Exploration stage: explore environment until completing representation
        while len(explore) < self.pars['explore_max_steps']:
            # Run step
            curr_step = self.exp_step(explore, go_to_goal)
            # Evaluate step when crossing heatmap grid lines
            if self.do_evaluate(explore[-1]['location'], curr_step['location']):
                curr_step = self.exp_evaluate(curr_step)
            # Do replay
            curr_step['replay'] = self.sim_replay(dict(curr_step)) \
                if (not go_to_goal and (len(explore) % self.pars['replay_interval'] == 1 
                                        or self.pars['replay_interval'] == 1)) else []
            # Store all information about current step in list
            explore.append(curr_step)  
            # Display progress
            if self.pars['print'] and len(explore) % 25 == 1:
                print('Finished explore', len(explore), '/', 
                      self.pars['explore_max_steps'], 
                      'at', curr_step['location'], 
                      'reps at', curr_step['representation'])
            # Stop if arrived at goal (or when you think you're at goal, in exp 3)
            if self.pars['experiment'] == 3 or self.pars['experiment'] == 5:
                if go_to_goal and \
                    self.get_goal_dist(curr_step['representation']['reward'][0]) \
                        < self.pars['line_of_sight']:
                    return explore
            else:
                if go_to_goal and \
                    self.get_goal_dist(curr_step['location']) < self.pars['line_of_sight']:
                    return explore
        return explore    
    
    def sim_replay(self, start_step):
        # Initialise replay weights if not set yet
        self.pars['replay_q_weights'] = np.zeros(
            (self.reps['place']['n'], 
             self.pars['replay_dirs'])) \
            if self.pars['replay_q_weights'] is None else self.pars['replay_q_weights']
        # Start with empty list that will contain lists of replays
        replay = []
        # Set k-variable thate counts replay steps to 0 to indicate start of replay
        start_step['k'] = 0
        # Each replay consists of a list of replay steps, which are dicts with step info
        for j in range(self.pars['replay_n']):
            # Start replay at current representation
            curr_replay = [start_step]
            # Replay until reaching max steps
            while len(curr_replay) < self.pars['replay_max_steps']:
                # Take replay step
                curr_step = self.replay_step(curr_replay)
                # Evaluate step when crossing heatmap grid lines
                if self.do_evaluate(curr_replay[-1]['location'], curr_step['location']):
                    curr_step = self.replay_evaluate(curr_step)                
                # Add this replay step to current replay list
                curr_replay.append(curr_step)
            # Display progress
            if self.pars['print']:            
                print('Finished replay', j, '/', self.pars['replay_n'], 
                      'in', curr_step['k'], 'steps at', curr_step['location'])                
            # After finishing replay iteration, add this replay to list of replays
            replay.append(curr_replay[1:])
        # And return replay list
        return replay       
    
    def exp_step(self, explore, go_to_goal=False):
        # Find new location in this step
        if len(explore) == 0:
            # First step: start from home
            curr_loc, curr_rep = self.exp_init()
            prev_action = None;
        else:
            # All other steps: get new location and representation from transition
            curr_loc, curr_rep = self.exp_transition(explore[-1])
            prev_action = explore[-1]['action']
        # Add representation to memory
        self.rep_encode(curr_loc, curr_rep)            
        # Select action for current location
        if go_to_goal:
            if self.pars['experiment'] == 3:
                # Experiment 3: only path integration, so use representation
                curr_action = self.get_learned_action(
                    location=None, representation=curr_rep)
            elif self.pars['experiment'] == 4:
                # Experiment 4: use q-values from replay, so replay action
                curr_action = self.get_replay_action(curr_loc)
            elif self.pars['experiment'] == 5:
                # Experiment 5: use representation for policy
                curr_action = self.get_learned_action(
                    location=None, representation=curr_rep)   
        else:
            curr_action = self.get_action(prev_action)
        # Return summary for this step
        curr_step = {'i': len(explore), 'location': curr_loc,
                     'representation': curr_rep, 'action': curr_action}
        return curr_step       
    
    def exp_init(self):
        # Set initial location and representation for explore walk: any free location      
        curr_loc = self.get_free_location()
        # Get representation for initial location
        curr_rep = self.rep_observe(curr_loc)
        return curr_loc, curr_rep
    
    def exp_final(self):
        # Set final location and representation for explore walk: reward loc,
        # with all representations set
        curr_loc = random.choice(self.get_goal_locs(self.env['env']))
        # Get representation for this location
        curr_rep = self.rep_observe(curr_loc)
        # But update all representations to correct representation
        curr_rep = {key: [curr_loc for _ in val] for key, val in curr_rep.items()}
        # Choose random action 
        curr_action = self.get_action()
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
    
    def exp_evaluate(self, curr_step):
        # Not much to do in model that doesn't learn policy  
        # Return updated step
        return curr_step       
    
    def replay_step(self, curr_replay):
        # Find most recent replay step
        prev_step = curr_replay[-1]
        if prev_step['k'] == 0:
            # In the first step of replay
            if self.pars['replay_reverse']: 
                # During reverse replay: start from goal
                curr_loc = self.get_goal_locs(self.env['env'])[0]
                curr_rep = self.rep_observe(curr_loc)         
            else:
                # During forward replay: continue from last real step
                curr_loc, curr_rep = prev_step['location'], prev_step['representation']
            prev_action = None;            
        else:
            # Transition location and representation
            curr_loc, curr_rep = self.replay_transition(prev_step)
            prev_action = prev_step['action']     
        # Add representation to memory, but make sure the actual location is replay start
        self.rep_encode(curr_loc, curr_rep, curr_loc=curr_replay[0]['location'])
        # Select appropriate action
        if self.pars['replay_reverse']:  
            # In reverse replay: continue from previous action
            curr_action = self.get_action(prev_action)
        else:
            # In forward replay: follow q-weight policy
            qs = np.matmul(self.reps['place']['rep']([curr_loc]),
                           self.pars['replay_q_weights'])
            # Get direction by sampling from softmax over q-weights
            curr_dir = self.get_action_decoded(np.random.choice(
                np.arange(len(qs)), p=np.exp(self.pars['beta'] * qs) 
                / np.sum(np.exp(self.pars['beta'] * qs))))
            # Get step from learned replay weights
            curr_action = [curr_dir,
                           min(self.pars['max_step'], self.get_goal_dist(curr_loc))]
        # Set current step 
        curr_step = {'i': prev_step['i'], 'k': prev_step['k'] + 1, 'location': curr_loc,
                     'representation': curr_rep, 'action': curr_action}
        # And return all information about this replay step
        return curr_step
    
    def replay_transition(self, prev_step):
        # Like explore transition, except no observation - get location from base
        prev_loc = prev_step['location']
        prev_action = prev_step['action']
        prev_rep = prev_step['representation']
        # Transition new location: that'll be the place representation
        # Get new representation by averaging rep from path integration and memory
        # Might change this, but for now: only use transitioned representation during 
        # replay, ignore the representation probability retrieved from memory
        curr_loc = self.loc_transition(prev_loc, prev_action, noise=self.pars['noise'])
        curr_rep = self.rep_transition(curr_loc, prev_rep, prev_action, w=1)
        if self.pars['replay_reverse']:
            # Get index of direction of previous action - add pi for opposite action
            prev_action_index = self.get_action_encoded(prev_step['action'][0] + np.pi)
            # Update replay q-weights - reversing curr and prev loc for reverse Bellman backup
            self.pars['replay_q_weights'][:, prev_action_index] = self.replay_weight_update(
                prev_loc, curr_loc, prev_action_index)
        else:       
            # Get index of direction of previous action
            prev_action_index = self.get_action_encoded(prev_step['action'][0])
            # Update replay q-weights 
            self.pars['replay_q_weights'][:, prev_action_index] = self.replay_weight_update(
                curr_loc, prev_loc, prev_action_index)            
        # Return both
        return curr_loc, curr_rep
    
    def replay_weight_update(self, curr_loc, prev_loc, prev_action):
        # TD-learn: set reward if within reach
        reward = 1*(self.get_goal_dist(curr_loc) < self.pars['line_of_sight'])
        # Get current and previous state representation from place cells
        curr_state = self.reps['place']['rep']([curr_loc])
        prev_state = self.reps['place']['rep']([prev_loc])
        # Calculate TD delta: r + gamma * max_a Q(curr_rep, a) - Q(prev_rep, prev_action)
        delta = reward + self.pars['gamma'] * \
            np.max(np.matmul(curr_state, self.pars['replay_q_weights'])) \
            - np.matmul(self.pars['replay_q_weights'][:,prev_action], prev_state)
        # Return Q weight update using TD rule: w = w + alpha * delta * dQ/dw
        return self.pars['replay_q_weights'][:, prev_action] \
            + self.pars['alpha'] * delta * prev_state            
    
    def replay_evaluate(self, curr_step):
        # Find if following replay would get you to goal
        d = self.replay_get_location_path(curr_step['location'], self.env['env'])
        # Add d to current step
        curr_step['d_bellman'] = d
        # Return updated step
        return curr_step       
    
    def replay_get_location_path(self, loc, env):
        # Get distances to reward loc under optimal policy
        opt_path, opt_dist = self.get_optimal_dist(loc, env, replay=True)
        # Now do the same for learned policy
        learned_path = [loc]
        while self.get_goal_dist(learned_path[-1], env) > self.pars['line_of_sight'] \
            and len(learned_path) < self.pars['max_eval_steps']:
                # Get q-values learned in replay for place state at curren location
                qs = np.matmul(self.reps['place']['rep']([learned_path[-1]]),
                               self.pars['replay_q_weights'])
                # Get step from learned replay weights
                action = [
                    self.get_action_decoded(np.where(qs == max(qs))[0][0]),
                    min(self.pars['max_step'], self.get_goal_dist(learned_path[-1], env))]  
                # And update location
                learned_path.append(self.loc_transition(learned_path[-1], action, env=env))
        # If learned policy didn't reach goal: set distance to -1
        learned_dist = -1 if len(learned_path) == self.pars['max_eval_steps'] \
            else np.sum([self.get_distance(l1, l2) 
                         for l1, l2 in zip(learned_path[:-1], learned_path[1:])])
        # Calculate difference between optimal dist and learned dist, if both arrived
        return -1 if opt_dist == -1 \
            else (self.pars['max_eval_steps'] if learned_dist == -1 
                  else learned_dist - opt_dist)      
    
    def loc_transition(self, prev_loc, prev_action, noise=None, env=None):
        # An action is a direction - length pair, location is a (x, y) coordinate
        # Add noise to action
        prev_action = prev_action if noise is None else \
            [a - n/2 + random.uniform(0, n) for a, n in zip(prev_action, noise)]
        
        # Calculate x,y components of action 
        new_a = np.array([prev_action[1] * np.cos(prev_action[0]),
                          prev_action[1] * np.sin(prev_action[0])])

        # Get lines for all walls in environment
        walls = self.env['walls'] if env is None else env.get_border_lines()
        # For each new collision: get wall normal FROM location TO wall
        v_to_wall = [self.get_wall_normal(wall, prev_loc)
                     for wall in walls]
        # Create a line for collision detection with finite thickness wall        
        col_wall = [self.get_wall_border(wall, v)
                    for wall, v in zip(walls, v_to_wall)]
        
        # Get current collisions: all walls that prev_loc is in contact with
        prev_collision = [i for i, wall in enumerate(col_wall)
                          if Point(prev_loc).distance(wall)
                          < 1e-6]
        # If there are any collisions
        if len(prev_collision) > 0:        
            # Get normal vectors for each wall that prev_loc collides with
            v = [v_to_wall[i] for i in prev_collision]
            # Sort dot product to find walls with smallest angle = largest dot prod
            sort_order = np.flip(np.argsort([np.dot(curr_v, new_a) for curr_v in v]))
            # Remove component of action along vector towards first wall
            new_a = new_a - max([0, np.dot(new_a, v[sort_order[0]])]) * v[sort_order[0]]
            # If after that the second vector still has a positive dot product: stop              
            if len(v) > 1 and np.dot(v[sort_order[1]], new_a) > 0:
                new_a = np.zeros(new_a.shape)          
                
        # Now action takes current collisions into account. Get new location
        curr_loc = [c + a for c, a in zip(prev_loc, new_a)]        
        # Get line from previous to proposed location
        curr_line = LineString([prev_loc, curr_loc])            
        # But the action might induce new collisions. Find those,
        # ignoring wall that are already contacted and walls parallel to motion
        new_collision = [i for i, wall in enumerate(col_wall)
                         if curr_line.intersects(wall)
                         and i not in prev_collision
                         and not self.is_parallel(np.diff(curr_line.xy), np.diff(wall.xy))]
        # If there are any collisions: stop right where they occur
        if len(new_collision) > 0:
            # Find locations of intersection with wall thickness line
            new_coll_intersect = [curr_line.intersection(wall) 
                                  for wall in [col_wall[i] for i in new_collision]]
            # Find distance to intersection points
            new_coll_dist = [self.get_distance(prev_loc, [i.x, i.x])
                             for i in new_coll_intersect]
            # And set new location to nearest intersect
            curr_loc = new_coll_intersect[new_coll_dist.index(min(new_coll_dist))]
            curr_loc = [curr_loc.x, curr_loc.y]
        # Once the line from previous to current location isn't blocked: return
        return curr_loc
    

    def rep_update(self, curr_loc, prev_rep, prev_action):
        # Transition representation by combining path integrated and retrieved rep
        curr_rep = self.rep_transition(curr_loc, prev_rep, prev_action)
        # Fix any representations that can directly be observed from the enfironment
        if self.pars['experiment'] < 3:
            # Don't use observation in replay experiments
            curr_rep = self.rep_fix(curr_rep, self.rep_observe(curr_loc))
        # And return new representations
        return curr_rep    
                
    def rep_fix(self, curr_rep, observed_rep):
        # Update representation from observed representation, where available
        fixed_rep = {}
        for name, rep in curr_rep.items():
            # Copy over observed representation, if there is one
            fixed_rep[name] = [r_c if r_o is None else r_o
                               for r_o, r_c in zip(observed_rep[name], rep)]
        return fixed_rep
                    
    def rep_observe(self, curr_loc=None):
        # Get observed representation-location for given true-location
        curr_rep = {}
        for name, component in self.env['env'].components.items():
            curr_rep[name] = [
                curr_loc 
                if curr_loc is not None and \
                    self.get_distance(curr_loc, loc) < self.pars['line_of_sight']
                else None 
                for loc in component['locations']]
        # Return observed representation
        return curr_rep  
    
    def rep_transition(self, curr_loc, prev_rep, prev_action, w=None):
        # Use value from parameters if not specified
        w = self.pars['path_int_weight'] if w is None else w
        # Combine path integrated and retrieved representation
        curr_rep = self.rep_observe()
        # Retrieve representation from memory; if w = 1, only use path integration.
        rep_mem = self.rep_retrieve(curr_loc) if w < 1 else self.rep_observe()
        # Path integrate representation; if w = 0, only use memory retrieval
        rep_path_int = self.rep_path_integrate(prev_rep, prev_action) if w > 0 \
            else self.rep_observe()
        # Then take weighted average of rep from each
        for name, rep in curr_rep.items():
            # Run through each location of this representation
            for r_i in range(len(rep)):
                # Get not-None reps
                r_all = [r for r in [rep_path_int[name][r_i], rep_mem[name][r_i]]
                         if r is not None]
                # Average reps if both are not None
                r_avg = None if len(r_all) == 0 else \
                    r_all[0] if len(r_all) == 1 else \
                        [w * np.array(r_pt) + (1 - w) * np.array(r_m)
                         for r_pt, r_m in zip(*r_all)]
                # But the average might end up going through a wall! Prevent that
                if r_avg is not None:
                    # Get 'action' from current location (which is valid) to rep
                    a = np.array(r_avg) - np.array(curr_loc)
                    # Get direction and step size
                    d = np.arctan2(a[1], a[0])
                    l = np.sqrt(np.sum(a**2))
                    # Do step towards representation, stopping at walls
                    r_avg = self.loc_transition(curr_loc, [d, l])
                # Set average rep 
                curr_rep[name][r_i] = None if r_avg is None else [r for r in r_avg]
        return curr_rep
    
    def rep_path_integrate(self, prev_rep, prev_action):
        # Transition all representations that have been observed
        curr_rep = {}
        for name, rep in prev_rep.items():
            # Transition representation if it was observed before
            curr_rep[name] = [None if r is None else 
                              self.loc_transition(r, prev_action, noise=self.pars['noise'])
                              for r in rep]
        return curr_rep        
    
    def rep_retrieve(self, curr_loc):
        # Retrieve previously observed representations,
        # weighted by their similarity to the current location
        # Initialise empty representation
        curr_rep = self.rep_observe()
        # Get currently active memories
        active = self.pars['memory']['active'] == True
        # Get distances to current representation
        dists = self.get_distance(curr_loc, self.pars['memory']['location'])
        # Only active memories include memories within distance (np.nan < x is False)
        mems = np.logical_and(active, dists < self.pars['line_of_sight'])
        # If there are no active memories within distance: return empty representation
        if sum(mems) == 0:
            return curr_rep
        # Get current place cell representation
        curr_place = self.reps['place']['rep']([curr_loc])
        # Get place representation for each memory
        mem_place = self.pars['memory']['place'][mems]
        # Calculate dot products of current place with memory places
        dot_prods = mem_place.dot(curr_place) / len(curr_place)
        # Now calculate memory-retrieved representation for each component
        for name, rep in curr_rep.items():
            for r_i in range(len(rep)):
                # Get memories for current representation
                curr_reps = self.pars['memory']['ovc_' + name + '_' + str(r_i)][mems]
                # Get not-none memories within range
                include = np.all(np.logical_not(np.isnan(curr_reps)), axis=-1)
                # Calculate weights: softmax over included dotprods
                weights = np.exp(self.pars['mem_beta'] * dot_prods[include])
                weights = weights / np.sum(weights) if np.sum(weights) > 0 \
                    else []
                # Set representation to weights * included memories
                curr_rep[name][r_i] = None if len(weights)==0 else \
                    [r for r in np.sum(
                        weights.reshape(-1,1) * curr_reps[include], axis=0)]
        return curr_rep        
    
    def rep_encode(self, mem_loc, mem_rep, active=False, curr_loc=None):
        # By default, current physical location is memory location 
        # - but in case of remote replay, that might not actually be the case
        curr_loc = mem_loc if curr_loc is None else curr_loc
        # Collect new memory dictionary
        new_mem = {}
        # Add whether new memory is active
        new_mem['active'] = active
        # Add new memory's location
        new_mem['location'] = mem_loc
        # And place representaiton of memory location
        new_mem['place'] = self.reps['place']['rep']([mem_loc])
        # Add all representations for this memory
        for key, val in mem_rep.items():
            for i, r in enumerate(val):
                new_mem['ovc_' + key + '_' + str(i)] = np.nan if r is None \
                    else np.array(r)
        # Now update the memory array for each field of new memory
        for key, val in new_mem.items():
            self.pars['memory'][key] = self.set_new_mem(self.pars['memory'][key], val)
        # Get distances of inactive memory locations to current physical location
        dists = self.get_distance(curr_loc, self.pars['memory']['location'][
            np.logical_not(self.pars['memory']['active'])])
        # And evaluate whether inactive memories become active:
        # When distance from physical location has exceeded line of sight
        self.pars['memory']['active'][np.logical_not(self.pars['memory']['active'])] = \
            dists > self.pars['line_of_sight']
    
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
        # Representation parameters
        pars_dict['ovc_n'] = 400 # Was 100; increase for higher resolution
        pars_dict['place_n'] = int(pars_dict['ovc_n'] / 4)
        pars_dict['ovc_scale'] = 0.05
        # Simulation parameters        
        pars_dict['explore_max_steps'] = 200
        pars_dict['test_max_steps'] = 20        
        pars_dict['print'] = True
        pars_dict['rep_control'] = False
        pars_dict['experiment'] = 2
        pars_dict['do_evaluate'] = False
        pars_dict['max_eval_steps'] = 100
        pars_dict['x_grid'] = np.linspace(0, 1, 11)
        pars_dict['y_grid'] = np.linspace(0, 1, 11)
        pars_dict['visits'] = np.zeros((len(pars_dict['x_grid']) - 1, 
                                        len(pars_dict['y_grid']) - 1))
        # Replay parameters        
        pars_dict['replay_interval'] = 5 # -1 for no replay
        pars_dict['replay_n'] = 3
        pars_dict['replay_max_steps'] = 10
        pars_dict['replay_dirs'] = 16
        pars_dict['replay_q_weights'] = None
        pars_dict['replay_reverse'] = False
        pars_dict['alpha'] = 0.8
        pars_dict['gamma'] = 0.7
        pars_dict['beta'] = 1
        # Behaviour parameters
        pars_dict['wall_thickness'] = 0.025
        pars_dict['line_of_sight'] = 0.05
        pars_dict['noise'] = [0.5 * 2*np.pi, 0.05]
        pars_dict['max_step'] = 0.1
        pars_dict['step_std'] = 0.01
        pars_dict['dir_std'] = 0.1 * 2*np.pi
        pars_dict['path_int_weight'] = 0.2
        # Memory parameters
        pars_dict['mem_size'] = 5000
        pars_dict['memory'] = self.init_memory(n_mem=pars_dict['mem_size'],
                                               n_place=pars_dict['place_n'])
        pars_dict['mem_beta'] = 1
        # Return default dictionar
        return pars_dict        
                
    def init_env(self, env, wall_thickness = 0.025):
        # Create dictionary of environment variables
        env_dict = {}
        # Create environment
        env_dict['env'] = env
        # Create borders for walls with thickness
        env_dict['walls'] = env.get_border_lines()
        # Return environment dictionary
        return env_dict
    
    def init_representations(self, env):
        # Create population of ovcs
        ovcs = self.get_ovc_population(env, self.pars['ovc_n']);
        # Create population of place cells based on ovc layout
        place = self.get_place_population(ovcs)        
        # Update number of ovcs to the number we ended up with
        self.pars['ovc_n'] = ovcs['n']
        self.pars['place_n'] = ovcs['n']        
        # Create function that produces representation: firing rates of cells
        ovc_rep = lambda x: self.get_location_rates(
            x, ovcs['center'], np.eye(2) * self.pars['ovc_scale'] 
            * min([ovcs['height'], ovcs['width']]))
        place_rep = lambda x: self.get_location_rates(
            x, place['center'], np.eye(2) * self.pars['ovc_scale'] 
            * min([place['height'], place['width']]))        
        # Add representation to ovc dictionary
        ovcs['rep'] = ovc_rep
        place['rep'] = place_rep
        # Return representation dictionary that contains place cells and ovcs
        return {'place': place, 'ovc': ovcs}
    
    def init_policy(self, env, reps):
        # By default, simply use the policy provided by environment
        return 0
    
    def init_memory(self, n_mem=1000, n_place=10):
        # Memory dict holds numpy arrays: one for each representation location
        mem_dict = {}
        # Create include array
        mem_dict['active'] = np.full(n_mem, np.nan)
        # Create memory location array
        mem_dict['location'] = np.full((n_mem, 2), np.nan)
        # Create memory place representation array
        mem_dict['place'] = np.full((n_mem, n_place), np.nan)
        # Get empty representation to see what it looks like
        rep = self.rep_observe()
        # Then create memory array for each representation location
        for key, val in rep.items():
            for i in range(len(val)):
                mem_dict['ovc_' + key + '_' + str(i)] = np.full((n_mem, 2), np.nan)
        # Return empty memory
        return mem_dict
        
    def set_new_mem(self, mems, mem):
        # Shift old memory array
        mems[1:] = mems[0:-1]
        # Add new memory to first entry
        mems[0] = mem
        # Return new memory
        return mems
    
    def get_ovc_population(self, env, N):
        # Get environment surface
        env_surf = (env.xlim[1] - env.xlim[0]) * (env.ylim[1] - env.ylim[0])
        # Divide surface by number of ovcs available - N/4 so all layouts covered
        ovc_surf = env_surf / (N / 4)
        # Get side of square that covers ovc surface
        ovc_side = np.sqrt(ovc_surf)
        # And find number of ovcs needed to cover full area
        ovc_nx = int(np.ceil((env.xlim[1] - env.xlim[0]) / ovc_side))
        ovc_ny = int(np.ceil((env.ylim[1] - env.ylim[0]) / ovc_side))
        # Get width and height of ovc
        ovc_w = (env.xlim[1] - env.xlim[0]) / ovc_nx
        ovc_h = (env.ylim[1] - env.ylim[0]) / ovc_ny
        # Get updated N: ovcs that are actually needed
        N = 4 * ovc_nx * ovc_ny
        # And distribute ovc firing field centers around 0
        ovcs = [[(-ovc_nx + 0.5 + x_i) * ovc_w, (-ovc_ny + 0.5 + y_i) * ovc_h]
                for x_i in range(ovc_nx * 2) for y_i in range(ovc_ny * 2)]
        # Return ovc population centres
        return {'rows': ovc_ny * 2, 'cols': ovc_nx * 2, 'n': N,
                'width': ovc_w, 'height': ovc_h, 'center': ovcs}
    
    def get_place_population(self, ovcs):
        # Reuse ovc cell layout, but only need to cover the environment
        place = {'rows': int(ovcs['rows'] / 2), 'cols': int(ovcs['cols'] / 2),
                 'n': int(ovcs['n'] / 4), 
                 'width': ovcs['width'], 'height': ovcs['width']}
        # And distribute place firing fields covering environment from origin
        place_center = [[(0.5 + x_i) * place['width'], 
                         (0.5 + y_i) * place['height']]
                        for x_i in range(place['cols'])
                        for y_i in range(place['rows'])]
        # Add place center to place dict
        place['center'] = place_center
        # Return ovc population centres
        return place
    
    def get_location_representation(self, location, env=None, rep='ovc', comp=None):
        # If env not provided: use own env
        env = self.env['env'] if env is None else env
        # If component not provided: use all components
        comp = [key for key in env.components.keys()] if comp is None else comp
        # then only select requested components
        comp = [val for key, val in env.components.items()
                if key in comp]
        # Find relative position for each component - or absolute, if doing control
        d = [[l - l_c for l_c, l in zip([0, 0] if self.pars['rep_control'] 
                                        else loc, location)]
             for comp in env.components.values()
             for loc in comp['locations']] if rep == 'ovc' else [[location]]
        # Translate relative position to ovc activity
        representation = self.reps[rep]['rep'](d)
        # Return full representation vector
        return representation
    
    def get_representation_rates(self, representation, env=None):
        # If env not provided: use own env
        env = self.env['env'] if env is None else env        
        # Create list of relative positions from representation to its component
        d = [None if rep_loc is None else
             [l_r - l_c for l_c, l_r in zip(comp_loc, rep_loc)]
             for comp, rep in zip(self.env['env'].components.values(), 
                                  representation.values())
             for comp_loc, rep_loc in zip(comp['locations'], rep)]
        # Find representation rates for each representation; zero if unseen (None)
        r = [np.zeros(self.pars['ovc_n']) if d_i is None else
             self.reps['ovc']['rep']([d_i]) for d_i in d]
        # Return concatenated full representation with zeros for unseen components
        return np.concatenate(r)
    
    def get_location_rates(self, locations, mvn_mean, mvn_cov):
        # Location is array of locations that we want the full population
        # rate for. Shift location by mvn mean, so we can use a single mvn pdf
        x = np.concatenate([np.array(l) - np.array(mvn_mean) for l in locations])
        # Evaluate zero-centred mvn at x, with peak firing rate of 1
        return multivariate_normal.pdf(x, mean=[0, 0], cov=mvn_cov) \
            * (np.sqrt(2 * np.pi) * mvn_cov[0,0])
    
    def get_action(self, prev_action=None):
        # Get previous action
        if prev_action is None:
            prev_dir = random.uniform(0, 2*np.pi)
            prev_step = random.uniform(self.pars['max_step'] * 0.2, self.pars['max_step'])
        else:
            prev_dir, prev_step = prev_action
        # Get direction from optimal policy in environment
        curr_dir = prev_dir + np.random.normal() * self.pars['dir_std']
        curr_step = max(self.pars['max_step'] * 0.2, 
                        min(self.pars['max_step'], 
                            prev_step + np.random.normal() * self.pars['step_std']))
        # Return new action
        return [curr_dir, curr_step]
    
    def get_replay_action(self, location, env=None):
        # If env not provided: use own env
        env = self.env['env'] if env is None else env        
        # Get q-values learned in replay for place state at curren location
        qs = np.matmul(self.reps['place']['rep']([location]),
                       self.pars['replay_q_weights'])
        # Get step from learned replay weights
        action = [
            self.get_action_decoded(np.where(qs == max(qs))[0][0]),
            min(self.pars['max_step'], self.get_goal_dist(location, env))]
        return action        
    
    def get_optimal_action(self, location):
        # Get direction from optimal policy in environment
        curr_dir = self.env['env'].get_location_direction(location)
        # Take max step, or distance to goal if that's closer
        curr_step = min(self.pars['max_step'], self.get_goal_dist(location))
        return [curr_dir, curr_step]  
    
    def get_optimal_dist(self, loc, env, replay=False, pol_env=None):
        # Environment to derive policy may be different from real environment,
        # but set it to real environment by defeault
        if pol_env is None: pol_env = env
        # Get distances to reward loc under optimal policy
        opt_path = [loc]
        while self.get_goal_dist(opt_path[-1], env) > self.pars['line_of_sight'] \
            and len(opt_path) < self.pars['max_eval_steps']:
                # Get optimal direction from curren loc
                opt_dir = pol_env.get_location_direction(opt_path[-1])
                # In case of replay action: use discretised optimal direction
                if replay:
                    opt_dir = self.get_action_decoded(self.get_action_encoded(opt_dir))
                # Get optimal step
                action = [
                    opt_dir,
                    min(self.pars['max_step'], self.get_goal_dist(opt_path[-1], env))]  
                # And update location
                opt_path.append(self.loc_transition(opt_path[-1], action, env=env))
        # If optimal policy can't reach goal: set distance to -1
        opt_dist = -1 if len(opt_path) == self.pars['max_eval_steps'] \
            else np.sum([self.get_distance(l1, l2) 
                         for l1, l2 in zip(opt_path[:-1], opt_path[1:])]) \
                + self.get_goal_dist(opt_path[-1])
        return opt_path, opt_dist
    
    def get_learned_action(self, location=None, representation=None):
        # If location is not provided: use location from reward representation
        location = representation['reward'][0] if location is None \
            else location        
        # Base model doesn't learn, so just return optimal action
        return self.get_optimal_action(location)
    
    def get_action_encoded(self, direction, n=None):
        # If n not specified: use n from parameters
        n = self.pars['replay_dirs'] if n is None else n
        return int(np.round((direction % (2 * np.pi)) / (2 * np.pi) * n)) % n

    def get_action_decoded(self, i, n=None):
        # If n not specified: use n from parameters
        n = self.pars['replay_dirs'] if n is None else n
        return  (i / n) * 2 * np.pi
    
    def get_goal_locs(self, env):
        # Find all goal locations in environment
        return [flatten(comp['locations']) for comp in env.components.values()
                if comp['type'] == 'shiny']
    
    def get_goal_dist(self, location, env=None):
        # If env not specified: use default env
        env = self.env['env'] if env is None else env
        # Find all goals
        goals = self.get_goal_locs(env)
        # Find distances to goals
        dists = [self.get_distance(location, goal) for goal in goals]
        # Return distance to closest goal
        return min(dists)        
    
    def get_wall_normal(self, wall, loc):
        # Get dy, dx from wall
        dy = wall.xy[1][1] - wall.xy[1][0]
        dx = wall.xy[0][1] - wall.xy[0][0]
        # Get normal vector
        v = np.array([-1, dx / dy] if dx == 0 else [dy / dx, -1])
        # Find component along normal of vector FROM loc TO wall
        comp_to_loc = np.dot(np.array([wall.xy[0][0] - loc[0], 
                                       wall.xy[1][0] - loc[1]]), v)
        # Flip v if the direction to wall is negative
        v = v * np.sign(comp_to_loc) if abs(comp_to_loc) > 0 else v
        # Normalise
        v = v / np.sqrt(np.sum(v**2))
        # Return wall normal vector
        return v
    
    def get_wall_parallel(self, wall):
        # Get dy, dx from wall
        dy = wall.xy[1][1] - wall.xy[1][0]
        dx = wall.xy[0][1] - wall.xy[0][0]
        # Get parallel vector
        v = np.array([dx, dy])
        # Normalise
        v = v / np.sqrt(np.sum(v**2))
        # Return wall normal vector
        return v    
    
    def get_wall_border(self, wall, v):
        # Get parallel and orthogonal vectors for wall
        v_par = self.get_wall_parallel(wall)
        # Get wall start and end, taking wall thickness into account
        wall_start = [wall.xy[0][0] + self.pars['wall_thickness'] * v_par[0],
                      wall.xy[1][0] + self.pars['wall_thickness'] * v_par[1]]
        wall_stop = [wall.xy[0][1] - self.pars['wall_thickness'] * v_par[0],
                     wall.xy[1][1] - self.pars['wall_thickness'] * v_par[1]]
        # Then create collision wall on side of 
        col_wall = LineString(
            [(wall_start[0] - self.pars['wall_thickness'] * v[0],
              wall_start[1] - self.pars['wall_thickness'] * v[1]),
             (wall_stop[0] - self.pars['wall_thickness'] * v[0],
              wall_stop[1] - self.pars['wall_thickness'] * v[1])])
        # Return both walls
        return col_wall
        
    def is_parallel(self, vec1, vec2, ignore_opposite=True):
        # Calculate absolute value of dot product
        dot_prod = np.sum(np.array(vec1) * np.array(vec2))
        # If the vectors are parallel but opposite, the dot product will be negative
        if ignore_opposite: dot_prod = np.abs(dot_prod)
        # Calculate product of vector norms
        norm_prod = np.prod([np.sqrt(np.sum(np.square(np.array(v)))) for v in [vec1, vec2]])
        # Vectors are parallel if their dot product equals product of norms
        return dot_prod == norm_prod
    
    def get_distance(self, loc_from, loc_to):
        # Calculate distances to all requested locations
        return np.sqrt(np.sum((np.array(loc_from) - np.array(loc_to))**2, axis=-1))
    
    def get_memory_distances(self, curr_loc, mems=None):
        # If memory not provided: use full memory
        mems = self.pars['memory'] if mems is None else mems
        # Get memory locations stacked in numpy array for efficiency
        mem_loc = np.stack([mem[0] for mem in mems])
        # Get distances to current representation
        return np.sqrt(np.sum(np.square(np.array(curr_loc) - mem_loc), axis=1))
        
    def get_visits(self, curr_loc):
        # Get number of visits to each evuation grid location, and update 
        # correct entry in visits matrix
        x_i = [i for i, (x_1, x_2) in 
               enumerate(zip(self.pars['x_grid'][:-1], self.pars['x_grid'][1:]))
               if curr_loc[0] > x_1 and curr_loc[0] < x_2][0]
        y_i = [i for i, (y_1, y_2) in 
               enumerate(zip(self.pars['x_grid'][:-1], self.pars['x_grid'][1:]))
               if curr_loc[1] > y_1 and curr_loc[1] < y_2][0]
        # Increase visits 1, now that we have visitied that location
        self.pars['visits'][x_i, y_i] += 1
        # And return current visits
        return self.pars['visits'][x_i, y_i]
    
    def do_evaluate(self, prev_loc, curr_loc):
        # Only evaluate if this step crossed any grid lines,
        # So this is a new entry to a grid position
        x_cross = any([(prev_loc[0] < x_grid and curr_loc[0] > x_grid)
                       or (prev_loc[0] > x_grid and curr_loc[0] < x_grid)
                       for x_grid in self.pars['x_grid']])
        y_cross = any([(prev_loc[1] < y_grid and curr_loc[1] > y_grid)
                       or (prev_loc[1] > y_grid and curr_loc[1] < y_grid)
                       for y_grid in self.pars['y_grid']])
        # Return either crossing        
        return self.pars['do_evaluate'] and (x_cross or y_cross)
        
    def get_sample(self, probability):
        # Sample from array containing probability distribution
        return np.random.choice(len(probability), p=probability)
                
    def get_free_location(self, env=None):
        # If env nog provided: use default env
        env = self.env['env'] if env is None else env
        # Sample location
        loc = [random.uniform(env.xlim[0] + self.pars['wall_thickness'], 
                              env.xlim[1] - self.pars['wall_thickness']),
               random.uniform(env.ylim[0] + self.pars['wall_thickness'],
                              env.ylim[1] - self.pars['wall_thickness'])]
        # Create point object for location
        point = Point(loc)
        # Calculate distance to all walls
        d = [point.distance(wall) 
             for wall in env.get_border_lines(include_borders=False)]
        # If there are any walls: make sure the new point is not on wall
        if len(d) > 0:
            while min(d) < self.pars['wall_thickness']:
                # Sample location
                loc = [random.uniform(env.xlim[0] + self.pars['wall_thickness'], 
                                      env.xlim[1] - self.pars['wall_thickness']),
                       random.uniform(env.ylim[0] + self.pars['wall_thickness'],
                                      env.ylim[1] - self.pars['wall_thickness'])]
                # Create point objct for location
                point = Point(loc)
                # Calculate distance to all walls
                d = [point.distance(wall) 
                     for wall in env.get_border_lines(include_borders=False)]
        # Return final location
        return loc
    
class FitPolicy(Model):
    
    # Evaluate policy from partial vs full representation
    def exp_evaluate(self, curr_step):
        # Update visits and add to step
        curr_step['visits'] = self.get_visits(curr_step['location'])
        
        # Get additional distance for full representation
        d = self.learn_get_location_path(self.policy_net, curr_step['location'],
                                         self.env['env'])
        # Update curr step with additional distance for full rep
        curr_step['d_full_rep'] = d
        
        # Get additional distance for current partial representation
        d = self.learn_get_location_path(self.policy_net, curr_step['location'],
                                         self.env['env'], curr_step['representation'])
        # Update curr step with additional distance for full rep
        curr_step['d_curr_rep'] = d
    
        # Return updated step
        return curr_step
        
    def replay_evaluate(self, curr_step):
        # Update visits and add to step
        curr_step['visits'] = self.get_visits(curr_step['location'])        
        
        # Find if full representation gets you to goal
        d = self.learn_get_location_path(self.policy_net, curr_step['location'],
                                         self.env['env'])        
        # Update curr step with additional distance for full rep
        curr_step['d_memory'] = d        
        
        # Find if following replay would get you to goal
        d = self.replay_get_location_path(curr_step['location'], self.env['env'])
        # Add d to current step
        curr_step['d_bellman'] = d
        
        # Return updated step
        return curr_step              
        

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
        # Bias for selecting locations where policy doesn't go straight to goal
        pars_dict['DQN']['loc_bias'] = 0
        # Number of training episodes
        pars_dict['DQN']['episodes'] = 20
        # Number of training steps per episode
        pars_dict['DQN']['steps'] = 500
        # Number of transitions to sample from memory for traning
        pars_dict['DQN']['mem_sample'] = 5
        # Number of transitions in memory
        pars_dict['DQN']['mem_size'] = 100 * pars_dict['DQN']['mem_sample']
        # Number of maximum steps to try when evaluating performance
        pars_dict['max_eval_steps'] = 100
        # Hidden layer dimension for DQN
        pars_dict['DQN']['hidden_dim'] = None                
        # Save model parameters after training - set to None to not save
        pars_dict['DQN']['save'] = './trained/' \
            + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pt'      
        # Load model parameters instead of training - set to None to not load
        pars_dict['DQN']['load'] = None #'./sim/exp2/env1_rep1.pt'
        # 20211125_103123.pt: single wall NOT ANYMORE
        # 20211125_105421.pt: two walls NOT ANYMORE
        return pars_dict
    
    def get_learned_action(self, location=None, representation=None):
        # If location is not provided: use location from reward representation
        location = representation['reward'][0] if location is None \
            else location
        # If representation is not provided: get action from full representation
        if representation is None:
            # Get direction from representation for location
            curr_dir = self.learn_get_location_direction(
                self.policy_net, [location], self.env['env'])[0]
        else:            
            # Get direction from representation for location
            curr_dir = self.learn_get_representation_direction(
                self.policy_net, [representation], self.env['env'])[0]
        # Take max step, or distance to goal if that's closer
        curr_step = min(self.pars['max_step'], self.get_goal_dist(location))
        return [curr_dir, curr_step]      
    
    def init_policy(self, env, reps):        
        # Try loading policy net
        self.policy_net = self.load_policy_net()
        # If loading wasn't successful: learn policy net
        self.policy_net = self.learn_policy_net() if self.policy_net is None \
            else self.policy_net
    
    def load_policy_net(self):
        # If loading disabled: return None
        if self.pars['DQN']['load'] is None:
            return None
        else:
            try:
                # Get dimension of representation: concatenate all representations
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
        # Store current file so it's easy to retrieve parameters
        shutil.copy2('./model.py', self.pars['DQN']['save'].replace('.pt','.py'))
        
        # Get network dimensions
        d_in, d_out = self.learn_init_network()
        
        # Get wall representation parameters
        wall_length, wall_start = self.learn_init_network_walls()
        
        # Initialise networks
        policy_net = WallDQN(d_in, d_out, wall_length, wall_start, 
                             hidden_dim=self.pars['DQN']['hidden_dim'])

        # Initialise networks
        #policy_net = DQN(d_in, d_out)
        
        # Initialise optimiser
        optimiser = torch.optim.Adam(policy_net.parameters(), lr = 1e-3)
        # Create a tensor board to stay updated on training progress. 
        # Start tensorboard with tensorboard --logdir=./trained
        writer = SummaryWriter(self.pars['DQN']['save'].replace('.pt','_tb'))                
                
        # Run training loop
        for i_episode in range(self.pars['DQN']['episodes']):
            # Get all variables to start episode
            envs, location, state = self.learn_init_episode(
                self.env['env'], self.pars['DQN']['envs'])
            # Now run around through environments
            for t in range(self.pars['DQN']['steps']):
                # Calculate current number of training steps
                steps = i_episode * self.pars['DQN']['steps'] + t
                
                # Do transition: choose action, find new state, add to memory
                location, state = self.learn_do_transition(
                    state, steps, policy_net, envs, location)                
                # Get optimal directions at each location
                next_dirs = torch.tensor([env.get_location_direction(loc)
                                          for loc, env in zip(location, envs)],
                                         dtype=torch.float)
                # And get what the policy net would predict at those locations
                next_preds = policy_net(state)
                # Learn action values from correct policy 
                self.learn_optimise_step(policy_net, optimiser, next_dirs, next_preds)                
                    
                # Display progress
                if t % 50 == 0:
                    # Calculate optimality from angle differences
                    ang_err = np.mean(np.array(
                        self.learn_evaluate_performance(policy_net, envs, mode='ang')))
                    print('-- Step', t, '/', self.pars['DQN']['steps'],
                          'angle error', ang_err)
                    writer.add_scalar('error', ang_err, steps)                   
            # Save policy net and performance at the end of each episode
            if self.pars['DQN']['save'] is not None:
                torch.save(policy_net.state_dict(), self.pars['DQN']['save'])          
            # Display progress
            print('- Finished episode', i_episode, '/', self.pars['DQN']['episodes'])
        # Calculate and save final accuracy    
        if self.pars['DQN']['save'] is not None:
            # Calculate optimality from distances following policy
            curr_optimality = np.array(
                self.learn_evaluate_performance(policy_net, envs))
            curr_arrived = np.logical_and(
                curr_optimality > -1, 
                curr_optimality < self.pars['max_eval_steps'])
            curr_failed = np.logical_and(
                curr_optimality > -1, 
                curr_optimality == self.pars['max_eval_steps'])
            curr_d = np.mean(curr_optimality[curr_arrived]) \
                if any(flatten(curr_arrived)) else 100
            curr_o = np.sum(curr_arrived) / (np.sum(curr_arrived) 
                                             + np.sum(curr_failed))
            print('Finished training after ', steps, 'steps',
                  'optimilaty', curr_o,
                  'mean dist', curr_d)            
            with open(self.pars['DQN']['save'].replace('.pt','.npy'), "wb") as f:
                np.save(f, curr_optimality)              
        # Return trained policy network
        return policy_net    
    
    def learn_init_network(self):
        # Find dimensions of input and output of network
        # Input: dimension of state by concatenating all representations
        d_in = len(self.get_location_representation([0,0]))
        # Output: cos(theta), sin(theta)
        d_out = 2
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
        # Get type and dimension for each component        
        for val in self.env['env'].components.values():
            # Get length of this component's representation
            curr_length = self.pars['ovc_n'] * len(val['locations'])
            # Check if this is a wall representation
            if val['type'] == 'wall':
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
        # Initialise locations: random location for each environment
        location = self.learn_get_location_biased(envs) \
            if self.pars['DQN']['loc_bias'] > 0 else self.learn_get_location(envs)
        # And get state for these locations
        state = self.learn_get_state(location, envs)
        # Return all variables for starting an episode
        return envs, location, state
        
    def learn_do_transition(self, state, steps, policy_net, envs, 
                            location):
        # Select and perform an action
        location = self.learn_get_location_biased(envs) \
            if self.pars['DQN']['loc_bias'] > 0 else self.learn_get_location(envs)
        state = self.learn_get_state(location, envs)
        # Return updated location, state, and memory          
        return location, state
    
    def learn_optimise_step(self, policy_net, optimiser, dirs, preds):
        # Calculate cos and sin of dirs
        ys = torch.stack([torch.cos(dirs), torch.sin(dirs)], dim=1)
        # Compute cross-entropy loss
        criterion = torch.nn.MSELoss()
        loss = criterion(preds, ys)

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
            # Finally add environment to list of training environments
            envs.append(new_env)
        return envs
        
    def learn_get_location(self, envs):
        # Randomly select location from each environment
        return [self.get_free_location(env) for env in envs]
    
    def learn_get_location_biased(self, envs, n_sample=10):
        # Randomly select location from each environment, but create bias
        # for locations where correct direction is further from shiny direction
        env_locs = []
        for env in envs:
            # Get shiny loc
            reward_loc = [comp['locations'][0] for comp in env.components.values()
                          if comp['type'] == 'shiny'][0]
            # Sample random locations
            locs = [[random.uniform(env.xlim[0], env.xlim[1]),
                     random.uniform(env.ylim[0], env.ylim[1])] for _ in range(n_sample)]
            # Get optimal directions
            dirs = np.array([env.get_location_direction(loc) for loc in locs])
            # Get angle difference between optimal policy and to shiny
            reward_dir = np.arctan2(reward_loc[1] - np.array(locs)[:, 1],
                                    reward_loc[0] - np.array(locs)[:, 0])
            # Get absolute angle difference
            ang_diff = np.abs(((dirs - reward_dir)*360/(2*np.pi) + 180) % 360 - 180)
            # Set sampling probability to some offset plus normalised angle difference
            p = (1 - self.pars['DQN']['loc_bias']) * np.ones(ang_diff.shape) / n_sample\
                + self.pars['DQN']['loc_bias'] * (ang_diff / np.sum(ang_diff)
                                                  if np.sum(ang_diff) > 0 else
                                                  np.ones(ang_diff.shape) / n_sample)
                   
            # Finally: sample location using probability
            env_locs.append(random.choices(locs, p)[0])
        return [[random.uniform(env.xlim[0], env.xlim[1]),
                 random.uniform(env.ylim[0], env.ylim[1])] for env in envs]    

    def learn_get_state(self, locs, envs):
        # Build state tensor: representations of each location
        return torch.stack([
            torch.tensor(self.get_location_representation(loc, env), dtype=torch.float)
            for loc, env in zip(locs, envs)])
    
    def learn_evaluate_performance(self, policy_net, envs, mode='dist'):
        # Find in each env how far off the predicted direction is
        optimality = []
        # We'll use the policy net to get a policy, but we dont want to optimise
        with torch.no_grad():
            # Switch policy to evaluation mode
            policy_net.eval()
            # Run through training environments
            for env in envs:
                # Evaluate performance for requested mode
                if mode == 'dist':
                    # Calculate difference between optimal and learned distance
                    optimality.append(self.learn_evaluate_path(policy_net, env))
                elif mode == 'ang':
                    # Calculate mean difference between real and predicted angle
                    optimality.append(self.learn_evaluate_direction(policy_net, env))
            # Switch policy net back to train mode
            policy_net.train()
        # Return optimality
        return optimality    
    
    def learn_evaluate_direction(self, policy_net, env, n_locs=3):
        # Collect some random locations
        locs = [self.learn_get_location([env])[0] for _ in range(n_locs)]
        # Get correct angle for each of these
        dirs = np.array([env.get_location_direction(loc) for loc in locs])
        # Get predicted angle for each location
        angles = self.learn_get_location_direction(policy_net, locs, env)
        # Get angle difference between true and predicted direction
        d = dirs - angles
        d = (d*360/(2*np.pi) + 180) % 360 - 180
        # Return mean absolute angular distance
        return np.abs(d)
    
    def learn_evaluate_path(self, policy_net, env, n_locs=50):
        # Collect some random locations
        locs = [self.learn_get_location([env])[0] for _ in range(n_locs)]
        # Get distance difference between optimal and learned policy
        dists = np.array([self.learn_get_location_path(policy_net, loc, env) 
                          for loc in locs])
        # Return distances
        return dists
    
    def learn_get_location_direction(self, policy_net, locs, env):
        # Get states for these locs
        states = self.learn_get_state(locs, [env for _ in locs])
        # Get predicted cos and sin of angle at l
        preds = policy_net(states).detach().numpy()
        # Get angles from predicted cos and sin
        angles = np.arctan2(preds[:,1], preds[:,0])
        # Return angles predicted by policy net
        return angles
    
    def learn_get_representation_direction(self, policy_net, reps, env):
        # Get states for these reps
        states = torch.stack([
            torch.tensor(self.get_representation_rates(rep, env), dtype=torch.float)
            for rep in reps])
        # Get predicted cos and sin of angle at l
        preds = policy_net(states).detach().numpy()
        # Get angles from predicted cos and sin
        angles = np.arctan2(preds[:,1], preds[:,0])
        # Return angles predicted by policy net
        return angles    
    
    def learn_get_location_path(self, policy_net, loc, env, rep=None):
        # Get distances to reward loc under optimal policy
        opt_path, opt_dist = self.get_optimal_dist(loc, env, replay=True)
        # Now do the same for learned policy
        learned_path = [loc]
        while self.get_goal_dist(learned_path[-1], env) > self.pars['line_of_sight'] \
            and len(learned_path) < self.pars['max_eval_steps']:
                if rep is None:
                    # If comp not provided: get action from full representation at loc
                    action = [
                        self.learn_get_location_direction(
                            policy_net, [learned_path[-1]], env)[0],
                        min(self.pars['max_step'], 
                            self.get_goal_dist(learned_path[-1], env))]
                else:
                    # If comp is provided: use partial representation to get direction
                    curr_rep = self.rep_observe()
                    # Update the requested components by setting location
                    for key in rep.keys():
                        curr_rep[key] = [None if r is None else learned_path[-1]
                                         for r in rep[key]]
                    # Get action from partial representation
                    action = [
                        self.learn_get_representation_direction(
                            policy_net, [curr_rep], env)[0],
                        min(self.pars['max_step'], 
                            self.get_goal_dist(learned_path[-1], env))]
                # And update location
                learned_path.append(self.loc_transition(learned_path[-1], action, env=env))
        # If learned policy didn't reach goal: set distance to -1
        learned_dist = -1 if len(learned_path) == self.pars['max_eval_steps'] \
            else np.sum([self.get_distance(l1, l2) 
                         for l1, l2 in zip(learned_path[:-1], learned_path[1:])]) \
                + self.get_goal_dist(opt_path[-1])
                
        #import pdb; pdb.set_trace()
        # Calculate difference between optimal dist and learned dist, if both arrived
        return -1 if opt_dist == -1 \
            else (self.pars['max_eval_steps'] if learned_dist == -1 
                  else learned_dist - opt_dist)      

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