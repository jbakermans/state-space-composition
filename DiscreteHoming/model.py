#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:21:34 2021

@author: jbakermans
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path

# Base model with standard functionality for exploration and escape
class Model():    

    def __init__(self, env, pars=None):
        # Initialise environment: set home, retrieve distances
        self.env = self.init_env(env)
        # Initialise parameters, copying over all fields and setting missing defaults
        self.pars = self.init_pars(pars)        
        # Initialise model: set cells initialise all required synaptic weights
        self.model = self.init_model()
        
    def reset(self, env, pars=None):
        # Reset a model to restart fresh (e.g. with new params, or different env)
        # This simple calls __init__, but has a separate name for clarity
        self.__init__(env, pars)
        
    def simulate(self):
        # Simulate exploration
        explore = self.sim_explore()
        # Simulate escape from final step of explore
        escape = self.sim_escape(explore[-1])
        # And return both
        return explore, escape
        
    def sim_explore(self):
        # Start with empty list that will contain info about each explore step
        explore = []
        # Exploration stage: wander around randomly through environment
        for i in range(self.pars['sim']['explore_steps']):
            # Run step
            curr_step = self.exp_step(explore)
            # Replay at fixed intervals. Set interval to -1 to never replay
            curr_step['replay'] = self.sim_replay(dict(curr_step)) \
                if i % self.pars['sim']['replay_interval'] == 1 else []
            # Store all information about current step in list
            explore.append(curr_step)  
            # Display progress
            if self.pars['sim']['print']:
                print('Finished explore', i, '/', self.pars['sim']['explore_steps'], 
                      'at', curr_step['location'], 
                      'virtually at', self.rep_decode(curr_step['representation']))        
        return explore
    
    def sim_escape(self, prev_step):
        # Start with empty list that will contain info about each escape step
        escape = [] 
        # Homing stage: follow policy until arriving at home
        while self.rep_decode(prev_step['representation']) != self.env['home'] \
            and prev_step['i'] < self.pars['sim']['explore_steps'] \
                + self.pars['sim']['escape_max_steps'] :
            # Get previous location and action       
            if len(escape) > 0:
                prev_step = escape[-1]
            # Run step
            curr_step = self.esc_step(prev_step)
            # Store all information about current step in steps list
            escape.append(curr_step)
            # Display progress
            if self.pars['sim']['print']:            
                print('Finished escape', curr_step['i'], 
                      'at', curr_step['location'], 
                      'virtually at', self.rep_decode(curr_step['representation']))        
        # Goes on for one step too long due to how I set up the while loop
        return escape[:-1]
    
    def sim_replay(self, start_step):
        # Start with empty list that will contain lists of replays
        replay = []
        # If replaying from home: initialise start step at home
        if self.pars['sim']['replay_from_home']:
            start_step['location'], start_step['representation'] = self.exp_init()
        # Set k-variable thate counts replay steps to 0 to indicate start of replay
        start_step['k'] = 0
        # Each replay consists of a list of replay steps, which are dicts with step info
        for j in range(self.pars['sim']['replay_n']):
            # Start replay at current representation
            curr_replay = []
            curr_step = start_step
            # Replay until you think you are home, or reaching max steps
            while (self.pars['sim']['replay_from_home'] or 
                   curr_step['location'] != self.env['home']) and \
                curr_step['k'] < self.pars['sim']['replay_max_steps']:
                # Take replay step
                curr_step = self.replay_step(curr_step)
                # Add this replay step to current replay list
                curr_replay.append(curr_step)
            # Display progress
            if self.pars['sim']['print']:            
                print('Finished replay', j, '/', self.pars['sim']['replay_n'], 
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
        # Select action for current location
        curr_action = self.exp_action(curr_loc)
        # Return summary for this step
        curr_step = {'i': len(explore), 'location': curr_loc,
                     'representation': curr_rep, 'action': curr_action}
        return curr_step     
    
    def exp_init(self):
        # Set initial location and representation for explore walk: home      
        curr_loc = self.env['home']
        curr_rep = self.rep_encode(curr_loc)
        return curr_loc, curr_rep        
    
    def exp_transition(self, prev_step):
        # Collect previous step
        prev_loc = prev_step['location']
        prev_action = prev_step['action']
        prev_rep = prev_step['representation']
        # Transition both location and representation
        curr_loc = self.loc_transition(prev_loc, prev_action)
        curr_rep = self.rep_transition(prev_rep, prev_action)
        # And return both
        return curr_loc, curr_rep
        
    def exp_action(self, curr_loc):
        # Find policy at current location
        curr_pol = self.loc_policy(curr_loc)
        # And choose action from policy
        return self.get_action(curr_pol)        
    
    def esc_step(self, prev_step):
        # Transition location and representation
        curr_loc, curr_rep = self.esc_transition(prev_step)
        # Get action from representation
        curr_action = self.esc_action(curr_rep)
        # Return summary for this step
        curr_step = {'i': prev_step['i'] + 1, 'location': curr_loc,
                     'representation': curr_rep, 'action': curr_action}
        return curr_step
            
    def esc_transition(self, prev_step):
        # By default, transition during escape is identical to transition during explore
        return self.exp_transition(prev_step)                
        
    def esc_action(self, curr_rep):        
        # Get greedy policy for current representation to get home quickly
        curr_pol = self.rep_policy_greedy(curr_rep)
        # Select action from escape policy
        return self.get_action(curr_pol)
    
    def replay_step(self, prev_step):        
        if prev_step['k'] == 0:
            # In the first step of replay: copy location and representation from last real step
            curr_loc, curr_rep = prev_step['location'], prev_step['representation']            
        else:
            # Transition location and representation
            curr_loc, curr_rep = self.replay_transition(prev_step)
        # Decide next action based on softmax of Q-values
        curr_action = self.replay_action(curr_rep)
        # Set current step 
        curr_step = {'i': prev_step['i'], 'k': prev_step['k'] + 1, 'location': curr_loc,
                     'representation': curr_rep, 'action': curr_action}
        # And return all information about this replay step
        return curr_step    

    def replay_transition(self, prev_step):
        # Collect previous step        
        prev_action = prev_step['action']
        prev_rep = prev_step['representation']        
        # In replay transition the representation transitions...
        curr_rep = self.rep_transition(prev_rep, prev_action)
        # ... But the location is simply decoded from representation
        curr_loc = self.rep_decode(curr_rep)
        # Return both
        return curr_loc, curr_rep    
    
    def replay_action(self, curr_rep):
        # Get policy from q-weights for current representation
        curr_pol = self.rep_policy(curr_rep)
        # If replaying away from home: invert policy
        curr_pol = (1-curr_pol) if self.pars['sim']['replay_from_home'] else curr_pol
        # Disallow stand-still actions in replay and renormalise
        curr_pol = np.concatenate([[0], curr_pol[1:]]) / np.sum(curr_pol[1:])
        # Select action from escape policy
        return self.get_action(curr_pol)        

    def loc_transition(self, prev_loc, prev_action):
        # Get new location from environment transition of previous action
        return int(np.flatnonzero(np.cumsum(
            self.env['env'].locations[prev_loc]['actions'][prev_action]['transition'])
            > np.random.rand())[0])
    
    def loc_policy(self, curr_loc):
        # Get policy from location (i.e. as described in environment object)
        return np.array([action['probability']
                         for action in self.env['pol'][curr_loc]['actions']])        
        
    def rep_decode(self, representation, key='ovc'):
        # Get appropriate representation
        representation = self.rep_select(representation, key)
        # Default decoding funtion: l_t = argmax(x_t * D)
        return np.argmax(np.matmul(representation, self.model['mat_decode']))
    
    def rep_encode(self, location):
        # Default encoding funtion: x_t = one-hot(l_t) * E
        return np.matmul(np.eye(self.env['env'].n_locations)[location,:], 
                         self.model['mat_encode'])
    
    def rep_transition(self, prev_rep, prev_action, key='ovc'):
        # Get appropriate representation
        prev_rep = self.rep_select(prev_rep, key)        
        # Same for transition matrix: select appropriate one
        trans_mat = self.model['mat_trans'][key] if isinstance(self.model['mat_trans'], dict) \
            else self.model['mat_trans']
        # Get probability distribution over next locations from (noisy) transition
        trans_prob = np.matmul(prev_rep, trans_mat[prev_action])
        # And sample from transition probability
        curr_rep = np.eye(trans_prob.shape[0])[self.get_sample(trans_prob)]        
        return curr_rep
    
    def rep_retrieve(self, location):
        # Get probability distribution over retrieved representations from
        # default retrieval funtion: x_t = one-hot(l_t) * R    
        ret_prob = np.matmul(np.eye(self.env['env'].n_locations)[location], 
                             self.model['mat_retrieve'])
        # Sample from retrieval probability
        curr_rep = np.eye(ret_prob.shape[0])[self.get_sample(ret_prob)]
        return curr_rep
    
    def rep_policy(self, curr_rep, key='ovc'):
        # Get appropriate representation
        curr_rep = self.rep_select(curr_rep, key)        
        # Get q-values from representation and q-weights
        curr_qs = np.matmul(curr_rep, self.model['mat_Q'])
        # Then calculate policy from softmax over q-values
        curr_exp = np.exp(self.pars['model']['beta'] * curr_qs)
        return curr_exp / np.sum(curr_exp)                
        
    def rep_policy_greedy(self, curr_rep, key='ovc'):
        # Get appropriate representation
        curr_rep = self.rep_select(curr_rep, key)                
        # Get q-values from representation and q-weights
        curr_qs = np.matmul(curr_rep, self.model['mat_Q'])
        # Greedy policy: equal probability for only max q-values
        return (curr_qs == np.max(curr_qs)) / np.sum((curr_qs == np.max(curr_qs)))        

    def rep_select(self, representation, key):
        # This is a bit ugly but it's a consequence of having multiple represenations
        # (place, as memory key, and ovc, as memory value) in landmark model:
        # Represenation can be a single vector, or a dictionary of vectors
        # If it's a vector, simply return the vector
        # If it's a dictionary pick out one from key argument
        return representation[key] if isinstance(representation, dict) \
            else representation        

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
        # Create sub-dict for model parameters
        pars_dict['model'] = {}
        # Set required parameters to default values:
        # Number of cells
        pars_dict['model']['n_cells'] = self.env['env'].n_locations
        # Transition noise level (softmax temperature)
        pars_dict['model']['rep_transition_noise'] = 0     
        # Create sub-dict for simulation parameters
        pars_dict['sim'] = {}
        # Set required parameters to default values:
        # Print progress in terminal
        pars_dict['sim']['print'] = True
        # Steps to take during exploration
        pars_dict['sim']['explore_steps'] = 25
        # Maximum number of steps to take during escape
        pars_dict['sim']['escape_max_steps'] = 25
        # Interval during explore for running replay. -1 for no replay
        pars_dict['sim']['replay_interval'] = -1
        # Set whether replay goes towards or away from home
        pars_dict['sim']['replay_from_home'] = False
        # Return default dictionar
        return pars_dict        
                
    def init_env(self, env):
        # Create dictionary of environment variables
        env_dict = {}
        # Create environment from json file in envs directory
        env_dict['env'] = env
        # Use shiny mechanism to create home in fixed position for easy comparison
        env_dict['env'].init_shiny({'n': 1, 'can_be_shiny': 
                                    [int(np.sqrt(env.n_locations)*1.5)]})
        # Get location of home from world
        env_dict['home'] = env.shiny['locations'][0]
        # Get distance matrix between all locations [i, j]: FROM i TO j
        env_dict['dist'] = shortest_path(csgraph=np.array(env.adjacency), 
                                         directed=True)
        # Set environment default policy during exploration
        # For random diffusive behaviour use env.locations
        # Alternatively, generate more explore-like behaviour away from home
        # env_dict['pol'] = env.policy_opposite(env.shiny['policies'][0])
        env_dict['pol'] = env.policy_distance_opposite(env_dict['home'])
        # Make sure all unavailable actions, indicated by all-zero transitions,
        # now get a stand-still transition
        for l_i, location in enumerate(env_dict['env'].locations):
            for action in location['actions']:
                if np.sum(action['transition']) == 0:
                    action['transition'] = np.eye(env_dict['env'].n_locations)[l_i]
        # Return environment dictionary
        return env_dict
    
    def init_model(self):
        # Create dictionary of environment variables
        model_dict = {}
        # Create n_cells x n_cells confusion matrix C
        # C is a single matrix if noise is number, or dict if noise is dict
        model_dict['mat_confuse'] = \
            {key: self.get_mat_confusion(val) 
             for key, val in self.pars['model']['rep_transition_noise'].items()}\
                if isinstance(self.pars['model']['rep_transition_noise'], dict) \
                    else self.get_mat_confusion(self.pars['model']['rep_transition_noise'])
        # Create n_cells x n_locs decoding matrix D
        model_dict['mat_decode'] = self.get_mat_decode()
        # Create n_locs x n_cells encoding matrix E
        model_dict['mat_encode'] = self.get_mat_encode()        
        # Create n_cells x n_cells representation transition matrix T for each action
        # Use confusion matrix that were just created - or dict, if noise is dict
        model_dict['mat_trans'] = \
            {key: self.get_mat_transition(val) 
             for key, val in model_dict['mat_confuse'].items()}\
                if isinstance(model_dict['mat_confuse'], dict) \
                    else self.get_mat_transition(model_dict['mat_confuse'])        
        # Create n_cells x n_actions representation-action weights matrix
        model_dict['mat_Q'] = self.get_mat_Q()  
        # Return model dictionary
        return model_dict
    
    def get_mat_confusion(self, noise):
        # Create confusion matrix between locations, where each location can be
        # confused for its neighbours with a probability given by noise
        # Confusion matrix: C_ij is probability that REAL i gets REPLACED BY j
        # The confusion matrix is defined for representations (so n_cells x n_cells) 
        # but assumes a one-to-one correspondence between representation and location
        loc_confusion = noise * (self.env['dist'] == 1) \
            / np.reshape(np.sum(self.env['dist']==1,1),[self.env['dist'].shape[0],1]) \
                + (1 - noise) * np.eye(self.env['dist'].shape[0])
        return loc_confusion
    
    def get_mat_retrieve(self, confusion=None):
        # If confusion is not provided: set to identity so there is no confusion
        confusion = np.eye(self.pars['model']['n_cells']) \
            if confusion is None else confusion        
        # Create n_locs x n_cells retrieval matrix R, so that x_t = one-hot(l_t) * R
        # retrieves the cell representation for location l_t
        # Default: identity matrix, assuming correspondence of representation and location        
        mat_retrieve = np.eye(self.pars['model']['n_cells'])
        # Then add confusion in retrieval from noise. Confusion is defined between
        # representations so multiply on right side, to keep n_locs x n_cells matrix
        mat_retrieve = np.matmul(mat_retrieve, confusion)
        return mat_retrieve
    
    def get_mat_decode(self):
        # Create n_cells x n_locs decoding matrix D, so that l_t = argmax(x_t * D)
        # Default: identity matrix, assuming correspondence of representation and location
        mat_decode = np.eye(self.pars['model']['n_cells'])        
        return mat_decode
    
    def get_mat_encode(self):
        # Create n_locs x n_cells decoding matrix E, so that x_t = one-hot(l_t) * E
        # Default: identity matrix, assuming correspondence of representation and location
        mat_encode = np.eye(self.pars['model']['n_cells'])        
        return mat_encode           
    
    def get_mat_transition(self, confusion=None):
        # If confusion is not provided: set to identity so there is no confusion
        confusion = np.eye(self.pars['model']['n_cells']) \
            if confusion is None else confusion
        # Create n_cells x n_cells transition matrix T for each action. 
        # We'll work with row vectors, so that x_t+1 = x+t * T, 
        # where T_ij is the probability to go FROM i TO j
        mat_transition = np.zeros((self.env['env'].n_actions, 
                                   self.pars['model']['n_cells'], 
                                   self.pars['model']['n_cells']))
        # Run through all available locations and all available actions to fill T
        # This assumes one-to-one correspondence of representation and location
        for l_i, location in enumerate(self.env['env'].locations):
            for a_i, action in enumerate(location['actions']):
                # If action is not available: keep representation the same
                mat_transition[a_i, l_i, :] = action['transition'] \
                    if np.sum(action['transition']) > 0 \
                        else np.eye(self.env['env'].n_locations)[l_i]
        # Now introduce confusion into transition matrix: multiply noise-less
        # transitions by confusion to spread transition probability to close locations
        mat_transition = np.matmul(mat_transition, confusion)
        return mat_transition
    
    def get_mat_Q(self):
        # Create default n_cells x n_actions representation-action weights matrix
        mat_Q = np.zeros((self.pars['model']['n_cells'],self.env['env'].n_actions))
        return mat_Q
    
    def get_action(self, policy):
        # Get action from policy
        return self.get_sample(policy)
    
    def get_sample(self, probability):
        # Sample from array containing probability distribution
        return np.random.choice(np.arange(len(probability)), p=probability)
    
# OVC model: each represenation has a pre-learned optimal escape action
class OVC(Model):
    
    # Overwrite function for Q matrix for optimal policy
    def get_mat_Q(self):
        # Object vector cells have prelearned optimal policy. Get that optimal policy
        optimal_policy = self.env['env'].policy_optimal(
            self.env['env'].policy_distance(self.env['home']))
        # Create default n_cells x n_actions representation-action weights matrix
        mat_Q = np.zeros((self.pars['model']['n_cells'],self.env['env'].n_actions))        
        # This is ugly but easy: assume one-to-one correspondence between cells and locations
        # Then for each representation = location, we can fill out the optimal actions
        for l_i, location in enumerate(optimal_policy):
            for a_i, action in enumerate(location['actions']):
                mat_Q[l_i, a_i] = action['probability']
        return mat_Q
        
# Place model: retrieve represenation from memory, replay to get escape actions
class Place(Model):

    # Add replay parameter defaults to model parameters
    def init_defaults(self):
        # First create standard defaults from parent class
        pars_dict = Model.init_defaults(self)
        # Then add additional place parameters for model:
        # Temporal discounting for reward in TD learning
        pars_dict['model']['gamma'] = 0.7
        # Inverse temperature for policy in TD learning
        pars_dict['model']['beta'] = 1.5
        # Learning rate in TD learning
        pars_dict['model']['alpha'] = 0.8
        # And additional place parameters for simulation:
        # Number of replays
        pars_dict['sim']['replay_n'] = 5;
        # Replay interval
        pars_dict['sim']['replay_interval'] = 2
        # Set maximum replay length
        pars_dict['sim']['replay_max_steps'] = 15
        # Overwrite default for replay direction: always towards home
        pars_dict['sim']['replay_from_home'] = False                
        # Then return combined standard and place default parameter dictionary
        return pars_dict
    
    # Add retrieval matrix to synaptic weight matrices
    def init_model(self):
        # First create all standard synaptic weight matrices
        model_dict = Model.init_model(self)
        # Then add n_locs x n_cells retrieval matrix R, so that x_t = l_t * R
        # retrieves the cell representation for location l_t
        # Optionally: add confusion in retrieval from model_dict['mat_confuse']
        model_dict['mat_retrieve'] = self.get_mat_retrieve()
        # And return combined standard and place model dictionary
        return model_dict     
    
    # Explore step is like standard except for representation update: 
    # instead of transitioning representation, it will be retrieved
    def exp_transition(self, prev_step):
        # All other steps: get new location from environment transition of previous action
        prev_loc = prev_step['location']
        prev_action = prev_step['action']
        # Transition location, retrieve representation for new location
        curr_loc = self.loc_transition(prev_loc, prev_action)
        curr_rep = self.rep_retrieve(curr_loc)
        # And return both
        return curr_loc, curr_rep

    # Make escape transition identical to explore
    def esc_transition(self, prev_step):
        # Use the same transition as during explore
        return self.exp_transition(prev_step)
    
    # Replay should be identical to standard replay, but gets addtional TD weights update
    def replay_step(self, prev_step):
        # Do original replay step from previous step
        curr_step = Model.replay_step(self, prev_step)
        # Then TD learn updated Q-weights
        self.model['mat_Q'][:, prev_step['action']] = self.replay_weight_update(
            curr_step['location'], curr_step['representation'],
            prev_step['representation'], prev_step['action'])
        # And return all information about this replay step
        return curr_step    
    
    def replay_weight_update(self, curr_loc, curr_rep, prev_rep, prev_action):
        # TD-learn: if the animal thinks it's home, set reward
        reward = 1*(curr_loc == self.env['home'])
        # Calculate TD delta: r + gamma * max_a Q(curr_rep, a) - Q(prev_rep, prev_action)
        delta = reward + self.pars['model']['gamma'] * np.max(np.matmul(curr_rep, self.model['mat_Q'])) \
            - np.matmul(self.model['mat_Q'][:,prev_action], prev_rep)
        # Return Q weight update using TD rule: w = w + alpha * delta * dQ/dw
        return self.model['mat_Q'][:, prev_action] + self.pars['model']['alpha'] * delta * prev_rep         
    
# Landmark model: retrieve path-integrated representation from memory
# Do multiple inheritance to inherit ovc policy and place replay mechanisms
# Multiple inheritance searches for functions depth-first, left-to-right 
# There is also multi-level inheritance to inherit standard model functionality
class Landmark(Place, OVC):
    
    # Overwrite additional parameter defaults to add to model parameters
    def init_defaults(self):
        # First create standard defaults from parent class
        pars_dict = Model.init_defaults(self)
        # Additional landmark model parameters
        # Retrieval noise sigma
        pars_dict['model']['rep_transition_noise'] = {'ovc': 0, 'place': 0}
        # Inverse temperature for policy during replay
        pars_dict['model']['beta'] = 1.5
        # Weighting of path integration vs memory
        pars_dict['model']['path_int_weight'] = 0.25
        # Additional landmark parameters for simulation
        # Number of replays
        pars_dict['sim']['replay_n'] = 5;
        # Replay interval
        pars_dict['sim']['replay_interval'] = 2
        # Set maximum replay length
        pars_dict['sim']['replay_max_steps'] = 15   
        # Overwrite default for replay direction: always from home
        pars_dict['sim']['replay_from_home'] = True        
        # Then return combined standard and place default parameter dictionary
        return pars_dict
    
    # Add memory dictionary to model matrices
    def init_model(self):
        # First create all model matrices as in place model
        model_dict = Place.init_model(self)
        # Then add memories: empty list for each representation
        model_dict['memory'] = [[] for _ in range(self.pars['model']['n_cells'])]
        # And return combined standard and landmark model dictionary
        return model_dict
    
    # Initialise both ovc and place reoresentation at first step of exploration
    def exp_init(self):
        # Set initial location and representation for explore walk: home      
        curr_loc = self.env['home']
        curr_rep = {'ovc': self.rep_encode(curr_loc), 
                    'place': self.rep_encode(curr_loc)}
        return curr_loc, curr_rep        
    
    # Explore step is identical to standard model, but additionally make memory
    def exp_step(self, explore):
        # Run standard explore step
        curr_step = Model.exp_step(self, explore)
        # Then write the new memory: new ovc representation as value under place key
        self.rep_mem_write(np.argmax(curr_step['representation']['place']), 
                           curr_step['representation']['ovc'])
        # Return standard step without further changes
        return curr_step    
    
    # The transition function mostly stays the same, but the representation transition
    # now also need the new location for retrieval
    def exp_transition(self, prev_step):
        # Collect previous step
        prev_loc = prev_step['location']
        prev_action = prev_step['action']
        prev_rep = prev_step['representation']
        # Transition location and retrieve place representation for new location
        curr_loc = self.loc_transition(prev_loc, prev_action)
        curr_place = self.rep_retrieve(curr_loc)
        # Transition ovc representation by combining memory and path integration
        curr_rep = self.rep_combine(curr_place, prev_rep, prev_action)
        # And return both
        return curr_loc, curr_rep    
    
    # Make the escape transition identical to explore
    def esc_transition(self, prev_step):
        return self.exp_transition(prev_step)
    
    # Replay step is identical to standard replay, but additionally make memory
    def replay_step(self, prev_step):
        # Run standard replay step
        curr_step = Model.replay_step(self, prev_step)
        # Then write the new memory: new ovc representation as value under place key
        self.rep_mem_write(np.argmax(curr_step['representation']['place']), 
                           curr_step['representation']['ovc'])
        # Return standard step without further changes
        return curr_step    
    
    # The replay transition is very similar to the explore transition, except for
    # the place representation which transitions instead of being retrieved
    # I also ended up deciding against using memory to error-correct ovc path integration in replay
    def replay_transition(self, prev_step):
        # Collect previous step
        prev_loc = prev_step['location']
        prev_action = prev_step['action']
        prev_rep = prev_step['representation']
        # Transition location and retrieve place representation for new location
        curr_loc = self.loc_transition(prev_loc, prev_action)
        # Transition place representation: non-error-corrected path integration in memory key
        curr_place = self.rep_transition(prev_rep, prev_action, key='place')
        # Transition ovc representation by combining memory and path integration
        # curr_rep = self.rep_combine(curr_place, prev_rep, prev_action)
        # Actually, I think error-correcting replay could reinforce incorrect memories
        # It's better to just path-integrate ovcs independently during replay
        curr_ovc = self.rep_transition(prev_rep, prev_action, key='ovc')
        # Combine place and ovc representations into representation dictionary
        curr_rep = {'ovc': curr_ovc, 'place': curr_place}                
        # And return both
        return curr_loc, curr_rep
    
    def rep_combine(self, curr_place, prev_rep, prev_action):
        # Use place cell as key to read memory as probability distribution over ovcs
        mem_prob = self.rep_mem_read(np.argmax(curr_place))
        # Select appropriate transition matrix: ovc if transition matrix is a dict
        trans_mat = self.model['mat_trans']['ovc'] if isinstance(self.model['mat_trans'], dict) \
            else self.model['mat_trans']    
        # Get probability distribution over next locations from (noisy) transition        
        trans_prob = np.matmul(prev_rep['ovc'], trans_mat[prev_action])
        # Calculate combined probability distribution from memory and path integration
        ovc_prob = (self.pars['model']['path_int_weight'] * trans_prob + \
            (1 - self.pars['model']['path_int_weight']) * mem_prob) \
                if np.sum(mem_prob) > 0 else trans_prob
        # And sample from combined ovc probability
        curr_ovc = np.eye(trans_prob.shape[0])[self.get_sample(ovc_prob)]
        # Build current representation dictionary from ovc and place
        curr_rep = {'ovc': curr_ovc, 'place': curr_place}
        return curr_rep        
    
    def rep_mem_read(self, key):
        # Memory is list of values, where key acts as an index
        # Each value is a list of representations encountered at the key
        # Return the probability distribution over represenations at the key
        # If no memories exist yet for this key, return zeros for probability
        mems = np.stack([np.zeros(self.pars['model']['n_cells'])] 
                        + self.model['memory'][key])
        # Probability is simply normalised sum of memories
        prob = np.sum(mems, axis=0)/np.sum(mems) if np.sum(mems) > 0 else mems
        return prob
        
    def rep_mem_write(self, key, val):
        # Memory is list of values, where key acts as an index
        # Each value is a list of representations encountered at the key
        # This adds the provided representation (val) to the requested key
        self.model['memory'][key] += [val]
        
# Hybrid model: learn place action weights from ovcs
# Inherit Landmark model for memory (but store weights, not representations)
class Hybrid(Landmark):
    
    # Explore step is identical to landmark, but memory value is Q, not representation
    def exp_step(self, explore):
        # Run standard explore step
        curr_step = Model.exp_step(self, explore)
        # Then write the new memory: new Q value as value under place key
        self.rep_mem_write(np.argmax(curr_step['representation']['place']),
                           np.matmul(curr_step['representation']['ovc'], 
                                     self.model['mat_Q']))
        # Return standard step without further changes
        return curr_step
    
    # Like landmark model, transition ovc and retrieve place - but don't combine
    # representations here
    def exp_transition(self, prev_step):
        # Run standard  model transition step for ovc 
        curr_loc, curr_ovc = Model.exp_transition(self, prev_step)
        # Update place representation by retrieval
        curr_place = self.rep_retrieve(curr_loc)
        # The full representation contains ovc and place
        curr_rep = {'ovc': curr_ovc, 'place': curr_place}        
        # And return both new location and representation
        return curr_loc, curr_rep
    
    # Set escape transition identical to explore
    def esc_transition(self, prev_step):                   
        return self.exp_transition(prev_step)
    
    # Replay step is identical to standard replay, but additionally make memory
    def replay_step(self, prev_step):
        # Run standard replay step
        curr_step = Model.replay_step(self, prev_step)
        # Then write the new memory: new Q value as value under place key
        self.rep_mem_write(np.argmax(curr_step['representation']['place']),
                           np.matmul(curr_step['representation']['ovc'], 
                                     self.model['mat_Q']))
        # Return standard step without further changes
        return curr_step    
        
    # In the hybrid model, Q-vals are taken from memory
    def rep_policy(self, curr_rep, key='place'):
        # Get appropriate representation. Now place by default: that's the memory index
        curr_rep = self.rep_select(curr_rep, key)        
        # Get q-values from representation and q-weights
        curr_qs = self.rep_mem_read(np.argmax(curr_rep))
        # Then calculate policy from softmax over q-values
        curr_exp = np.exp(self.pars['model']['beta'] * curr_qs)
        return curr_exp / np.sum(curr_exp)                
    
    # Similarly, greedy policy need to retrieve Q-vals from memory    
    def rep_policy_greedy(self, curr_rep, key='place'):
        # Get appropriate representation. Now place by default: that's the memory index
        curr_rep = self.rep_select(curr_rep, key)        
        # Get q-values from representation and q-weights
        curr_qs = self.rep_mem_read(np.argmax(curr_rep))
        # Greedy policy: equal probability for only max q-values
        return (curr_qs == np.max(curr_qs)) / np.sum((curr_qs == np.max(curr_qs)))   

    # Decode needs to use place cells by default: ovcs are only there for policy
    def rep_decode(self, representation, key='place'):
        # Decode function is identical to standard decode, just with place as default
        return Model.rep_decode(self, representation, key)

    # Reading memory is now slightly different because it produces Qs not probabilities
    def rep_mem_read(self, key):
        # Memory is list of values, where key acts as an index
        # Each value is a list of q-values encountered at the key
        # Return the mean of all q-values at the key
        # If no memories exist yet for this key, return zeros for q-values
        mems = np.stack(self.model['memory'][key] if len(self.model['memory'][key]) > 0
                        else [np.zeros(self.env['env'].n_actions)])
        # Retrieved q-val is mean of all memories
        qs = np.mean(mems, axis=0)
        return qs