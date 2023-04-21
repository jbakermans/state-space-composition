#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:24:36 2023

@author: jbakermans
"""

import world, plot
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy

# Create empty env to use for its functions
env = world.World(None)

def evaluate_replays(all_paths, replay_new, new_mems, add_mems, 
                     reward_locs, wall_locs, 
                     noisy_pol_before, noisy_pol_after, pol_random):
    # Get distances from all locations for each replay
    all_distances = []
    # Now evaluate each replay
    for rep_i, replay in enumerate(all_paths):
        # Either create memories of new representations, or add memories of existing representations
        if replay_new:
            # This replay can only create new memories after finding representation
            has_found_memory = np.cumsum([l in new_mems for l in replay]) > 0
            # Combine new memories before and after this replay
            curr_new_mems = list(set(
                new_mems + [l for l, m in zip(replay, has_found_memory) if m]))
            # The additional memories of existing representations remain the same
            curr_add_mems = add_mems
        else:
            # Combine existing and replayed additional memories of existing representations
            curr_add_mems = list(set(add_mems + replay))
            # The memories of new representations remain the same
            curr_new_mems = new_mems
        # Create current before and after transition policies, depending on mems
        curr_pol_before = [l_low_noise if l_i in curr_add_mems else l_high_noise
                           for l_i, (l_high_noise, l_low_noise) in enumerate(zip(*noisy_pol_before))]
        curr_pol_after = [l_low_noise if l_i in curr_add_mems else l_high_noise
                          for l_i, (l_high_noise, l_low_noise) in enumerate(zip(*noisy_pol_after))]
        # Create current transition matrices
        curr_T_before = np.array(get_transition_matrix(curr_pol_before))
        curr_T_after = np.array(get_transition_matrix(curr_pol_after))
        curr_T_random = np.array(get_transition_matrix(pol_random))
        # Now add memory locs as absorbing locations in the before and random matrix
        curr_T_before[curr_new_mems,:] = np.eye(len(curr_T_before))[curr_new_mems,:]
        curr_T_random[curr_new_mems,:] = np.eye(len(curr_T_random))[curr_new_mems,:] 
        # Calculate expected time and probability for getting absorbed at each location,
        # by following the full policy, partial policy, and random policy
        abs_trans = [get_absorbing_transient(T) for T in [curr_T_before, curr_T_after, curr_T_random]]
        p_abs = [expand_absorption_prob(T, absorption_prob(T))
                 for T in [curr_T_before, curr_T_after, curr_T_random]]
        T_abs = [expand_absorption_time(T, absorption_time(T))
                 for T in [curr_T_before, curr_T_after, curr_T_random]]
        # Now I have all the ingredients to calculate expected distance to reward 
        # 1. Immediately to reward following partial policy
        beforeToRewardProb = p_abs[0][:, reward_locs]
        beforeToRewardTime = T_abs[0][:, reward_locs]
        # 2. To memories following partial policy
        beforeToMemoryProb = p_abs[0][:, curr_new_mems]
        beforeToMemoryTime = T_abs[0][:, curr_new_mems]
        # 3. To other absorbing non-wall locations - agent gets stuck
        stuck_locs = [l for l in np.where(abs_trans[0][0])[0]
                      if l not in curr_new_mems + wall_locs + reward_locs]
        beforeToStuckProb = p_abs[0][:, stuck_locs]
        beforeToStuckTime = T_abs[0][:, stuck_locs]
        # 4. From getting stuck to reward, random policy
        stuckToRewardProb = p_abs[2][np.ix_(stuck_locs, reward_locs)]
        stuckToRewardTime = T_abs[2][np.ix_(stuck_locs, reward_locs)]
        # 5. From getting stuck to memory, random policy
        stuckToMemoryProb = p_abs[2][np.ix_(stuck_locs, curr_new_mems)]
        stuckToMemoryTime = T_abs[2][np.ix_(stuck_locs, curr_new_mems)]
        # 6. From memory to reward, optimal policy
        memoryToRewardProb = p_abs[1][np.ix_(curr_new_mems, reward_locs)]
        memoryToRewardTime = T_abs[1][np.ix_(curr_new_mems, reward_locs)]

        # Get paths that go directly to reward
        expected_distance_1 = np.zeros(len(pol_random))
        # Start with paths immediately to reward
        curr_locs = np.squeeze(beforeToRewardProb > 0)
        expected_distance_1[curr_locs] += np.squeeze(
            beforeToRewardProb[curr_locs] * beforeToRewardTime[curr_locs])
        
        # Add paths that go immediately to memory, then from memory to reward
        expected_distance_2 = np.zeros(len(pol_random))    
        for m_i in range(len(curr_new_mems)):
            # Get all locations that can end up at memory
            curr_locs = np.squeeze(beforeToMemoryProb[:, m_i] > 0)
            # First add distance towards this memory
            expected_distance_2[curr_locs] += \
                beforeToMemoryProb[curr_locs, m_i] * beforeToMemoryTime[curr_locs, m_i]
            # Then add distance from this memory to reward
            expected_distance_2[curr_locs] += \
                beforeToMemoryProb[curr_locs, m_i] * memoryToRewardProb[m_i] * memoryToRewardTime[m_i]
                
        # Add paths that go get stuck, then from stuck to reward or to memory to reward
        expected_distance_3 = np.zeros(len(pol_random))    
        for s_i in range(len(stuck_locs)):
            # Get all locations that can end up stuck
            curr_locs = np.squeeze(beforeToStuckProb[:, s_i] > 0)
            # First add distance before getting stuck
            expected_distance_3[curr_locs] += \
                beforeToStuckProb[curr_locs, s_i] * beforeToStuckTime[curr_locs, s_i]
            # Then add distance from getting stuck to reward
            expected_distance_3[curr_locs] += \
                beforeToStuckProb[curr_locs, s_i] * stuckToRewardProb[s_i] * stuckToRewardTime[s_i]            
            # Then add distance from getting stuck to memory, plus distance from those memories to reward
            stuck_mems = np.squeeze(stuckToMemoryProb[s_i, :] > 0)
            # First add distance from this stuck loc towards each memory
            expected_distance_3[curr_locs] += beforeToStuckProb[curr_locs, s_i] * sum(
                stuckToMemoryProb[s_i, stuck_mems]
                * (stuckToMemoryTime[s_i, stuck_mems]
                   + np.squeeze(memoryToRewardProb[stuck_mems]) 
                   * np.squeeze(memoryToRewardTime[stuck_mems])))
        # Print progress
        print('Finished path ' + str(rep_i))
        # And finally, the expected distance to reward is the expectation over all these!        
        all_distances.append(expected_distance_1 + expected_distance_2 + expected_distance_3) 
    return all_distances

def prepare_policies(env_before, env_after, noise, do_plot=False, baseline_noise=True):
    # Get optimal policy for both: to reward if present, else random
    if 'reward' in env_before.components:
        pol_before = env_before.policy_distance(
            env_before.get_policy(in_place=False), 
            env_before.components['reward']['locations'],
            disable_if_worse=True, optimal=True)
    else:
        pol_before = env_before.get_policy(in_place=False)
    if 'reward' in env_after.components:
        pol_after = env_after.policy_distance(
            env_after.get_policy(in_place=False),
            env_after.components['reward']['locations'],
            disable_if_worse=True, optimal=True,
            adjacency = env_after.get_adjacency(env_after.locations))
    else:
        pol_after = env_after.get_policy(in_place=False)
        
    # Now adjust the policy before: copy over the probabilities, but in the new env
    for l_b, l_a in zip(pol_before, env_after.locations):
        for a_b, a_a in zip(l_b['actions'], l_a['actions']):
            # Copy over transitions from new environment
            a_b['transition'] = a_a['transition']
            # Disable actions that are not available in new environment
            if a_a['probability'] == 0: a_b['probability'] = 0
        # Renormalise
        tot_prob = sum([a_b['probability'] for a_b in l_b['actions']])
        for a_b in l_b['actions']:
            if tot_prob == 0:
                # If there are no available actions: random policy, self transition
                a_b['probability'] = 1/len(l_b['actions'])
                a_b['transition'] = np.eye(env_before.n_locations)[l_b['id']]
            else:
                # If there are actions available: normalise
                a_b['probability'] = a_b['probability'] / tot_prob
    # Also set disabled actions in pol_after to self-transitions (make absorbing)
    for l in pol_after:
        # If there are no available actions: random policy, self transition
        if sum([a['probability'] for a in l['actions']]) == 0:
            for a in l['actions']:
                a['probability'] = 1/len(l['actions'])
                a['transition'] = np.eye(env_before.n_locations)[l['id']]
    
    # Make noisy policies: mix one location's policy with its neighbours
    # This reflects what happens when a path integration error occurs
    noisy_pol_before = [deepcopy(pol_before) for _ in noise]
    noisy_pol_after = [deepcopy(pol_after) for _ in noise]
    for orig, pol, p_noise in zip(
            [pol_before] * len(noise) + [pol_after] * len(noise),
            noisy_pol_before + noisy_pol_after, noise + noise):
        # For each location: mix action probability with probability of its neighbours
        for l_i, loc in enumerate(pol):
            # Don't change absorbing states
            if sum([a['probability']*a['transition'][l_i] for a in orig[l_i]['actions']]) < 1:            
                # Keep track of the policy at each neighbour, with optional baseline random policy
                neighbour_actions = [[1/env_after.n_actions for a in range(env_after.n_actions)]] \
                    if baseline_noise else []
                # Find neighbours of this loc in env_after
                for action in env_after.locations[l_i]['actions']:
                    # Find all states this action can lead to
                    for l_j in [l for l, p in enumerate(action['transition']) 
                                if p > 0 and l != l_i]:
                        # Get policy from this neighbour in original pol and append
                        neighbour_actions.append([a['probability'] 
                                                  for a in orig[l_j]['actions']])
                # Stack all neighbour actions together, then average - only if they exist
                if len(neighbour_actions) > 0:
                    neighbour_actions = np.mean(np.stack(neighbour_actions), axis=0)
                    # Then mix together based on noise
                    for action, p in zip(loc['actions'], neighbour_actions):
                        action['probability'] = (1 - p_noise) * action['probability'] + p_noise * p
    
    # One more policy variant: undo absorbing reward for replay
    wall_locs = [l for c in env_after.components.values() if c['type'] == 'wall' for l in c['locations']]
    replay_pol = env_after.get_policy(in_place=False)
    for l in env_after.components['reward']['locations']:
        for a_r, a_b in zip(replay_pol[l]['actions'], env_after.locations[l]['actions']):
            # Only copy over base transition & probability if actions don't lead into walls
            if sum([a_b['components']['base']['transition'][w] for w in wall_locs]) < 1:
                a_r['probability'] = a_b['components']['base']['probability']
                a_r['transition'] = a_b['components']['base']['transition']
        # Normalise actions
        env_after.normalise_policy(env_after.locations[l])
        
    # Plot resulting policies    
    if do_plot:
        # Get transition matrices for each policy in this experiment
        T_before = np.array(env_after.get_transition_matrix(pol_before))
        T_after = np.array(env_after.get_transition_matrix(pol_after))
        T_random = np.array(env_after.get_transition_matrix(env_after.locations))    
            
        # Find nr of rows and columns on grid graph
        rows = len(set([l['y'] for l in env_after.locations]))
        cols = len(set([l['x'] for l in env_after.locations]))        
        
        # Plot policies and transition matrices
        plt.figure(); 
        ax = plt.subplot(2,3,1);
        plot.plot_map(env_after, ax=ax, shape='square', radius=1/max(rows, cols), location_cm='Greys');
        plot.plot_actions(pol_after, ax=ax, radius=0.02, action_cm='Reds');
        ax.set_aspect('equal')
        plt.title('Full policy')
        ax = plt.subplot(2,3,2);
        plot.plot_map(env_after, ax=ax, shape='square', radius=1/max(rows, cols), location_cm='Greys');
        plot.plot_actions(pol_before, ax=ax, radius=0.02, action_cm='Reds');
        ax.set_aspect('equal')
        plt.title('Partial policy')
        ax = plt.subplot(2,3,3);
        plot.plot_map(env_after, ax=ax, shape='square', radius=1/max(rows, cols), location_cm='Greys');
        plot.plot_actions(env_after.locations, ax=ax, radius=0.02, action_cm='Reds');
        ax.set_aspect('equal')
        plt.title('Random policy')
        for i, T in enumerate([T_after, T_before, T_random]):
            plt.subplot(2,3,3+i+1)
            plt.imshow(T)
            
        # Plot noisy variants
        plt.figure()
        for i, (name, pol) in enumerate(zip(['full', 'partial'], [noisy_pol_after, noisy_pol_before])):
            for j, noisy_pol in enumerate(pol):
                ax = plt.subplot(2,2,i*2+j+1)
                plot.plot_map(env_after, ax=ax, shape='square', radius=1/7, location_cm='Greys');
                plot.plot_actions(noisy_pol, ax=ax, radius=0.02, action_cm='Reds');
                plt.title('Noise ' + str(noise[j]) + ' for ' + name + ' policy')  
    
    # Return all policies: noisy before, noisy after, random, replay
    return noisy_pol_before, noisy_pol_after, env_after.get_policy(in_place=False), replay_pol

def prepare_environments(env_before, env_after, reward_locs, wall_locs):
    # Make reward absorbing for both
    env_before.policy_absorbing_reward(env_before.locations, reward_locs)
    env_after.policy_absorbing_reward(env_after.locations, reward_locs)
    # And make wall states absorbing for both
    env_before.policy_absorbing_reward(env_before.locations, wall_locs)
    env_after.policy_absorbing_reward(env_after.locations, wall_locs)
    # Set impossible actions as transition-to-self
    env_before.policy_zero_to_self_transition(env_before.locations,
                                              zero_policy=True, change_policy=True)
    env_after.policy_zero_to_self_transition(env_after.locations,
                                             zero_policy=True, change_policy=True)
    # And only for plotting: make locations appear on square grid
    env_before.set_locs_on_grid()
    env_after.set_locs_on_grid()
    # Return updated env_before and env_after (not stricly necessary, they're updated in-place)
    return env_before, env_after

# Recursively build all replays from one location
def extend_path(pol, path, max_steps=4, all_paths=None):
    # If full collection of all paths is not specified: start empty
    if all_paths is None: all_paths = []
    # Stop extending if path has reached maximum length
    if len(path) == max_steps:
        all_paths.append(path)
    else: 
        # Start from last step in path, and search tree until max depth is reached
        for action in [a for a in pol[path[-1]]['actions'] if a ['probability'] > 0]:
            # Find all transitions for this action, and extend
            for loc in [l for l, t in enumerate(action['transition']) if t > 0]:
                extend_path(pol, path + [loc], max_steps, all_paths)
    # Return list of all paths
    return all_paths

def absorption_time(T):
    # I want the time for getting absorbed for *each particular absorbing state*,
    # not just *any* (that's what absorption_time_any does).
    # For that purpose, follow https://math.stackexchange.com/a/4600370
    # 1. Get probability of absorption in each location
    g = absorption_prob(T)
    # 2. Get modified transition matrix P_tilde for each absorbing state
    # p_ij = p(to j | from i, end in a)
    #   = p(end in a | from i, to j) * p(to j | from i) / p(end in a | from i)
    #   = g_j * T_ij / g_i
    # First find the transient locations in T that need to be updated
    _, transient = get_absorbing_transient(T)
    # Then make a list of g_j / g_i for all absorption probabilities
    prob_ratio = [np.zeros((g.shape[0], g.shape[0])) for _ in range(g.shape[1])]
    # Whenever g_j or g_i equals zero, the ratio should be 0
    for p_, g_ in zip(prob_ratio, g.transpose()):
        p_[np.ix_(g_>0, g_>0)] = g_[g_>0].reshape((1,-1)) / g_[g_>0].reshape((-1,1))
    # And copy over the original transition matrix to modify it
    T_mod = [T.copy() for _ in range(g.shape[-1])]
    # Then update transient location transitions separately for each absorbing state
    for T_a, g_a in zip(T_mod, prob_ratio):
        T_a[np.ix_(transient,transient)] *= g_a
    # 3. Collect absorption time in the modified transition matrix
    times = [absorption_time_any(T_a) for T_a in T_mod]
    # But set the times to inf for any locations that can't reach the state at all
    for t, g_ in zip(times, g.transpose()):
        t[g_==0] = np.inf
    # For output: stack times into columns, so shape is the same as probabilities
    return np.stack(times, axis=-1)

def absorption_time_any(T):
    # Calculate expected time until absorption in any absorbing location
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Expected_number_of_steps
    Q, R, N = get_transition_components(T)
    # Absorption time is given by summing rows of N
    return np.sum(N, axis=1)

def absorption_prob(T):
    # Calculate absorption probability for each absorbing location
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Absorbing_probabilities
    Q, R, N = get_transition_components(T)
    # Probabilities are given by B = N * R    
    return np.matmul(N, R)

def get_transition_components(T):
    # Find which locations are absorbing and which are transient
    absorbing, transient = get_absorbing_transient(T)
    # Find transitions from transient to transient states
    Q = T[np.ix_(transient, transient)]
    # Find transitions from transient to absorbing states
    R = T[np.ix_(transient, absorbing)]
    # Calculate fundamental matrix
    N = np.linalg.inv(np.eye(sum(transient)) - Q)
    # Return components
    return Q, R, N

def expand_absorption_prob(T, M):
    # M is an absorption probability matrix with shape transient x absorbing states
    absorbing, transient = get_absorbing_transient(T)
    # For indexing it would be better if it's always states by states
    # Each absorbing state will with probability 1 to itself
    # All other absorption probabilities are zero
    full_mat = np.zeros_like(T)
    full_mat[np.ix_(transient, absorbing)] = M
    full_mat[absorbing, absorbing] = 1
    return full_mat

def expand_absorption_time(T, M):
    # M is an absorption time matrix with shape transient x absorbing states
    absorbing, transient = get_absorbing_transient(T)
    # For indexing it would be better if it's always states by states
    # Each absorbing state will reach itself in 0 steps
    # All other absorption times are infinite
    full_mat = np.full(T.shape,np.inf)
    full_mat[np.ix_(transient, absorbing)] = M
    full_mat[absorbing, absorbing] = 0
    return full_mat

def get_absorbing_transient(T):
    return np.diag(T) == 1, np.diag(T) < 1

def get_transition_matrix(pol):
    return env.get_transition_matrix(pol)