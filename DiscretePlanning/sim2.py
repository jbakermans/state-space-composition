#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:05:33 2021

@author: jbakermans
"""

import model, world, plot
import numpy as np
import os
import matplotlib.pyplot as plt

def collect_data(steps, max_vis, fields):
    out = []
    step_vis = [[] for _ in range(max_vis)]
    for s in steps:
        if s['visits'] - 1 < max_vis:
            step_vis[s['visits'] - 1].append(s)
    for f in fields:
        accuracy = np.zeros(max_vis)
        for v_i, v in enumerate(step_vis):
            count = [s[f] for s in v]
            accuracy[v_i] = sum(count)/len(count) if len(count) > 0 else np.nan
            if v_i == (max_vis-1):
                print(len(count), 'steps with', v_i, 'visits')
        out.append(accuracy)
    return out

# Set number of environments to run
N = 25
# Set max number of visits to count
V = 15
# Keep track of numpy overflows
np.seterr(all='raise')

# 1. Explore after finding reward: latent vs non-latent learning
exp = 1
L = 4000
dat_1 = [np.zeros((N, V)), np.zeros((N, V))]
# Set directory to save results to
curr_dir = './sim/exp4'
# Run simulations if files don't exist yet
if not os.path.exists(curr_dir):
    os.makedirs(curr_dir)
    # Run experiments
    for i in range(N):
        # Create environment with walls 
        env = world.World('./envs/7x7.json', env_type={'name':'walls_n', 
                                                       'n_walls': 1,
                                                       'wall_length': 4,
                                                       'centre': True})
        # Create model with loaded weights
        m = model.Model(env, {'experiment': exp,
                              'test_max_steps': L,
                              'print': False})
        # Simulate one run of exploration after reward
        _, test = m.simulate()
        # Only include locations where partial representation policy is not optimal
        test = [t for t in test if not m.pars['part_rep_opt'][t['location']]]
        # Collect data from test, skip first step
        curr_data = collect_data(test[1:], V, ['full_rep_corr', 'curr_rep_corr'])
        for d, c_d in zip(dat_1, curr_data):
            d[i, :] = c_d
    # Save results            
    with open(curr_dir + '/dat.npy', "wb") as f:
        np.save(f, dat_1)
# Load results
print('Loading from ' + curr_dir)
dat_1 = np.load(curr_dir + '/dat.npy')

# 2. Replay after finding reward: memory vs bellman updates
exp = 2
R = 25
L = 100
dat_2 = [np.zeros((N, V)), np.zeros((N, V))]
# Set directory to save results to
curr_dir = './sim/exp5a'
# Run simulations if files don't exist yet
if not os.path.exists(curr_dir):
    os.makedirs(curr_dir)
    # Run experiments
    for i in range(N):
        # Create environment with walls 
        env = world.World('./envs/7x7.json', env_type={'name':'walls_n', 
                                                       'n_walls': 1,
                                                       'wall_length': 4,
                                                       'centre': True})
        # Create model with loaded weights
        m = model.Model(env, {'experiment': exp, 
                              'replay_n': R,
                              'replay_max_steps': L,
                              'replay_reverse': True,
                              'print': False})
        # Simulate a bunch of replays after reward
        _, test = m.simulate()
        # Collect data from test, skip first step
        curr_data = collect_data(model.flatten(test), V, ['replay_rep_corr', 'replay_bm_corr'])
        for d, c_d in zip(dat_2, curr_data):
            d[i, :] = c_d
    # Save results            
    with open(curr_dir + '/dat.npy', "wb") as f:
        np.save(f, dat_2)
# Load results
print('Loading from ' + curr_dir)
dat_2 = np.load(curr_dir + '/dat.npy')

# 3. Replay after finding reward: memory vs bellman updates - but now optimal reverse replay
exp = 2
R = 500
L = 15
dat_3 = [np.zeros((N, V)), np.zeros((N, V))]
# Set directory to save results to
curr_dir = './sim/exp5b'
# Run simulations if files don't exist yet
if not os.path.exists(curr_dir):
    os.makedirs(curr_dir)
    # Run experiments
    for i in range(N):
        # Create environment with walls 
        env = world.World('./envs/7x7.json', env_type={'name':'walls_n', 
                                                       'n_walls': 1,
                                                       'wall_length': 4,
                                                       'centre': True})
        # Create policy that moves exactly in opposite way from reward
        opposite_pol = env.get_policy(in_place=False)
        opposite_pol = env.policy_distance(
            opposite_pol, env.components['reward']['locations'], 
            adjacency=env.get_adjacency(opposite_pol), opposite=True, disable_if_worse=True)
        # Collect all replay states that are terminal: no opposite action available
        replay_terminal = []
        for l_i, location in enumerate(opposite_pol):
            if all([action['probability'] == 0 for action in location['actions']]):
                replay_terminal.append(l_i)
        # And only keep optimal actions
        opposite_pol = env.policy_optimal(opposite_pol)
        # Create model with loaded weights
        m = model.Model(env, {'experiment': exp, 
                              'replay_n': R,
                              'replay_max_steps': L,
                              'replay_pol': opposite_pol,
                              'replay_reverse': True,
                              'replay_terminal': replay_terminal,
                              'print': False})
        # Simulate a bunch of replays after reward
        _, test = m.simulate()    
        # Collect data from test, skip first step
        curr_data = collect_data(model.flatten(test), V, ['replay_rep_corr', 'replay_bm_corr'])
        for d, c_d in zip(dat_3, curr_data):
            d[i, :] = c_d
    # Save results            
    with open(curr_dir + '/dat.npy', "wb") as f:
        np.save(f, dat_3)
# Load results
print('Loading from ' + curr_dir)
dat_3 = np.load(curr_dir + '/dat.npy')

# Plot policy accuracy behind wall with and without latent learning for Figure 4c
plt.figure(figsize=(3, 3)); 
ax = plt.axes()
ax.errorbar(np.arange(1, V+1), np.mean(dat_1[0], axis=0), 
            yerr=np.std(dat_1[0], axis=0) / np.sqrt(dat_1[0].shape[0]),
            label='Learn representation \n before reward');
ax.errorbar(np.arange(1, V+1), np.mean(dat_1[1], axis=0), 
            yerr=np.std(dat_1[1], axis=0) / np.sqrt(dat_1[1].shape[0]),
            label='Learn representationa \n after reward');
ax.set_ylim([0, 1.05])
ax.set_yticks([0, 0.5, 1])
ax.set_xticks([0, 5, 10, 15])
#ax.set_xlabel('Visits')
#ax.set_ylabel('Accuracy')
#ax.set_title('a. Latent learning')
ax.legend()

# Plot policy accuracy for memory encoding vs value backup replay for Figure 5b
plt.figure(figsize=(6, 3)); 
# First subplot: replay along reverse optimal policy
ax.errorbar(np.arange(1, V+1), np.mean(dat_2[0], axis=0), 
            yerr=np.std(dat_2[0], axis=0) / np.sqrt(dat_2[0].shape[0]),
            label='Replay creates \n memories');
ax.errorbar(np.arange(1, V+1), np.mean(dat_2[1], axis=0), 
            yerr=np.std(dat_2[1], axis=0) / np.sqrt(dat_2[1].shape[0]),
            label='Replay does \n bellman backups');
ax.set_ylim([0, 1.05])
ax.set_yticks([0, 0.5, 1])
ax.set_xticks([0, 5, 10, 15])
#ax.set_xlabel('Visits')
#ax.set_ylabel('Accuracy')
#ax.set_title('b. Replay')
ax.legend()
# Second subplot: replay along random trajectories
ax = plt.subplot(1,2,2)
ax.errorbar(np.arange(1, V+1), np.mean(dat_3[0], axis=0), 
            yerr=np.std(dat_3[0], axis=0) / np.sqrt(dat_3[0].shape[0]),
            label='Replay creates \n memories');
ax.errorbar(np.arange(1, V+1), np.mean(dat_3[1], axis=0), 
            yerr=np.std(dat_3[1], axis=0) / np.sqrt(dat_3[1].shape[0]),
            linestyle=(0, (5,5)),
            label='Replay does \n bellman backups');
ax.set_ylim([0, 1.05])
ax.set_yticks([0, 0.5, 1])
ax.set_xticks([0, 5, 10, 15])
#ax.set_xlabel('Visits')
#ax.set_ylabel('Accuracy')
#ax.set_title('b. Replay')
ax.legend()

# Prepare environment for inset figures showing example replays in Figure 5b
env = world.World('./envs/7x7.json', env_type={'name':'walls_n', 
                                               'n_walls': 1,
                                               'wall_length': 4,
                                               'centre': True})
# Create policy that moves exactly in opposite way from reward
opposite_pol = env.get_policy(in_place=False)
opposite_pol = env.policy_distance(
    opposite_pol, env.components['reward']['locations'], 
    adjacency=env.get_adjacency(opposite_pol), opposite=True, disable_if_worse=True)
# Collect all replay states that are terminal: no opposite action available
replay_terminal = []
for l_i, location in enumerate(opposite_pol):
    if all([action['probability'] == 0 for action in location['actions']]):
        replay_terminal.append(l_i)
# And only keep optimal actions
opposite_pol = env.policy_optimal(opposite_pol)

# Create model for random walk, and model for reverse optimal walk
m1 = model.Model(env, {'experiment': 2, 
                      'replay_n': 1,
                      'replay_max_steps': 10,
                      'replay_reverse': True,
                      'print': False})
m2 = model.Model(env, {'experiment': 2, 
                      'replay_n': 1,
                      'replay_max_steps': 10,
                      'replay_pol': opposite_pol,
                      'replay_reverse': True,
                      'replay_terminal': replay_terminal,
                      'print': False})
# Simulate replay for both
_, test1 = m1.simulate()
_, test2 = m2.simulate()    

# Plot example replay for optimal reverse vs random replay for Figure 5b
plt.figure(figsize=(3,3))
ax = plt.subplot(2,1,1)
plot.plot_grid(env, do_plot_actions=False, ax=ax)
plot.plot_walk(env, [t['location'] for t in test1[0]], ax=ax)
ax = plt.subplot(2,1,2)
plot.plot_grid(env, do_plot_actions=False, ax=ax)
plot.plot_walk(env, [t['location'] for t in test2[0]], ax=ax)