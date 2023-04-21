#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:47 2021

@author: jbakermans
"""

import model, world, plot
import numpy as np
import os
import matplotlib.pyplot as plt

# Little helper function for loading new type of optimality data
def unpack(data):
    if data.ndim == 1:
        data = data.reshape(1,-1)
    arrived = np.logical_and(
        data > -1, 
        data < 100)
    failed = np.logical_and(
        data > -1, 
        data == 100)
    curr_d = [np.mean(curr_d[curr_a]) if any(curr_a) else 100
              for curr_d, curr_a in zip(data, arrived)]
    curr_o = [np.sum(curr_a) / (np.sum(curr_a) + np.sum(curr_f))
              for curr_a, curr_f in zip(arrived, failed)]
    return np.array(curr_d), np.array(curr_o)

# Environment 1: only 1 reward
env_1 = world.World('./envs/7x7.json', env_type={'name': 'reward'});
# Environment 2: 1 reward and 2 walls
env_2 = world.World('./envs/7x7.json', env_type={'name': 'walls_n', 'n_walls': 3});
# Collect envs
envs = [env_1, env_2]
env_names = ['Reward only', 'Walls and reward']

# Set number of environments to generate
N = 25
# Set values for control vs true representation
do_control_rep = [True, False]
rep_names = ['Traditional', 'Compositional']

# Collect all data in list of experiments
experiments = []
exp_names = ['c. Training, single env',
             'd. Training, many envs',
             'e. Test, many envs']

# 1. Learning in single env
curr_dir = './sim/exp1'
# Simulate if simulations don't exist
if not os.path.exists(curr_dir):
    os.makedirs(curr_dir)
    for e_i, env in enumerate(envs):
        for r_i, rep in enumerate(do_control_rep):
            for i in range(N):
                # Make new env like original env
                curr_env = env.get_full_copy()
                # Do model
                m = model.FitPolicy(
                    curr_env, {'rep_control': rep, 'use_fast_reps': True, 'print': False,
                          'DQN': {'envs': 1,
                                  'episodes': 1,
                                  'steps': 1000,
                                  'hidden_dim': [1000, 750, 500],
                                  'save': curr_dir + '/env' + str(e_i) 
                                  + '_rep' + str(r_i) + '_i' + str(i) + '.pt'}})
# Fill data array by loading files
data = np.zeros((N, len(envs), len(do_control_rep)))
for e_i in range(len(envs)):
    for r_i in range(len(do_control_rep)):
        for i in range(N):
            curr_dat = np.load(
                curr_dir + '/env' + str(e_i) 
                + '_rep' + str(r_i) 
                + '_i' + str(i) + '.npy')
            _, data[i, e_i, r_i] = unpack(curr_dat)
experiments = experiments + [data]

# 2. Learning in multi env
curr_dir = './sim/exp2'
if not os.path.exists(curr_dir):
    os.makedirs(curr_dir)
    for e_i, env in enumerate(envs):
        for r_i, rep in enumerate(do_control_rep):
            # Make new env like original env
            curr_env = env.get_full_copy()
            m = model.FitPolicy(
                curr_env, {'rep_control': rep, 'use_fast_reps': True, 'print': False,
                      'DQN': {'envs': N,
                              'episodes': 100,
                              'steps': 200,
                              'hidden_dim': [1000, 750, 500],
                              'save': curr_dir + '/env' + str(e_i) 
                              + '_rep' + str(r_i) + '.pt'}})
# Fill data array by loading files
data = np.zeros((N, len(envs), len(do_control_rep)))
for e_i in range(len(envs)):
    for r_i in range(len(do_control_rep)):
        curr_dat = np.load(
            curr_dir + '/env' + str(e_i) 
            + '_rep' + str(r_i) + '.npy')
        _, data[:, e_i, r_i] = unpack(curr_dat)
experiments = experiments + [data]                

# 3. Testing on new env
curr_dir = './sim/exp3'
if not os.path.exists(curr_dir):
    os.makedirs(curr_dir)
    for e_i, env in enumerate(envs):
        for r_i, rep in enumerate(do_control_rep):
            for i in range(N):
                # Make new env like original env
                curr_env = env.get_full_copy()
                # Find number of invalid locations: those that cannot reach reward
                invalid_n = sum([sum([a['probability'] 
                                      for a in l['actions']])==0 
                                 for l in curr_env.locations])
                # Find number of locations that are part of walls
                wall_n = sum([any([l in c['locations'] if c['type']=='wall' else False 
                                   for c in curr_env.components.values()]) 
                              for l in range(curr_env.n_locations)])
                # Sometimes walls close of parts of environments, and only a subset
                # of locations can actually reach reward. Resample if that's the case
                while invalid_n > wall_n:
                    curr_env = env.get_full_copy()
                    invalid_n = sum([sum([a['probability'] 
                                          for a in l['actions']])==0 
                                     for l in curr_env.locations])
                    wall_n = sum([any([l in c['locations'] if c['type']=='wall' else False 
                                       for c in curr_env.components.values()]) 
                                  for l in range(curr_env.n_locations)])
                m = model.FitPolicy(
                    curr_env, {'rep_control': rep, 'use_fast_reps': True, 'print': False,
                          'DQN': {'envs': N,
                                  'episodes': 1,
                                  'hidden_dim': [1000, 750, 500],
                                  'load': './sim/exp2' + '/env' + str(e_i)
                                  + '_rep' + str(r_i) + '.pt'}})
                with open(curr_dir + '/env' + str(e_i) 
                          + '_rep' + str(r_i) 
                          + '_i' + str(i) + '.npy', "wb") as f:
                    np.save(f, m.learn_evaluate_policy(m.pol))
# Fill data array by loading files
data = np.zeros((N, len(envs), len(do_control_rep)))
for e_i in range(len(envs)):
    for r_i in range(len(do_control_rep)):
        for i in range(N):
            curr_dat = np.load(
                curr_dir + '/env' + str(e_i) 
                + '_rep' + str(r_i) 
                + '_i' + str(i) + '.npy')
            _, data[i, e_i, r_i] = unpack(curr_dat)
experiments = experiments + [data]
        
# Set plotting colours
red = np.array([255,77,77])*1/255
blue = np.array([77, 77, 214])*1/255

# Plot two example environments with optimal actions for Figure 3b
plt.figure(figsize=(3,4))
for env_i, env in enumerate(envs):
    # Make environment
    curr_env = env.get_full_copy()
    # Load model
    m = model.Model(curr_env)
    # Get optimal policy
    opt_pol = m.get_policy_optimal()
    # Plot map and optimal policy
    ax = plt.subplot(len(env_names), 1, env_i + 1)
    plot.plot_grid(curr_env, opt_pol, ax=ax)

# Plot policy accuracy for different representations for Figure 3b
plt.figure(figsize=(3,4))
curr_exp_names = [exp_names[0], exp_names[2]]
for env_i, env in enumerate(env_names):
    for exp_i, exp in enumerate([0, 2]):
        ax = plt.subplot(len(env_names), len(curr_exp_names),
                          env_i * len(curr_exp_names) + exp_i + 1)
        ax.bar(np.arange(len(rep_names)), 
                [np.mean(experiments[exp][:, env_i, r_i]) 
                for r_i in range(len(rep_names))], 
                yerr=[np.std(experiments[exp][:, env_i, r_i]) 
                      / np.sqrt(len(experiments[exp][:, env_i, r_i]))
                      for r_i in range(len(rep_names))], 
                align='center', zorder=0, color=[red, blue])
        ax.set_ylim([0,1])
        ax.set_xticks([])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1])

# Plot a bunch of environment layouts to illustrate training in Figure 3b
plt.figure(figsize=(5,2))
for i in range(5):
    ax = plt.subplot(2,5,i+1)
    env = world.World('./envs/7x7.json', env_type={'name': 'reward'});
    plot.plot_grid(env, ax=ax, do_plot_actions=False)
    ax = plt.subplot(2,5,i+6)
    env = world.World('./envs/7x7.json', env_type={'name': 'walls_n', 'n_walls': 3});
    plot.plot_grid(env, ax=ax, do_plot_actions=False)


