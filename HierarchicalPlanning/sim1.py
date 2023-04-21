#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:44:56 2021

@author: jbakermans
"""

import world, model, plot
import matplotlib.pyplot as plt
import os

# Create hierarchical environment with dummy walls (empty representation)
env = world.World([['./envs/loop_5.json'], 
                   ['./envs/3x5.json', './envs/5x3.json', './envs/4x4.json']], 
                  {'dummy': True})

# Create directory for trained model
curr_dir = './sim/exp1'
# Create model for optimal policy
m1 = model.Model(env)
# Train if trained model don't exist
if not os.path.exists(curr_dir):
    # Make directory for trained model
    os.makedirs(curr_dir)    
    # Train model for learned policy
    m2 = model.FitPolicy(env, {'DQN': {'save': curr_dir + '/m.pt',
                                   'load': None,
                                   'steps': 500, 
                                   'episodes': 10,
                                   'envs': 25}})    
else:        
    # Load model for learned policy
    m2 = model.FitPolicy(env, {'DQN': {'load': curr_dir + '/m.pt'}})

# Get number of rooms and reward room
n_rooms = env.env['n_locations']
reward_room = env.env['components']['reward']['locations'][0]
# Plot environment, learned policy, and high-level replay steps for Figure 7
plt.figure(figsize=(20,4))
ax = plt.subplot(1,5,1)
plot.plot_env(env.env, ax=ax, do_plot_actions=False)
ax = plt.subplot(1,5,2)
plot.plot_env({'locations': m2.pol}, ax=ax)
ax = plt.subplot(1,5,3)
plot.plot_env({'locations': m2.pol}, ax=ax, 
              do_plot_actions=[i in [reward_room] for i in range(n_rooms)])
ax = plt.subplot(1,5,4)
plot.plot_env({'locations': m2.pol}, ax=ax,
              do_plot_actions=[i in [reward_room, 
                                     (reward_room - 1) % n_rooms] 
                               for i in range(n_rooms)])
ax = plt.subplot(1,5,5)
plot.plot_env({'locations': m2.pol}, ax=ax,
              do_plot_actions=[i in [reward_room, 
                                     (reward_room - 1) % n_rooms, 
                                     (reward_room - 2) % n_rooms] 
                               for i in range(n_rooms)])
