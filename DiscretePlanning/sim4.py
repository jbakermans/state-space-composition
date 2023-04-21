#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:49:25 2022

@author: jbakermans
"""

import world, plot
from matplotlib import pyplot as plt
import numpy as np
import optimal

# Setup some environment parameters: the walls, rewards, and memories
wall_locs = [3*7+1 + i for i in range(5)]
reward_locs = [5*7+3]
new_mems = [3*7+6]
add_mems = []
p_noise = [0.2, 0.1]
n_replays = 3

# Create environment before wall discovery
env_before = world.World('./envs/7x7.json')
# Add the reward
env_before.init_shiny('reward', shiny_in={'locations':reward_locs})

# Create environment after wall discovery
env_after = world.World('./envs/7x7.json')
# Add the reward
env_after.init_shiny('reward', shiny_in={'locations':reward_locs})
# Add the wall
env_after.init_wall('obstacle', wall_in={'locations':wall_locs})

# Setup the environments for optimal replay analysis
env_before, env_after = optimal.prepare_environments(
    env_before, env_after, reward_locs, wall_locs)

# Setup the policies 
pol_before, pol_after, pol_random, pol_replay = optimal.prepare_policies(
    env_before, env_after, p_noise, do_plot=False)        

# Get distances to reward by following optimal policy as a baseline
opt_dist = env_after.get_distance(env_after.get_adjacency(env_after.locations))[:,reward_locs]
# Set distances at wall to 0 and squeeze
opt_dist[~np.isfinite(opt_dist)] = 0
opt_dist = opt_dist.squeeze()

# Get all replay paths of length 5 from that location
all_paths = optimal.extend_path(pol_replay, new_mems, max_steps=5)

# Replay multiple times, and keep the best one each time
for curr_replay in range(n_replays):
    # Calculate expected distance from each location
    all_distances = []
    for replay_new in [True, False]:
        # Now evaluate all replay paths
        all_distances += optimal.evaluate_replays(all_paths, replay_new, new_mems, add_mems, 
                                                  reward_locs, wall_locs, 
                                                  pol_before, pol_after, pol_random)
    
    # Calculate the best replays: lowest total expected additional distance
    total_expected_additional_distance = np.sum(np.stack(all_distances) - opt_dist, axis=1)
    order = np.argsort(total_expected_additional_distance)
    
    # Use the mode of replay that produced the best replay
    replay_new = order[0] > len(all_paths)
    # And evaluate what happens without any replay
    no_replay = optimal.evaluate_replays([[]], replay_new, new_mems, add_mems, 
                                             reward_locs, wall_locs, 
                                             pol_before, pol_after, pol_random)[0]
        
    # Plot the currently best replay, and the distances without replay, for Figure 6c
    plt.figure(figsize=(3,2))
    ax = plt.subplot(1,2,1)
    plot.plot_grid(env_after, values=(no_replay - opt_dist), ax=ax, val_cm='Blues',
                  min_val=0, max_val=max((no_replay - opt_dist)), do_plot_actions=False)
    plt.title('Before replay')        
    ax = plt.subplot(1,2,2)
    plot.plot_grid(env_after, values=(all_distances[order[0]] - opt_dist), ax=ax, val_cm='Blues',
                  min_val=0, max_val=max((no_replay - opt_dist)), do_plot_actions=False)
    plot.plot_walk(env_after, all_paths[order[0] % len(all_paths)], ax=ax, jitter=0)
    plt.title('After replay')

    # Then keep the best replay
    if order[0] < len(all_paths):
        new_mems = list(set(new_mems + all_paths[order[0] % len(all_paths)]))
    else:
        add_mems = list(set(add_mems + all_paths[order[0] % len(all_paths)]))