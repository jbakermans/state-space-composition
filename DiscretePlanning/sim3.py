#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:29:11 2023

@author: jbakermans
"""

import world, plot
from matplotlib import pyplot as plt
import numpy as np
import optimal

# Optionally multiply by SR
do_SR = False

# Linear track
line_reward_locs = [9]
line_wall_locs = []
line_new_mems = [9]
line_add_mems = []
# Create environment before reward discovery
line_before = world.World('./envs/1x10.json')
# Create environment after reward discovery
line_after = world.World('./envs/1x10.json')
# Add the reward
line_after.init_shiny('reward', shiny_in={'locations':line_reward_locs})

# Rectangular maze
maze_reward_locs = [8]
maze_wall_locs = [[9*1 + 2 + i*9 for i in range(3)],
                  [9*0 + 7 + i*9 for i in range(3)],
                  [9*4 + 5]]
maze_new_mems = [8]
maze_add_mems = []
# Create environment before wall discovery
maze_before = world.World('./envs/6x9.json')
# Add the walls
for i, w in enumerate(maze_wall_locs):
    maze_before.init_wall_fast('obstacle' + str(i), wall_in={'locations': w})
# Create environment after wall discovery
maze_after = world.World('./envs/6x9.json')
# Add the walls
for i, w in enumerate(maze_wall_locs):
    maze_after.init_wall_fast('obstacle' + str(i), wall_in={'locations': w})
# Add the reward
maze_after.init_shiny('reward', shiny_in={'locations':maze_reward_locs})
    
# Noise is irrelevant here but set anyway
p_noise = [0.2, 0.1]

# Now collect both environments in a list
all_env_before = [line_before, maze_before]
all_env_after = [line_after, maze_after]
all_reward_locs = [line_reward_locs, maze_reward_locs]
all_wall_locs = [line_wall_locs, [l for w in maze_wall_locs for l in w]]
all_start_locs = [[0], [18]]
all_new_mems = [line_new_mems, maze_new_mems]
all_add_mems = [line_add_mems, maze_add_mems]

# Simulate for both the linear and maze environment
for env_before, env_after, reward_locs, wall_locs, start_locs, new_mems, add_mems in zip(
        all_env_before, all_env_after, 
        all_reward_locs, all_wall_locs, all_start_locs,
        all_new_mems, all_add_mems):
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
    
    # Simulate *early*, when there are no reward memories yet,
    # and *late*, when reward memories exist everywhere
    for replay_new, start_loc in zip([True, False], [new_mems, start_locs]):
        # If required: calculate SR for current start and policy
        if do_SR:
            if replay_new:
                # For replaying new reps: use the before policy (but make rewards non-absorbing)
                SR = np.sum(env_after.get_successor_representation(
                    [l_r if l_i in reward_locs else l_b
                     for l_i, (l_b, l_r) in enumerate(zip(pol_before[0], pol_replay))])[start_loc], axis=0)
            else:
                # For replaying existing reps: use the after policy
                SR = np.sum(env_after.get_successor_representation(pol_after[0])[start_loc], axis=0)
        else: 
            # If you want to ignore SR, just make it all ones (so later multiplication won't matter)
            SR = np.ones(env_after.n_locations)
            
        # Get all replay paths of length 5 from that location
        all_paths = optimal.extend_path(pol_replay, start_loc, max_steps=5)
        # Scenario where you have created all new memories
        if not replay_new:
             new_mems = [l for l in range(env_after.n_locations) if l not in wall_locs]
        
        # Now evaluate all replay paths
        all_distances = optimal.evaluate_replays(all_paths, replay_new, new_mems, add_mems, 
                                                 reward_locs, wall_locs, 
                                                 pol_before, pol_after, pol_random)
        
        # And evaluate what happens without any replay
        no_replay = optimal.evaluate_replays([[]], replay_new, new_mems, add_mems, 
                                                 reward_locs, wall_locs, 
                                                 pol_before, pol_after, pol_random)[0]
        
        # Calculate the best replays: lowest total expected additional distance
        # Calculate additional distance, multiply by SR, then sum        
        total_expected_additional_distance = np.sum((np.stack(all_distances) - opt_dist) * SR, axis=1)
        order = np.argsort(total_expected_additional_distance)        

        # Plot the best replay, and the distances without replay, for Figure 6a,b
        plt.figure(figsize=(3,2))
        ax = plt.subplot(1,2,1)
        plot.plot_grid(env_after, values=(no_replay - opt_dist)*SR, ax=ax, val_cm='Blues',
                      min_val=0, max_val=max((no_replay - opt_dist)*SR), do_plot_actions=False)
        plt.title('Before replay')        
        ax = plt.subplot(1,2,2)
        plot.plot_grid(env_after, values=(all_distances[order[0]] - opt_dist)*SR, ax=ax, val_cm='Blues',
                      min_val=0, max_val=max((no_replay - opt_dist)*SR), do_plot_actions=False)
        plot.plot_walk(env_after, all_paths[order[0]], ax=ax, jitter=0)
        plt.title('After replay')
