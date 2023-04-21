#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:15:34 2021

@author: jbakermans
"""


import model, world, plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Plot heatmap with optimal policy from reward only for inset in Figure 4c
env_wall = world.World(env_type={'name':'wall'})
env_reward = world.World(
    env_type={'name':'reward',
              'locations': env_wall.components['reward']['locations']})
# Create model for the environment without wall
m = model.Model(env_reward)
# Get policy without wall
locs, dirs = env_reward.get_policy(n_x=8, n_y=8);
# Find for each location if policy is right
dist_locs = env_wall.get_grid_locs(n_x=20, n_y=20)
dist = [0 if m.get_optimal_dist(loc, env_wall, pol_env=env_reward)[1] > -1 else 1
        for loc in dist_locs]
# Create figure and axes
f = plt.figure(figsize=(2,2))
ax = plt.axes()
# Create colour map for plotting
cm = ListedColormap(np.array([[1,1,1],[1, 0.7, 0.7]]))
# Plot data
ax.imshow(np.array(dist).reshape(20, 20), 
          extent=[env_wall.xlim[0], env_wall.xlim[1], env_wall.xlim[0], env_wall.ylim[1]],
          origin='lower', vmin=0, vmax=1, cmap=cm)
# Plot policy
plot.plot_policy(env_reward, locs, dirs, ax=ax, big_arrow_head=False); 
# Plot map
plot.plot_map(env_wall, black_walls=True, ax=ax)