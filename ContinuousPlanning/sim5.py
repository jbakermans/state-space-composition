#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:28:15 2023

@author: jbakermans
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:27:55 2023

@author: jbakermans
"""

import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

def get_location_rates(locations, mvn_mean, mvn_cov):
    # location is numpy array of locations that we the firing rate for
    # mvn_mean is numpy array of rate map peaks for the current cell
    # Convert locations to n_locs x 1 x 2 array
    locs = locations.reshape([-1, 1, 2])
    # Convert firing rate peaks to 1 x cells x 2 array
    means = mvn_mean.reshape([1, -1, 2])
    # Calculate array of position differences as input for mvn
    diffs = (locs - means).reshape([-1, 2])
    # Evaluate multivariate normal with peak rate at 1 for each
    rates = multivariate_normal.pdf(diffs, mean=[0, 0], cov=np.eye(2)*mvn_cov) \
        * (np.sqrt(2 * np.pi) * mvn_cov)
    # Then cast rates back into original shape
    rates = rates.reshape([locations.shape[0], mvn_mean.shape[0]])
    # Return the total rate, summed across peaks, in each location
    return np.sum(rates, -1)

def get_grid_locations(n, angle=np.pi/180*6, offset=0):
    # I'll do this the lazy way: generate a big grid, then cut out piece
    # Start from a square grid that extends the [0, 1] arena on all sides
    locs = np.stack(np.meshgrid(np.linspace(-1,2,int(np.ceil(n*3))), 
                                np.linspace(-1,2,int(np.ceil(n*3)))), axis=-1).reshape([-1,2])
    # Shear locations to get hexagonal grid
    locs = np.matmul(locs, np.array([[1, 0.5],[0, 0.5*np.sqrt(3)]]).transpose())
    # Rotate by angle: subtract mean, rotate, re-add mean
    locs = np.matmul(locs - 0.5, np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).transpose()) + 0.5
    # Shift by offset in both directions, but cap at 1 radius
    locs = locs + (offset % (1/n))
    # Chop off any locations that are more than 1 radius outside of arena
    return locs[(locs[:,0] > -1/n) & (locs[:,0] < 1 + 1/n) & 
                (locs[:,1] > -1/n) & (locs[:,1] < 1 + 1/n)]

def plot_ratemap(data, ax, obj_loc=None, max_val=None):
    ax.imshow(data, extent=[0,1,0,1], origin='lower', vmax=max_val)
    ax.set_xticks([])
    ax.set_yticks([])
    if obj_loc is not None:
        for o in obj_loc:
            ax.add_patch(plt.Circle(o, radius=0.075, fc=[1,1,1], ec=[0,0,0]))    

# Set approximate nr of grid fields
N_fields = 3
# Set sampling rate for spatial maps
N_samples = 50
# Create sample locations
sample_locs = np.stack(np.meshgrid(np.linspace(0,1,N_samples), 
                                   np.linspace(0,1,N_samples)), 
                       axis=-1).reshape([-1,2])
# Calculate field sizes
field_size = 0.02*1/N_fields

# First figure: TEM replication of non-random place cell remapping
N_grids = 5
N_sens = 4
# Set random seed for reproducibility
# Actual correlations obviously will depend on this,
# because of the low nr of cells and environments 
# (to make it comparable to the situation in TEM)
np.random.seed(0)
# Set offsets of grid cells in first environment
grid_offsets = [np.random.rand(2) * 1/N_fields for cell in range(N_grids)]
# Grids keep correlation structure, so shift together in second environment
grid_offsets = [grid_offsets, [c + np.array([0.01, -0.02]) for c in grid_offsets]]
# Set locations for sensory observations
sens_locations = [[np.random.rand(2) for cell in range(N_sens)] for env in range(2)]
# Create all ratemaps
grids = [[get_location_rates(sample_locs, 
                            get_grid_locations(N_fields, offset=o), 
                            field_size).reshape([N_samples, N_samples])
         for o in offsets] for offsets in grid_offsets]
senses = [[get_location_rates(sample_locs, 
                           l.reshape([1, -1]), 
                           field_size).reshape([N_samples, N_samples])
         for l in locations] for locations in sens_locations]
hpcs = [[[g * s for s in env_sens] for g in env_grid] 
        for env_sens, env_grid in zip(grids, senses)]
# Finally: calculate TEM plot. Correlate grid value at hpc peak in both envs,
# for all pairs of hpc-grid combinations
grid_at_peak = []
for env_grid, env_hpc in zip(grids, hpcs):
    env_grid_at_peak = []
    for g in env_grid:
        for p in [v for r in env_hpc for v in r]:
            # Find peak in hpc map
            peak_loc = np.unravel_index(np.argmax(p, axis=None), p.shape)
            # Find value in grid map
            env_grid_at_peak.append(g[peak_loc])
    grid_at_peak.append(env_grid_at_peak)        
# Scale arbitrary axes
grid_at_peak = np.array(grid_at_peak)
grid_at_peak = grid_at_peak/np.max(grid_at_peak)
# Find ec firing rate at location of hpc peak for each pair of ec-hpc cells,
# and plot this value for env1 on the x-axis and env2 on the y-axis for Figure 2d
plt.figure(figsize=(3,3))
plt.scatter(*grid_at_peak, s=10)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
plt.title('Correlation: ' + '{0:.2f}'.format(np.corrcoef(*grid_at_peak)[0,1]))
plt.xlabel('Grid value at place peak, env 1')
plt.ylabel('Grid value at place peak, env 2')

# Second figure: landmark cells that respond to subset of objects in environment
# Repeat for a few different grid sizes
N_fields = 2
# Adjust field size to field number
field_size = 0.02*1/N_fields
# Set number of cells
N_grids = 3
N_ovc = 3
# Set offsets of grid cells
grid_offsets = np.array([[0.05, 0.15], [0.1, 0], [0.2, -0.3]])
grid_angles = np.array([0, np.pi/180*-25, np.pi/180*-25])
grid_number = np.array([1.75, 2, 2.5])
# Set object locaitons
obj_loc = np.array([[0.2, 0.75], [0.65, 0.8], [0.2, 0.25], [0.7, 0.2]])
# Set ovc offsets
ovc_offsets = np.array([[0.25, -0.05], [0.2, -0.2], [0.02, -0.2]])
# Create all ratemaps
grids = [get_location_rates(sample_locs, 
                            get_grid_locations(n, offset=o, angle=a), 
                            field_size).reshape([N_samples, N_samples])
         for o, a, n in zip(grid_offsets, grid_angles, grid_number)]
ovcs = [get_location_rates(sample_locs, 
                           (obj_loc + o).reshape([-1, 2]), 
                           field_size*0.6).reshape([N_samples, N_samples])
         for o in ovc_offsets]
hpcs = [[g * o for o in ovcs] for g in grids]
# Plot all ovc-grid conjunctions for Figure 2f; included examples on the diagonal
plt.figure(figsize=(5,5))
for row in range(N_grids + 1):
    for col in range(N_ovc + 1):
        # Create axes, except for top left
        if row > 0 or col > 0:
            ax = plt.subplot(N_grids + 1, N_ovc + 1, row * (N_ovc + 1) + col + 1)
        # First row: plot each ovc in one col
        if row == 0 and col > 0:
            plot_ratemap(ovcs[col-1], ax, obj_loc = obj_loc)
        # First col: plot each ovc in one row
        if row > 0 and col == 0:
            plot_ratemap(grids[row-1], ax, obj_loc = obj_loc)
        # All others: plot hpc
        if row > 0 and col > 0:
            plot_ratemap(hpcs[row-1][col-1], ax, obj_loc = obj_loc)
plt.suptitle('Hippocampal conjunctions') 
