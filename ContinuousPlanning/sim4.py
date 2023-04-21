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
    locs = np.stack(np.meshgrid(np.linspace(-1,2,n*3), 
                                np.linspace(-1,2,n*3)), axis=-1).reshape([-1,2])
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
        ax.add_patch(plt.Circle(obj_loc, radius=0.1, fc=[1,1,1], ec=[0,0,0]))    

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

# Set number of grid cells
N_grids = 5
# Set number of object vector cells
N_ovc = 4
# Set offsets of grid cells
grid_offsets = [np.random.rand(2) * 1/N_fields for _ in range(N_grids)]
# Actually, let's set them to fixed values for reproducibility
grid_offsets = [np.array([0.12,0.1]), 
                np.array([0.2,0.2]), 
                np.array([0.1,0.02]), 
                np.array([0.22, 0.02]),
                np.array([-0.08,0.12])]
# Set offsets of ovcs
ovc_offsets = [(np.random.rand(2) - 0.5) * 2/N_fields for _ in range(N_ovc)]
# Again, set to fixed values
ovc_offsets = [np.array([0.3,0]),
               np.array([0.15,-0.25]),
               np.array([-0.3,0.04]),
               np.array([-0.05,-0.15])]
# Set object location
obj_loc = np.array([0.7, 0.75])

# Create rate maps for both
grids = [get_location_rates(sample_locs, 
                            get_grid_locations(N_fields, offset=o), 
                            field_size).reshape([N_samples, N_samples])
         for o in grid_offsets]
ovcs = [get_location_rates(sample_locs, 
                           (obj_loc + o).reshape([1, -1]), 
                           field_size).reshape([N_samples, N_samples])
         for o in ovc_offsets]
# Calculate all hippocampal cells: conjunction
hpcs = [[g * o for o in ovcs] for g in grids]

# Calculate max firing rate for plotting ratemaps with same scale
max_rate = max([np.max(h) for h in hpcs])
    
# Plot a grid and ovc population and their conjunctions for environment 1 in Figure 2c,e
plt.figure()
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
            plot_ratemap(hpcs[row-1][col-1], ax, obj_loc = obj_loc, max_val=max_rate)
plt.suptitle('Hippocampal conjunctions, environment 1')            
            
# Then create second environment: move grids
grid_shift = np.array([0.05, 0.08])
grid_offsets = [g + grid_shift for g in grid_offsets]
grids2 = [get_location_rates(sample_locs, 
                             get_grid_locations(N_fields, offset=o), 
                             field_size).reshape([N_samples, N_samples])
          for o in grid_offsets]

# And change object location
obj_loc = np.array([0.3, 0.7])
ovcs2 = [get_location_rates(sample_locs, 
                            (obj_loc + o).reshape([1, -1]),
                            field_size).reshape([N_samples, N_samples])
         for o in ovc_offsets]

# Recalculate hippocampus
hpcs2 = [[g * o for o in ovcs2] for g in grids2]

# Calculate max firing rate for plotting ratemaps with same scale
max_rate = max([np.max(h) for h in hpcs2])

# Plot a grid and ovc population and their conjunctions for environment 2 in Figure 2c,e
plt.figure()
for row in range(N_grids + 1):
    for col in range(N_ovc + 1):
        # Create axes, except for top left
        if row > 0 or col > 0:
            ax = plt.subplot(N_grids + 1, N_ovc + 1, row * (N_ovc + 1) + col + 1)
        # First row: plot each ovc in one col
        if row == 0 and col > 0:
            plot_ratemap(ovcs2[col-1], ax, obj_loc = obj_loc)
        # First col: plot each ovc in one row
        if row > 0 and col == 0:
            plot_ratemap(grids2[row-1], ax, obj_loc = obj_loc)
        # All others: plot hpc
        if row > 0 and col > 0:
            plot_ratemap(hpcs2[row-1][col-1], ax, obj_loc = obj_loc, max_val=max_rate)
plt.suptitle('Hippocampal conjunctions, environment 2')                        