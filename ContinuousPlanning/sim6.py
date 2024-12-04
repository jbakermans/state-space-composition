#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:34:34 2024

@author: jacob
"""

import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import matplotlib.patches as patches
import scipy.ndimage
from scipy.io import savemat

# To make fonts match between matlab and python exported figs
fig_scale = 1 # Was 0.8

# To recover original matplotlib params:
#import matplotlib
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)

np.random.seed(0)

fig_path = '/Users/jbakermans/Google Drive/DPhil/Documents/OVC/Figures/Ingredients/'
do_save = True

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
        if np.ndim(obj_loc) == 1:
            obj_loc = obj_loc.reshape((1,-1))
        for o in obj_loc:
            ax.add_patch(plt.Circle(o, radius=0.075, fc=[1,1,1], ec=[0,0,0]))    

# Set approximate nr of grid fields
N_fields = 2
# Set sampling rate for spatial maps
N_samples = 50
# Create sample locations
sample_locs = np.stack(np.meshgrid(np.linspace(0,1,N_samples), 
                                   np.linspace(0,1,N_samples)), 
                       axis=-1).reshape([-1,2])
# Calculate field sizes
field_size = 0.02*1/N_fields

# Create a set of example neurons across both environments,
# to see how a population of conjunctive grid-ovc codes would remap
# The main free parameter here is the ratio of vector vs sensory conjunctions
# The vector-conjunctions will remap when the home location changes,
# while the sensory conjunctions remain the same. Small example: 5, 8, 4
N_grids = 10
N_sens = 40
N_ovc = 20
# Fix the random seed so we get the same neurons each time
np.random.seed(0)
# Set offsets of grid cells in first environment
grid_offsets = [np.random.rand(2) * 1/N_fields for cell in range(N_grids)]
# Set reward locations on both days, extracted from P&F data (there arena is 200x200 cm)
obj_loc = [np.array([86,82])/200,np.array([147,140])/200]
# Sample ovc offsets (where is the firing field with respect to the object)
# Sample in polar coordinates around the object, so all directions are covered
ovc_offsets = np.random.rand(N_ovc,2)
# Reorder them so the first four ovcs are near home, make for better plotting
# But not the ones all the way at home because they don't illustrate vectors nicely
ovc_offsets = ovc_offsets[np.roll(np.argsort(ovc_offsets[:,0]),-2)]
# Then convert polar coordianates to cartesian ones
ovc_offsets = np.stack([ovc_offsets[:,0]*np.cos(2*np.pi*ovc_offsets[:,1]), \
                        ovc_offsets[:,0]*np.sin(2*np.pi*ovc_offsets[:,1])], axis=1)
# Set locations for sensory observations
sens_locations = np.random.rand(N_sens,2)
# Create all ratemaps. Grid and sensory responses remain the same when reward moves
grids = [get_location_rates(sample_locs, 
                            get_grid_locations(N_fields, offset=o), 
                            field_size).reshape([N_samples, N_samples])
         for o in grid_offsets]
senses = [get_location_rates(sample_locs, 
                           l.reshape([1, -1]), 
                           field_size).reshape([N_samples, N_samples])
         for l in sens_locations]
# Object vector cells change when reward location changes, so extra list dimension
ovcs = [[get_location_rates(sample_locs, 
                           (obj + o).reshape([-1, 2]), 
                           field_size*0.6).reshape([N_samples, N_samples])
         for o in ovc_offsets] for obj in obj_loc]
# Calculate conjunctions of grids and senses (place) and grids and ovcs (landmarks)
places = [[g * s for g in grids] for s in senses]
landmarks = [[[g * o for g in grids] for o in ovc] for ovc in ovcs]

# For plotting: set names of grids, ovcs, and senses.
# Grids are numbers, senses and ovcs letters, so that their conjunctions are e.g. A1
grid_names = [str(i) for i in range(N_grids)]
sens_names = ['%c' % (x+97) for x in range(N_sens)]
ovc_names = ['%c' % (x+97) for x in range(26-N_ovc, 26)]

# Set the maximum firing rate for entorhinal cells to 20Hz, for hippocampal cells to 20 Hz
# Roughly inspired by Pfeiffer & Foster 2013 (hpc, Supplementary Fig 1) 
# and Hoydal et al 2019 (ec, Extended Data Figure 8, 9)
ec_max = 20
hpc_max = 20
for cells in [grids, senses, ovcs[0], ovcs[1]]:
    curr_max = np.max(np.stack(cells).reshape((-1)))
    for i in range(len(cells)):
        cells[i] *= ec_max/curr_max
for cells in [places, landmarks[0], landmarks[1]]:
    curr_max = np.max(np.stack(cells).reshape((-1)))
    for i, c in enumerate(cells):
        for j in range(len(c)):
            cells[i][j] *= hpc_max/curr_max

# Plot all conjunctions
N_to_plot=4
for i, (curr_other, curr_conjunction, curr_names, title) in enumerate(zip(
        [senses, ovcs[0], ovcs[1]],
        [places, landmarks[0], landmarks[1]],
        [sens_names, ovc_names, ovc_names],
        ['Sensory conjunctions','Home well 1 vector conjunctions', 'Home well 2 vector conjunctions'])):
    plt.figure(figsize=(4*fig_scale,4*fig_scale))
    for row in range(N_to_plot + 1):
        for col in range(N_to_plot + 1):
            # Create axes, except for top left
            if row > 0 or col > 0:
                ax = plt.subplot(N_to_plot + 1, N_to_plot + 1, row * (N_to_plot + 1) + col + 1)
            # First row: plot each ovc in one col
            if row == 0 and col > 0:
                if col - 1 < len(curr_other):
                    plot_ratemap(curr_other[col-1], ax, obj_loc = obj_loc[i-1] if i > 0 else None)
                    #plt.text(0,1,curr_names[col-1], color=[1,1,1], ha='left',va='top')
                    plt.text(0,1,f'{np.max(curr_other[col-1]):.0f}', color=[1,1,1], ha='left',va='top')
            # First col: plot each grid in one row
            if row > 0 and col == 0:
                if row - 1 < len(grids):
                    plot_ratemap(grids[row-1], ax, obj_loc = obj_loc[i-1] if i > 0 else None)
                    #plt.text(0,1,grid_names[row-1], color=[1,1,1], ha='left',va='top')   
                    plt.text(0,1,f'{np.max(grids[row-1]):.0f}', color=[1,1,1], ha='left',va='top')
            # All others: plot conjunctions
            if row > 0 and col > 0:
                if col - 1 < len(curr_conjunction) and row - 1 < len(curr_conjunction[col-1]):
                    plot_ratemap(curr_conjunction[col-1][row-1], ax, obj_loc = obj_loc[i-1] if i > 0 else None)
                    #plt.text(0,1,grid_names[row-1] + curr_names[col-1], color=[1,1,1], ha='left',va='top')
                    plt.text(0,1,f'{np.max(curr_conjunction[col-1][row-1]):.0f}', color=[1,1,1], ha='left',va='top')
    plt.suptitle(title,fontsize=plt.rcParams['font.size'],fontweight='bold')
    if do_save:
        plt.savefig(fig_path + 'Sim_a_Ratemaps_' + str(i) + '.svg')


# The first hypothesis is that replay creates conjunctions.
# I'm going to sample replay trajectories evenly distributed throughout the session
# I'll assume that home replays (that make vector cells) 
# and replays elswhere (that don't) occur in alternation 
# I'll assume the full session takes 3000 s, roughly based on P&F

# The ratemap change for place cells is easy: it's always 0
# For landmarks it's trickier. We'll assume their first spike creates the field,
# so for that spike it's exactly the full peak rate
# But for following spikes, there is some portion of zero rate and some of full rate
# for the activity before the spike, and always full rate after,
# so the change will depend on exactly when the first spike was, 
# and when the current spike is!

# Set all replay times, evenly through session
N_replays = 100
replay_times = np.linspace(10,3000,N_replays)

# Make an array of whether this is a home or away replay
replay_from_home = np.arange(N_replays) % 4 == 0

# Also keep track of when the first spike appeared - set to -1 initially
first_landmark_spike = [-1*np.ones((N_ovc, N_grids)) for _ in range(2)]

# Now let's simulate some replays!
replay_steps = 20
replay_trajectories = [[],[]]
for o, obj in enumerate(obj_loc):
    for h in replay_from_home:
        curr_length = np.random.rand()+0.25
        curr_dir = 2*np.pi*np.random.rand()
        curr_start = obj if h else np.random.rand(2)
        curr_trajectory = np.stack([np.linspace(s, s+curr_length*f(curr_dir), replay_steps) 
                                    for s, f in zip(curr_start, [np.cos, np.sin])], axis=1)
        # Cut trajectory when outside, or when too close to home for replay elsewhere
        # In the real data replay elsewhere can't get within 30cm from home, so that's 30/200
        valid = [all([x > 0 and x < 1 for x in t]) for t in curr_trajectory]
        if not h:
            valid = [v and np.sqrt(np.sum((t-obj)**2)) > 30/200 
                     for t, v in zip(curr_trajectory, valid)]
        curr_trajectory = np.stack([t for i, t in enumerate(curr_trajectory) if all(valid[:i])])
        # Add trajectory to the list
        replay_trajectories[o].append(curr_trajectory)

plt.figure()
for r,h in zip(replay_trajectories[0], replay_from_home):
    if h:
        plt.plot(r[:,0], r[:,1], color=[0,0,0])
    else:
        plt.plot(r[:,0], r[:,1], color=[0.5,0.5,0.5])
if do_save:
    plt.savefig(fig_path + 'Sim_b_Replays.svg')


# Run through replays and collect replay spikes
replay_spikes = [[],[]]
replay_changes = [[],[]]
change_noise = 0.5;
smooth_noise = 1; # To match how ratemaps are made: 4cm smoothing, so 1 bin
smooth = lambda x : scipy.ndimage.filters.gaussian_filter(
    x, [smooth_noise, smooth_noise], mode='constant')

for env, (trajectories, landmark) in enumerate(zip(replay_trajectories, landmarks)):
    for time, from_home, trajectory in zip(replay_times, replay_from_home, trajectories):
        # I want to sample replay spikes from ratemaps. But the ratemap frequencies
        # (e.g. peak of 10Hz) are modelled after those in physical behaviour, 
        # where an animal spends much more time in a bin than during fast replay.
        dt = 0.005 # Each replay timestep takes 5 ms in the empirical analysis
        # I'll need some arbitrary scaling factor to get spike probabilities
        # during replay given ratemaps of awake behaviour. It's not obvious how to
        # do that translation, so instead I just calibrate it to get ~30 replay spikes
        # per ripple, which roughly matches the proportion in the dataset
        ratemap_scale = 5;
        # Store cell_s, cell_g, x, y tuples in each step
        place_spikes = []
        landmark_spikes = []
        for step in trajectory:
            # Get bin indices for current step location
            bin_i, bin_j = [int(step[0]*N_samples), int(step[1]*N_samples)]
            # Only continue if both within the arena
            if bin_i < N_samples and bin_i >= 0 and bin_j < N_samples and bin_j>=0:
                for is_landmark, (cell_list, spike_list) in enumerate(zip(
                        [places, landmark], [place_spikes, landmark_spikes])):
                    for s, cells in enumerate(cell_list):
                        for g, cell in enumerate(cells):
                            # If this is a landmark cell in a non-home replay:
                            # can only fire if it has fired before (in home replay)
                            if not (is_landmark and not from_home 
                                    and first_landmark_spike[env][s,g] == -1):
                                # Sample nr of spikes given ratemap and time in bin
                                if np.random.poisson(cell[bin_j, bin_i]*dt*ratemap_scale) > 0:
                                    spike_list.append((s, g, step[0], step[1]))
        # Now prepare change maps for all cells
        place_change = [[smooth(change_noise * np.random.randn(N_samples, N_samples)) 
                         for cell in cells] for cells in places]
        landmark_change = [[smooth(change_noise * np.random.randn(N_samples, N_samples))
                            + (first_landmark_spike[env][s,g] / time * cell 
                               if first_landmark_spike[env][s,g] > 0 else cell)
                            for g, cell in enumerate(cells)] 
                           for s, cells in enumerate(landmark)]
        # Then run through all spikes and copy their change maps
        place_spike_change = np.full((len(place_spikes), N_samples*2, N_samples*2), np.nan)
        landmark_spike_change = np.full((len(landmark_spikes), N_samples*2, N_samples*2), np.nan)
        for spikes, changes, spike_change in zip(
                [place_spikes, landmark_spikes], 
                [place_change, landmark_change], 
                [place_spike_change, landmark_spike_change]):
            for i, spike in enumerate(spikes):
                bin_i, bin_j = [int(spike[2]*N_samples), int(spike[3]*N_samples)]
                spike_change[i][(N_samples - bin_j):(2*N_samples-bin_j), 
                                (N_samples-bin_i):(N_samples*2 - bin_i)] = \
                    changes[spike[0]][spike[1]]
                    
        # Now we have a set of spikes, with associated change maps.
        # In the data we average across spikes of the same cell to get 1 change map per replay
        # Do the same here. Also take care of first time landmark spikes, only in home replays
        all_place_changes = []
        all_landmark_changes = []        
        for is_landmark, (spikes, changes, (s, g), all_changes) in enumerate(
                zip([np.array(place_spikes), np.array(landmark_spikes)],
                    [place_spike_change, landmark_spike_change],
                    [(N_sens, N_grids), (N_ovc, N_grids)],
                    [all_place_changes, all_landmark_changes])):
            for curr_s in range(s):
                for curr_g in range(g):
                    curr_spikes = np.logical_and(spikes[:,0] == curr_s, spikes[:,1] == curr_g) \
                        if len(spikes)>0 else []
                    if np.any(curr_spikes):
                        if is_landmark and first_landmark_spike[env][curr_s, curr_g] == -1:
                            # This is a landmark cell that never fired before.
                            if from_home: # This should always be true (see spike sampling)
                                first_landmark_spike[env][curr_s, curr_g] = time
                                all_changes.append(np.nanmean(changes[curr_spikes],axis=0))
                            else:
                                print('Something wrong!! New landmark spike in replay elsewhere')
                        else:
                            all_changes.append(np.nanmean(changes[curr_spikes],axis=0))

        # Finally: make some pretty plots of cells that fire in replay,
        # to demonstrate the differences between replays from home and replays elsewhere
        if env == 1 and (time == replay_times[0] or time == replay_times[1]
                         or time == replay_times[int(0+np.floor(N_replays/4-1)*4)]
                         or time == replay_times[int(0+np.floor(N_replays/4-1)*4 + 1)]):
            # For setting the title: determine what kind of replay this is
            time_label = ('Late' if time > replay_times[-1]/2 else 'Early') + ' replay '
            home_label = 'from home' if from_home else 'elsewhere'
            plt.figure(figsize=(8*fig_scale,4*fig_scale))
            ax = plt.axes()
            # Figure will have three parts: place cell examples to the left,
            # arena in the center, landmark cell examples to the right
            N_examples = 4;
            for cells, spikes, changes, offset in zip(
                    [places, landmark], 
                    [place_spikes,landmark_spikes],
                    [place_change, landmark_change],
                    [-2/N_examples, 1]):
                # Get unique place cells
                if len(spikes) == 0:
                    continue;
                curr_cells = np.unique(np.array(spikes)[:,:2], axis=0)
                # Only keep the required number of them
                curr_cells = curr_cells[np.random.choice(len(curr_cells), min(N_examples, len(curr_cells)),replace=False)]
                # Then plot their ratemaps and their change maps
                for i, curr_cell in enumerate(curr_cells):
                    # Plot ratemap
                    plt.imshow(cells[int(curr_cell[0])][int(curr_cell[1])], 
                               extent=[offset,offset+1/N_examples,
                                       i/N_examples,(i+1)/N_examples], 
                               vmin=0, vmax = np.max(cells[int(curr_cell[0])][int(curr_cell[1])]),
                               origin='lower')
                    plt.text(offset, (i+1)/N_examples, 
                             f'{np.max(cells[int(curr_cell[0])][int(curr_cell[1])]):.0f}',
                             ha='left',va='top', color=[1,1,1])                    
                    # Plot changemap
                    change = changes[int(curr_cell[0])][int(curr_cell[1])]
                    plt.imshow(-change/np.nanmax(change)*(change>0)
                               + change/np.nanmin(change)*(change<0),                               
                               extent=[offset+1/N_examples,offset+2/N_examples, 
                                       i/N_examples, (i+1)/N_examples], 
                               vmin=-1, vmax=1, cmap='RdBu',
                               origin='lower')
                    plt.text(offset+1/N_examples, (i+1)/N_examples, 
                             f'{np.min(change):.1f}', color=[0,0,1],
                             ha='left',va='top')                    
                    plt.text(offset+2/N_examples, (i+1)/N_examples, 
                             f'{np.max(change):.1f}', color=[1,0,0],
                             ha='right',va='top')        
                    # Plot border around cell
                    ax.add_patch(patches.Rectangle((offset, i/N_examples), 1/N_examples, 1/N_examples, 
                                                   linewidth=2, edgecolor=[1,1,1], facecolor='none'))
                    ax.add_patch(patches.Rectangle((offset+1/N_examples, i/N_examples), 1/N_examples, 1/N_examples, 
                                                   linewidth=2, edgecolor=[1,1,1], facecolor='none'))                    
                    # Plot lines to spike locations
                    curr_spikes = np.where(np.all(np.array(spikes)[:,:2]==curr_cell, axis=1))[0]
                    for curr_spike in curr_spikes:
                        plt.plot([offset if offset == 1 else 0, spikes[curr_spike][2]], 
                                 [(i+0.5)/N_examples, spikes[curr_spike][3]], '--', 
                                 color=[1,0,0] if offset < 0 else [0,0,1])
            # Draw text above 
            plt.text(-2/N_examples, 1, 'Example place cells', ha='left', va='bottom')
            plt.text(1+2/N_examples, 1, 'Example landmark cells', ha='right', va='bottom')
            # Draw arena with replay trajectory and place spikes
            plt.scatter(trajectory[:,0], trajectory[:,1], color=[0,1,0], marker='o', label='Replay trajectory')
            if len(place_spikes) > 0:
                plt.scatter(np.array(place_spikes)[:,2]+0.01*np.random.randn(len(np.array(place_spikes)[:,2])), 
                            np.array(place_spikes)[:,3]+0.01*np.random.randn(len(np.array(place_spikes)[:,2])), 
                            color=[1,0,0], marker='x', label='Place cell replay spikes')
            if len(landmark_spikes) > 0:
                plt.scatter(np.array(landmark_spikes)[:,2]+0.01*np.random.randn(len(np.array(landmark_spikes)[:,2])), 
                            np.array(landmark_spikes)[:,3]+0.01*np.random.randn(len(np.array(landmark_spikes)[:,2])), 
                            color=[0,0,1], marker='x', label='Landmark cell replay spikes')
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor=[0,0,0], facecolor='none'))    
            plt.legend(loc='lower center')                
            # Draw border around arena
            plt.xlim([-2/N_examples,1+2/N_examples])
            plt.ylim([0,1])
            plt.xticks([])
            plt.yticks([])
            plt.title(time_label + home_label)
            if do_save:
                plt.savefig(fig_path + 'Sim_c_Example_'+time_label + home_label + '.svg')

        # Append results to arrays
        replay_spikes[env].append(place_spikes + landmark_spikes)
        replay_changes[env].append(all_place_changes + all_landmark_changes)
        # Display progress
        print(f'Finished env {env}, time {time}')
                            
# Now do the key analysis: aligned ratemap changes
changes = np.concatenate([np.stack(a) for a, h in zip(
    replay_changes[0] + replay_changes[1], replay_from_home + replay_from_home) 
    if h and len(a) > 0])
control_changes = np.concatenate([np.stack(a) for a, h in zip(
    replay_changes[0] + replay_changes[1], replay_from_home + replay_from_home) 
    if not h and len(a) > 0])
        
# Make circular ROI
radius = N_samples/2;
center = N_samples;
roi = np.zeros((N_samples*2, N_samples*2))
for row in range(int(center-radius),int(center+radius)):
    for col in range(int(np.ceil(center-np.sqrt(radius**2 - (center-row)**2))),
                     int(np.floor(center+np.sqrt(radius**2 - (center-row)**2)))):
        roi[row, col] = 1;

# Then do average across change maps
average_map = np.nanmean(changes,axis=0)
control_map = np.nanmean(control_changes,axis=0)
lims = [0, np.nanmax(np.stack([average_map[roi==1], control_map[roi==1]]))]
plt.figure(figsize=(4*fig_scale,2*fig_scale)); 
plt.subplot(1,2,1)
plt.imshow(np.ma.masked_array(average_map, mask=1-roi), origin='lower',
           vmin=lims[0], vmax=lims[1], extent=[-N_samples*4,N_samples*4, -N_samples*4, N_samples*4])
plt.xlim([-100,100])
plt.ylim([-100,100])
plt.colorbar();

plt.xlabel('dx from spike (cm)')
plt.ylabel('dy from spike (cm)')
plt.title('Replay from home',fontsize=plt.rcParams['font.size'],fontweight='bold')
plt.subplot(1,2,2)
plt.imshow(np.ma.masked_array(np.nanmean(control_changes,axis=0), mask=1-roi), origin='lower',
           vmin=lims[0], vmax=lims[1], extent=[-N_samples*4,N_samples*4, -N_samples*4, N_samples*4])
plt.xlim([-100,100])
plt.ylim([-100,100])
plt.xlabel('dx from spike (cm)')
plt.title('Replay elsewhere',fontsize=plt.rcParams['font.size'],fontweight='bold')
plt.colorbar();
plt.tight_layout()

if do_save:
    plt.savefig(fig_path + 'Sim_d_Aligned.svg')


# Do the additional stats on these change maps: radial lines and central roi
radius = 2; # Bins here are half those in data analysis! So radius should be 2, not 4
center = N_samples;
small_roi = np.zeros((N_samples*2, N_samples*2))
for row in range(int(center-radius),int(center+radius)):
    for col in range(int(np.ceil(center-np.sqrt(radius**2 - (center-row)**2))),
                     int(np.floor(center+np.sqrt(radius**2 - (center-row)**2)))):
        small_roi[row, col] = 1;        
angles = np.linspace(0,2*np.pi,21);
angles = angles[1:] # Because linspace includes start value
steps = np.arange(0, int(N_samples/2));

stats_roi = []
stats_angle = []
for curr_changes in [changes, control_changes]:
    curr_stats_roi = np.full(curr_changes.shape[0], np.nan)
    curr_stats_angle = np.full((curr_changes.shape[0], len(steps)), np.nan)
    for m, curr_map in enumerate(curr_changes):
        curr_stats_roi[m] = np.nanmean(np.ma.masked_array(curr_map, small_roi-1))
        curr_angles = []
        interp = RegularGridInterpolator((np.arange(N_samples*2), np.arange(N_samples*2)), 
                                         curr_map, bounds_error=False, fill_value=None)
        for a in angles:
            x = N_samples + steps*np.cos(a)
            y = N_samples + steps*np.sin(a)
            curr_angles.append(interp((y, x)))
        curr_stats_angle[m] = np.nanmean(np.stack(curr_angles, axis=0), axis=0)
    stats_roi.append(curr_stats_roi)
    stats_angle.append(curr_stats_angle)
stats_roi, control_stats_roi = stats_roi
stats_angle, control_stats_angle = stats_angle

plt.figure(figsize=(4*fig_scale,2*fig_scale)); 
plt.subplot(1,2,1)
plt.bar([0, 1], [np.nanmean(stats_roi), np.nanmean(control_stats_roi)])
plt.errorbar([0,1],[np.nanmean(stats_roi), np.nanmean(control_stats_roi)], 
             [np.nanstd(stats_roi)/np.sqrt(stats_roi.shape[0]), 
              np.nanstd(control_stats_roi)/np.sqrt(control_stats_roi.shape[0])], 
             linestyle='None', color=[0,0,0])
plt.xticks([0, 1], ['Home', 'Other'])
plt.xlabel('Replay trajectories')
plt.xlim([-1,2])
plt.ylim([0, np.nanmean(stats_roi*1.4)])
plt.ylabel('Ratemap change (Hz)')
plt.subplot(1,2,2)
plt.errorbar(steps*4,np.nanmean(stats_angle,axis=0),
             np.nanstd(stats_angle,axis=0)/np.sqrt(stats_angle.shape[0]))
plt.errorbar(steps*4,np.nanmean(control_stats_angle,axis=0), 
             np.nanstd(control_stats_angle,axis=0)/np.sqrt(control_stats_angle.shape[0]))
plt.legend(['Home', 'Other'])
plt.ylim([0, np.nanmean(stats_roi*1.4)])
plt.xlabel('Radial distance (cm)')
plt.tight_layout()

if do_save:
    plt.savefig(fig_path + 'Sim_e_Stats.svg')

# Next up: anticorrelated change maps on day 2.
# Conveniently, we *do* actually have a change map here
landmark_change = [[g2 - g1 + smooth(change_noise * np.random.randn(N_samples, N_samples)) 
                    for g1, g2 in zip(o1, o2)] for o1, o2 in zip(*landmarks)]
plt.figure(figsize=(4*fig_scale,4*fig_scale))
# Plot the change maps
for row in range(N_to_plot + 1):
    for col in range(N_to_plot + 1):
        # Create axes, except for top left
        if row > 0 or col > 0:
            ax = plt.subplot(N_to_plot + 1, N_to_plot + 1, row * (N_to_plot + 1) + col + 1)
        # First row: plot each ovc in one col
        if row == 0 and col > 0:
            plot_ratemap(ovcs[1][col-1], ax, obj_loc = obj_loc[1], max_val=ec_max)
            #plt.text(0,1,ovc_names[col-1], color=[1,1,1], ha='left',va='top')      
            plt.text(0,1,f'{np.max(ovcs[1][col-1]):.0f}', color=[1,1,1], ha='left',va='top')
        # First col: plot each grid in one row
        if row > 0 and col == 0:
            plot_ratemap(grids[row-1], ax, obj_loc = obj_loc[1], max_val=ec_max)
            #plt.text(0,1,grid_names[row-1], color=[1,1,1], ha='left',va='top') 
            plt.text(0,1,f'{np.max(grids[row-1]):.0f}', color=[1,1,1], ha='left',va='top')
        # All others: plot conjunctions
        if row > 0 and col > 0:
            plt.imshow(-landmark_change[col-1][row-1]/np.nanmax(landmark_change[col-1][row-1])*(landmark_change[col-1][row-1]>0)
                       + landmark_change[col-1][row-1]/np.nanmin(landmark_change[col-1][row-1])*(landmark_change[col-1][row-1]<0), 
                        extent=[0,1,0,1], origin='lower', 
                       vmin=-1, vmax=1, cmap='RdBu') # Change of sign needed because Rd->Bu
            ax.add_patch(plt.Circle(obj_loc[1], radius=0.075, fc=[1,1,1], ec=[0,0,0])) 
            plt.text(0,0,f'{np.nanmin(landmark_change[col-1][row-1]):.0f}', color=[0,0,1], ha='left',va='bottom')
            plt.text(1,0,f'{np.nanmax(landmark_change[col-1][row-1]):.0f}', color=[1,0,0], ha='right',va='bottom')            
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.text(0,1,grid_names[row-1] + ovc_names[col-1], color=[1,1,1], ha='left',va='top')
plt.suptitle('Ratemap changes after home location change',fontsize=plt.rcParams['font.size'],fontweight='bold')

if do_save:
    plt.savefig(fig_path + 'Sim_f_Ratemap_Changes.svg')

# Get aligned maps both for current and previous goal
aligned_corr = []
for do_place, curr_cells in enumerate([landmark_change, places]):
    curr_corr = []
    for o, cells in enumerate(curr_cells):
        for g, cell in enumerate(cells):        
            # Make two copies of change maps, one aligned on old home and one on new home
            align_old = np.full((N_samples*2, N_samples*2), np.nan)
            align_new = np.full((N_samples*2, N_samples*2), np.nan)
            # Fill them both
            for obj, curr_map in zip(obj_loc, [align_old, align_new]):
                # Find home well bin
                bin_i, bin_j = [int(i) for i in obj*N_samples]            
                # Copy each landmark's ratemap change from day 1 to day 2
                curr_map[(N_samples - bin_j):(2*N_samples-bin_j), 
                         (N_samples-bin_i):(N_samples*2 - bin_i)] = \
                    (smooth(change_noise * np.random.randn(N_samples, N_samples)) if do_place else cell)
            # Then correlate
            to_include = np.logical_and(np.logical_not(np.isnan(align_old)),
                                        np.logical_not(np.isnan(align_new)))
            curr_corr.append(np.corrcoef(align_old[to_include].reshape((-1)), 
                                         align_new[to_include].reshape((-1)))[0,1])
            # And make a figure just for the very first one
            if g == 0 and o == 3:
                plt.figure(figsize=(4*fig_scale,2*fig_scale));
                plt.subplot(1,2,1);
                plt.imshow(-align_new/np.nanmax(align_new)*(align_new>0)
                           + align_new/np.nanmin(align_new)*(align_new<0), 
                            extent=[-N_samples*4,N_samples*4, -N_samples*4, N_samples*4], 
                           vmin=-1, vmax=1, cmap='RdBu', origin='lower')  # Change of sign needed because Rd->Bu
                plt.scatter(0,0,marker='x',color=[0,0,0])
                plt.xlabel('dx from home (cm)')
                plt.ylabel('dy from home (cm)')
                plt.title('Total ratemap change',fontsize=plt.rcParams['font.size'],fontweight='bold')
                plt.subplot(1,2,2);
                plt.imshow(-align_old/np.nanmax(align_old)*(align_old>0)
                           + align_old/np.nanmin(align_old)*(align_old<0), 
                            extent=[-N_samples*4,N_samples*4, -N_samples*4, N_samples*4], 
                           vmin=-1, vmax=1, cmap='RdBu', origin='lower')  # Change of sign needed because Rd->Bu
                plt.scatter(0,0,marker='x',color=[0,0,0])                
                plt.xlabel('dx from home (cm)')                
                plt.title(f'Correlation: {curr_corr[-1]:.2f}',fontsize=plt.rcParams['font.size'],fontweight='bold')
                plt.tight_layout()
                
                if do_save:
                    plt.savefig(fig_path + 'Sim_g_Aligned_Example_' + str(do_place) + '.svg')

    aligned_corr.append(curr_corr)

# Plot a histogram
histbins = np.linspace(-np.max(np.abs(np.concatenate(aligned_corr))), 
                       np.max(np.abs(np.concatenate(aligned_corr))), 50)
histstep = (histbins[1]-histbins[0]);
plt.figure(figsize=(4*fig_scale,2*fig_scale));
plt.subplot(1,2,1)
counts, _ = np.histogram(np.concatenate(aligned_corr), histbins)
plt.bar(histbins[:-1]+histstep*0.5, counts, width=histstep)
plt.xlim([histbins[0], 
          histbins[-1]])
plt.ylim([0, np.max(counts)+4])
plt.xlabel('Change correlation')
plt.ylabel('Cell count')
plt.subplot(1,2,2)
counts = [np.histogram(c, histbins)[0] for c in aligned_corr]
plt.bar(histbins[:-1]+(histbins[1]-histbins[0])*0.5, counts[0], width=histstep)
plt.bar(histbins[:-1]+(histbins[1]-histbins[0])*0.5, counts[1], bottom=counts[0], width=histstep)
plt.xlim([histbins[0], 
          histbins[-1]])
plt.ylim([0, np.max(np.sum(np.stack(counts,axis=0),axis=0))+4])
plt.xlabel('Change correlation')
plt.legend(['Landmark cells', 'Place cells'])
plt.tight_layout()

if do_save:
    plt.savefig(fig_path + 'Sim_g_Aligned_Histogram.svg')

# To get identical-looking plots in matlab: export result as matlab matrices
matlab_path = '/Users/jbakermans/Google Drive/DPhil/Matlab/PfeifferFoster/output'
# Create a bit dictionary that exports all simulated data to a matlab struct
mat = {'changes': changes, 'changes_control': control_changes, 
       'stats_roi': stats_roi, 'stats_roi_control': control_stats_roi,
       'stats_angle': stats_angle,  'stats_angle_control': control_stats_angle,
       'aligned_changes': np.stack(landmark_change),
       'aligned_corr_landmark': aligned_corr[0],
       'aligned_corr_place': aligned_corr[1]}
if True:
    savemat(matlab_path + '/sim.mat', mat)
