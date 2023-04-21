#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:03:04 2021

@author: jbakermans
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import numpy as np

def plot_map(env, ax=None, radius=None, comp_cm='tab10', black_walls=False):
    # Initialise empty axis
    ax = initialise_axes(ax)
    # Set limits for axis according to env limist
    ax.set_xlim(env.xlim)
    ax.set_ylim(env.ylim)
    # Calculate radius of location circles based on how env size
    radius = 0.05 * min([lims[1] - lims[0] for lims in [env.xlim, env.ylim]]) \
        if radius is None else radius
    # For standardized plotting style: black walls, green reward
    if black_walls:
        colmap = {key: [0, 0, 0] if val['type'] == 'wall' \
                  else cm.get_cmap('tab10', 10)(2) 
                  for key,val in env.components.items()}
    else:
        # Create colours for each component
        colmap = {comp: cm.get_cmap(comp_cm, len(env.components.keys())*3)(c_i)
                  for c_i, comp in enumerate(env.components.keys())}
    # Create empty list of component patches and lines
    comp_patches, comp_lines = [], []    
    # Plot each environment component
    for c_i, (name, comp) in enumerate(env.components.items()):
        # Run through locations in this environment
        for l_i, loc_from in enumerate(comp['locations']):
            # Create patch for location, only if not jus plotting black walls
            if comp['type'] != 'wall' or not black_walls:
                comp_patches.append(plt.Circle(
                    loc_from, radius * (1 - 0.5 * (comp['type']=='wall')), 
                    fc=colmap[name], ec=[0, 0, 0], zorder=c_i + 1.3))
            # For walls: create line between locations
            if comp['type'] == 'wall':
                for l_j, loc_to in enumerate(comp['locations']):
                    if l_i > l_j:
                        # Line for outline
                        comp_lines.append(plt.Line2D(
                            [loc_from[0], loc_to[0]], [loc_from[1], loc_to[1]],
                            linewidth=6, color=[0, 0, 0], zorder=c_i + 1.1))
                        # Line for colour
                        comp_lines.append(plt.Line2D(
                            [loc_from[0], loc_to[0]], [loc_from[1], loc_to[1]],
                            linewidth=4, color=colmap[name], zorder=c_i + 1.2))
    # Add lines and locations to axis
    for patch in comp_patches:
        ax.add_patch(patch)   
    for line in comp_lines:
        ax.add_line(line)             
    # Return axes for further use
    return ax

def plot_values(env, locs, vals, ax=None, val_cm='viridis', n_x=20, n_y=20,
                max_val=None, min_val=None):
    # Initialise empty axis if axis wasn't provided
    if ax is None:
        ax = initialise_axes(ax)    
    # If min_val and max_val are not specified: take the minimum and maximum of the supplied values
    min_val = np.min(vals) if min_val is None else min_val
    max_val = np.max(vals) if max_val is None else max_val     
    # Create grid for image
    xs, ys = np.meshgrid(np.linspace(env.xlim[0], env.xlim[1], n_x),
                       np.linspace(env.ylim[0], env.ylim[1], n_y))    
    # Resample provided values
    resampled = griddata(np.array(locs), vals, (xs, ys), method='linear')
    # Plot values
    ax.imshow(resampled, extent=[env.xlim[0], env.xlim[1], env.xlim[0], env.ylim[1]],
              origin='lower', vmin=min_val, vmax=max_val)
    # Return axis for further use
    return ax

def plot_policy(env, locs, dirs, ax=None, comp_cm='tab10', big_arrow_head=False):
    # Initialise empty axis if axis wasn't provided
    if ax is None:
        ax = initialise_axes(ax)
    # Calculate arrow length based on number of arrows
    r = 0.75 * min([lims[1] - lims[0] for lims in [env.xlim, env.ylim]]) / np.sqrt(len(locs))
    # Plot arrow at each location
    for l, d in zip(locs, dirs):
        plt.arrow(l[0], l[1], r * np.cos(d), r * np.sin(d), 
                  #width=0.01, head_width=0.02, head_length=0.02,
                  width=r/4, 
                  head_width=r/2 + big_arrow_head * r/2, 
                  head_length=r/2 + big_arrow_head * r/4,
                  length_includes_head=True)
    # Return axes for further use
    return ax

def plot_model_env(model, env):
    plt.figure()
    # First subplot: just empty environment
    ax = plt.subplot(2, 2, 1)
    plot_map(env, ax=ax)
    ax.set_title('Environment')    
    # Below first: env with optimal policy
    ax = plt.subplot(2, 2, 3)
    plot_map(env, ax=ax)
    locs, dirs = env.get_policy(); 
    plot_policy(env, locs, dirs, ax=ax);    
    ax.set_title('Optimal policy')    
    # Second subplot: components example ovc
    ax = plt.subplot(2, 2, 2)
    cell = int(model.pars['ovc_n']*1.5 + 0.5*np.sqrt(model.pars['ovc_n']))
    reps = [model.get_location_representation(l)[cell] for l in locs]
    plot_map(env, ax=ax)
    plot_values(env, locs, reps, ax=ax, min_val=0, 
                max_val=max(model.get_location_representation([0, 0])))
    ax.set_title('Example ovc')
    # Final subplot: learned policy
    ax = plt.subplot(2, 2, 4)
    plot_map(env, ax=ax)
    dirs = model.learn_get_location_direction(model.policy_net, locs, env)
    plot_policy(env, locs, dirs, ax=ax);    
    ax.set_title('Learned policy')
    
def plot_model_pol(model, env):
    plt.figure()  
    # FIrst subplot: env with optimal policy
    ax = plt.subplot(1, 2, 1)
    plot_map(env, ax=ax)
    locs, dirs = env.get_policy(); 
    plot_policy(env, locs, dirs, ax=ax);    
    ax.set_title('Optimal policy')    
    # Second subplot: learned policy
    ax = plt.subplot(1, 2, 2)
    plot_map(env, ax=ax)
    dirs = model.learn_get_location_direction(model.policy_net, locs, env)
    plot_policy(env, locs, dirs, ax=ax);    
    ax.set_title('Learned policy')  
    
def plot_replay_pol(model, env):
    plt.figure()  
    # FIrst subplot: env with optimal policy
    ax = plt.subplot(1, 2, 1)
    plot_map(env, ax=ax)
    locs, dirs = env.get_policy(); 
    plot_policy(env, locs, dirs, ax=ax);    
    ax.set_title('Optimal policy')    
    # Second subplot: learned policy
    ax = plt.subplot(1, 2, 2)
    plot_map(env, ax=ax)
    qs = [np.matmul(model.reps['place']['rep']([loc]),
                    model.pars['replay_q_weights']) for loc in locs]
    dirs = [model.get_action_decoded(np.where(q == max(q))[0][0]) for q in qs]
    plot_policy(env, locs, dirs, ax=ax);    
    ax.set_title('Learned policy')      
    
def plot_model_rep(model, env):
    # Get locations on grid
    locs = env.get_grid_locs(20, 20)
    # Get full representation at each location
    reps = [model.get_location_representation(l) for l in locs]
    # Plot ovcs for each component
    for n_i, comp in enumerate(env.graph['v']):
        plt.figure()
        cell_to_start = n_i * model.reps['ovc']['n']
        for curr_cell in range(model.reps['ovc']['n']):
            ax = plt.subplot(model.reps['ovc']['rows'], model.reps['ovc']['cols'], curr_cell + 1)
            plot_map(env, ax=ax)
            plot_values(env, locs, [rep[cell_to_start + curr_cell] for rep in reps], ax=ax,
                        min_val=0, max_val=np.max(np.array(reps)))
        plt.suptitle('OVC representations, ' + comp['comp'] + ' object ' + str(comp['l_i']))
    # Plot place cells
    plt.figure()
    # Get full representation at each location
    reps = [model.get_location_representation(l, rep='place') for l in locs]    
    for curr_cell in range(model.reps['place']['n']):
        ax = plt.subplot(model.reps['place']['rows'], model.reps['place']['cols'], curr_cell + 1)
        plot_map(env, ax=ax)
        plot_values(env, locs, [rep[curr_cell] for rep in reps], ax=ax,
                    min_val=0, max_val=np.max(np.array(reps)))
    plt.suptitle('Place cell representations')

def plot_model_performance(model, env, N=5):
    # Get locations on grid
    locs = env.get_grid_locs(N, N)
    # Get surplus distance to goal vs optimal policy for full rep
    full_d = np.array(
        [model.learn_get_location_path(model.policy_net, l, env) for l in locs])
    # Get surplus distance to goal vs optimal policy for partial rep
    rep = model.rep_observe(); rep['reward'] = [[0,0] for r in rep['reward']]
    part_d = np.array(
        [model.learn_get_location_path(model.policy_net, l, env, rep=rep)
         for l in locs])
    
    # Now plot heatmaps of distance, or red if never makes the goal at all
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plot_map(env, ax=ax)
    plt.imshow(np.ma.masked_array(full_d, full_d==100).reshape(N, N), 
               extent=[env.xlim[0], env.xlim[1], env.xlim[0], env.ylim[1]],
               origin='lower', vmin=0, vmax=0.2, cmap='viridis')
    plt.imshow(np.ma.masked_array(full_d, full_d<100).reshape(N, N), 
               extent=[env.xlim[0], env.xlim[1], env.xlim[0], env.ylim[1]],
               origin='lower', vmin=99, vmax=101, cmap='Reds')
    ax.set_title('Full representation')
    ax = plt.subplot(1, 2, 2)    
    plot_map(env, ax=ax)
    plt.imshow(np.ma.masked_array(part_d, part_d==100).reshape(N, N), 
               extent=[env.xlim[0], env.xlim[1], env.xlim[0], env.ylim[1]],
               origin='lower', vmin=0, vmax=0.2, cmap='viridis')
    plt.imshow(np.ma.masked_array(part_d, part_d<100).reshape(N, N), 
               extent=[env.xlim[0], env.xlim[1], env.xlim[0], env.ylim[1]],
               origin='lower', vmin=99, vmax=101, cmap='Reds')
    ax.set_title('Reward representation only')    
    
def plot_reps(environment, walk, ax=None):
    # Build list of true locations
    locs = [s['location'] for s in walk]
    # Then a list for each representation
    reps = {}
    for name in walk[0]['representation'].keys():
        for r_i in range(len(walk[0]['representation'][name])):
            reps[name + '_' + str(r_i)] = [s['representation'][name][r_i]
                                           for s in walk
                                           if s['representation'][name][r_i] is not None]
    # Set a colour for each representation
    cols = cm.get_cmap('tab10', len(reps.keys())*3)
    # Plot environment
    ax = plot_map(environment, ax=ax)
    # Plot true walk in grey
    plot_walk(environment, locs, col=[0.5, 0.5, 0.5], ax=ax)
    # Then add each rep
    for i, (key, val) in enumerate(reps.items()):
        if len(val) > 0:
            plot_walk(environment, val, col=cols(i), ax=ax)

def plot_occupancy(environment, walk, visits):
    ax = plot_map(environment); 
    plt.imshow(visits.transpose(), 
               extent=[environment.xlim[0], environment.xlim[1], 
                       environment.ylim[0], environment.ylim[1]], 
               origin='lower'); 
    plt.colorbar(); 
    plot_walk(environment, [s['location'] for s in walk], ax=ax)

def plot_exp_4(model, env, exp, test):
    plt.figure(); 
    ax = plt.subplot(1,2,1); 
    plot_map(env, ax=ax); 
    plot_walk(env, [s['location'] for s in exp], ax=ax); 
    plot_walk(env, [s['location'] for s in test], ax=ax, col=(0,0,1)); 
    ax.set_title('Explore (red) then escape (blue)'); 
    ax = plt.subplot(1,2,2); 
    locs, dirs = env.get_policy();
    plot_map(env, ax=ax)
    qs = [np.matmul(model.reps['place']['rep']([loc]),
                    model.pars['replay_q_weights']) for loc in locs]
    dirs = [model.get_action_decoded(np.where(q == max(q))[0][0]) for q in qs]
    plot_policy(env, locs, dirs, ax=ax);    
    ax.set_title('Policy TD-learned in replay')
    
def plot_exp_3_5(env, exp, test):
    plt.figure(); 
    ax = plt.subplot(1,3,1); 
    plot_map(env, ax=ax); 
    plot_walk(env, [s['location'] for s in exp], ax=ax); 
    plot_walk(env, [s['location'] for s in test], ax=ax, col=(0,0,1)); 
    ax.set_title('Explore (red) then escape (blue)'); 
    ax = plt.subplot(1,3,2); 
    plot_reps(env, exp, ax=ax); 
    ax.set_title('True ( grey) and representation (blue) location');      
    ax = plt.subplot(1,3,3); 
    plot_reps(env, test, ax=ax); 
    ax.set_title('True ( grey) and representation (blue) location');      
    
def plot_memory(env, mem, ax=None):
    # Get all representations
    rep_keys = [key for key in mem.keys() if key[0:3] == 'ovc']
    # Set a colour for each representation
    cols = cm.get_cmap('tab10', len(rep_keys)*3)
    # Plot environment
    ax = plot_map(env, ax=ax)
    # Plot memories
    for i, rep in enumerate(rep_keys):
        mems = np.logical_not(np.any(np.isnan(mem[rep]), axis=-1))
        plt.plot(np.stack([mem['location'][mems, 0], mem[rep][mems, 0]]),
                 np.stack([mem['location'][mems, 1], mem[rep][mems, 1]]),
                 color=cols(i))
    # Plot true locations
    plt.scatter(mem['location'][:, 0], mem['location'][:, 1], 
             color=[0.5, 0.5, 0.5])
    ax.set_title('Memory location (blue) for real location (grey)')

def plot_replay(env, walk, step_interval=1, rep_interval=1):
    plt.figure(); 
    ax = plot_map(env); 
    plot_walk(env, [s['location'] for s in walk], ax=ax, col=(1,0,0));
    step_i = 0
    for step in walk:
        if 'replay' in step.keys() and len(step['replay']) > 0:
            step_i += 1
            if step_i % step_interval == 0:
                for rep_i, replay in enumerate(step['replay']):
                    if rep_i % rep_interval == 0:
                        plot_walk(env, [s['location'] for s in replay], 
                                  ax=ax, col=(0,0,1));             
    ax.set_title('Walk (red) and replay (blue)'); 

def plot_walk(environment, positions, max_steps=None, n_steps=1, ax=None, pred_correct=None, 
              col=(1, 0, 0), max_tint_shade=0.5, marker='.'):
    # Set maximum number of steps if not provided
    max_steps = len(positions) if max_steps is None else min(max_steps, len(positions))
    # Initialise empty axis if axis wasn't provided
    if ax is None:
        ax = initialise_axes(ax)
    # Use red color by default
    col = np.array(col)
    # Initialise previous location: location of first location
    prev_loc = np.array([positions[0][0], positions[0][1]])
    # Run through walk, creating lines
    for step_i in range(1, max_steps, n_steps):
        # Get location of current location, with some jitter so lines don't overlap
        new_loc = np.array([positions[step_i][0],
                            positions[step_i][1]])
        # If list of correct predictions was provided: plot x or o for prediction accuracy
        if pred_correct is not None:
            if pred_correct[step_i]:
                ax.scatter(new_loc[0], new_loc[1], s=25, color=[0, 1, 0], marker='o', zorder=1.5)
            else:
                ax.scatter(new_loc[0], new_loc[1], s=25, color=[1, 0, 0], marker='x', zorder=1.5)
        # Get current colour: from 0.5 tint to 0.5 shade (light to dark)
        curr_col = col + (1 - col) * max_tint_shade * (1 - step_i / (max_steps / 2)) if step_i < max_steps / 2 \
            else col * (1 - max_tint_shade * (step_i - (max_steps / 2)) / (max_steps / 2))
        # Plot line from previous location to current location
        ax.plot([prev_loc[0], new_loc[0]], [prev_loc[1], new_loc[1]], color=curr_col,
                marker=marker)
        # Update new location to previous location
        prev_loc = new_loc
    # Return axes that this was plotted on
    return ax



def initialise_axes(ax=None):
    """
    Initialise axes for plotting environment with default values (no ticks, 1:1 aspect ratio)
    """
    # If no axes specified: create new figure with new empty axes
    if ax is None:
        plt.figure()
        ax = plt.axes()
    # Set axes limits to 0, 1 as this is how the positions in the environment are setup
    ax.set_xticks([])
    ax.set_yticks([])
    # Force aspect ratio
    ax.set_aspect(1)
    # Return axes object
    return ax
