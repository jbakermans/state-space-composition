#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:24:20 2021

@author: jbakermans
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_map(environment, values, ax=None, min_val=None, max_val=None, num_cols=100, location_cm='viridis',
             action_cm='Pastel1', do_plot_actions=False, shape='circle', radius=None):
    """
    Plot map of environment where each location is coloured according to array of values with length of #locations
    """
    # If min_val and max_val are not specified: take the minimum and maximum of the supplied values
    min_val = np.min(values) if min_val is None else min_val
    max_val = np.max(values) if max_val is None else max_val
    # Create color map for locations: colour given by value input
    location_cm = cm.get_cmap(location_cm, num_cols)
    # Create color map for actions: colour given by action index
    action_cm = cm.get_cmap(action_cm, environment.n_actions)
    # Calculate colour corresponding to each value
    plotvals = np.floor((values - min_val) / (max_val - min_val) * num_cols) if max_val != min_val \
        else np.ones(values.shape)
    # Calculate radius of location circles based on how many nodes there are
    radius = 2 * (0.01 + 1 / (10 * np.sqrt(environment.n_locations))) if radius is None else radius
    # Initialise empty axis
    ax = initialise_axes(ax)
    # Create empty list of location patches and action patches
    location_patches, action_patches = [], []
    # Now start drawing locations and actions
    for i, location in enumerate(environment.locations):
        # Create patch for location
        location_patches.append(plt.Rectangle((location['x'] - radius / 2, location['y'] - radius / 2), radius, radius,
                                              color=location_cm(int(plotvals[i])))
                                if shape == 'square'
                                else plt.Circle((location['x'], location['y']), radius,
                                                color=location_cm(int(plotvals[i]))))
        # And create action patches, if action plotting is switched on
        if do_plot_actions:
            for a, action in enumerate(location['actions']):
                # Only draw patch if action probability is larger than 0
                if action['probability'] > 0:
                    # Find where this action takes you
                    locations_to = [environment.locations[loc_to]
                                    for loc_to in np.where(np.array(action['transition']) > 0)[0]]
                    # Create an action patch for each possible transition for this action
                    for loc_to in locations_to:
                        action_patches.append(action_patch(location, loc_to, radius, action_cm(action['id'])))
    # After drawing all locations, add shiny patches
    for location in environment.locations:
        # For shiny locations, add big red patch to indicate shiny
        if location['shiny']:
            # Create square patch for location
            location_patches.append(
                plt.Rectangle((location['x'] - radius / 2, location['y'] - radius / 2), radius, radius,
                              linewidth=1, facecolor='none', edgecolor=[1, 0, 0])
                if shape == 'square'
                else plt.Circle((location['x'], location['y']), radius,
                                linewidth=1, facecolor='none', edgecolor=[1, 0, 0]))
            # Add patches to axes
    for patch in location_patches + action_patches:
        ax.add_patch(patch)
    # Return axes for further use
    return ax


def plot_actions(locations, field='probability', ax=None, min_val=None, max_val=None,
                 num_cols=100, action_cm='viridis'):
    """
    Plot map of environment where all actions receive a colour according to field (e.g. 'probability' to plot policy)
    """
    # If min_val and max_val are not specified: take the minimum and maximum of the supplied values
    min_val = min([action[field] for location in locations for action in location['actions']]) \
        if min_val is None else min_val
    max_val = max([action[field] for location in locations for action in location['actions']]) \
        if max_val is None else max_val
    # Create color map for locations: colour given by value input
    action_cm = cm.get_cmap(action_cm, num_cols)
    # Calculate radius of location circles based on how many nodes there are
    radius = 1 * (0.01 + 1 / (10 * np.sqrt(len(locations))))
    # Initialise empty axis
    ax = initialise_axes(ax)
    # Create empty list of location patches and action patches
    location_patches, action_patches = [], []
    # Now start drawing locations and actions
    for i, location in enumerate(locations):
        # Create circle patch for location
        location_patches.append(plt.Circle((location['x'], location['y']), radius, color=[0, 0, 0]))
        # And create action patches
        for a, action in enumerate(location['actions']):
            # Only draw patch if action probability is larger than 0
            if action['probability'] > 0:
                # Calculate colour for this action from colour map
                action_colour = action_cm(int(np.floor((action[field] - min_val) / (max_val - min_val) * num_cols)))
                # Find where this action takes you
                locations_to = [locations[loc_to] for loc_to in np.where(np.array(action['transition']) > 0)[0]]
                # Create an action patch for each possible transition for this action
                for loc_to in locations_to:
                    action_patches.append(action_patch(location, loc_to, radius, action_colour))
    # Add patches to axes
    for patch in (location_patches + action_patches):
        ax.add_patch(patch)
    # Return axes for further use
    return ax


def plot_walk(environment, positions, max_steps=None, n_steps=1, ax=None, pred_correct=None, cmap='Reds'):
    """
    Sample array of positions in walk at regular intervals and plot a line changing colour along the walk
    """
    # Set maximum number of steps if not provided
    max_steps = len(positions) if max_steps is None else min(max_steps, len(positions))
    # Initialise empty axis if axis wasn't provided
    if ax is None:
        ax = initialise_axes(ax)
    # Find all circle patches on current axis
    location_patches = [patch_i for patch_i, patch in enumerate(ax.patches)
                        if type(patch) is plt.Circle or type(patch) is plt.Rectangle]
    # Get radius of location circles on this map
    radius = (ax.patches[location_patches[-1]].get_radius() if type(ax.patches[location_patches[-1]]) is plt.Circle
              else ax.patches[location_patches[-1]].get_width() / 2) if len(location_patches) > 0 else 0.02
    # Create color map position along walk
    cmap = cm.get_cmap(cmap)    
    # Initialise previous location: location of first location
    prev_loc = np.array([environment.locations[positions[0]]['x'], environment.locations[positions[0]]['y']])
    # Run through walk, creating lines
    for step_i in range(1, max_steps, n_steps):
        # Get location of current location, with some jitter so lines don't overlap
        new_loc = np.array([environment.locations[positions[step_i]]['x'],
                            environment.locations[positions[step_i]]['y']])
        # Add jitter (need to unpack shape for rand - annoyingly np.random.rand takes dimensions separately)
        new_loc = new_loc + 0.8 * (-radius + 2 * radius * np.random.rand(*new_loc.shape))
        # If list of correct predictions was provided: plot x or o for prediction accuracy
        if pred_correct is not None:
            if pred_correct[step_i]:
                ax.scatter(new_loc[0], new_loc[1], s=25, color=[0, 1, 0], marker='o', zorder=1.5)
            else:
                ax.scatter(new_loc[0], new_loc[1], s=25, color=[1, 0, 0], marker='x', zorder=1.5)
        # Plot line from previous location to current location
        ax.plot([prev_loc[0], new_loc[0]], [prev_loc[1], new_loc[1]], color=cmap(step_i / max_steps))
        # Update new location to previous location
        prev_loc = new_loc
    # Return axes that this was plotted on
    return ax

def plot_sim(env, model, explore, escape):
    """
    Plot walks during explore and escape part of simulation, both the actual locations
    and the locations decoded from representations at each step
    """    
    # Create figure with real and imagined path during explore and escape
    plt.figure(figsize=(10, 5))
    # Subplot 1: explore
    ax = plt.subplot(1, 2, 1)
    # Plot environment
    plot_map(env, np.ones(len(env.locations)), ax=ax);
    # Then plot explore walk on top
    plot_walk(env, [exp_step['location'] for exp_step in explore], ax=ax, cmap='Reds');
    # Then plot decoded locations of explore walk on top
    plot_walk(env, [model.rep_decode(exp_step['representation']) 
                    for exp_step in explore], ax=ax, cmap='Blues');
    plt.title('Real (red) and decoded (blue) exlore walk')
    # Subplot 2: escape
    ax = plt.subplot(1, 2, 2)
    # Plot environment
    plot_map(env, np.ones(len(env.locations)), ax=ax);
    # Then plot explore walk on top
    plot_walk(env, [esc_step['location'] for esc_step in escape], ax=ax, cmap='Reds');
    # Then plot decoded locations of explore walk on top
    plot_walk(env, [model.rep_decode(esc_step['representation']) 
                    for esc_step in escape], ax=ax, cmap='Blues');   
    plt.title('Real (red) and decoded (blue) escape walk')
    
    # Create figure with location and representation during explore and escape
    plt.figure(figsize=(10,5))
    # Plot representation for explore and escape
    for d_i, (data, name) in enumerate(zip([explore, escape],['Explore','Escape'])):
        # See which representation this walk has: only one, or dict?
        rep_keys = data[0]['representation'].keys() \
            if isinstance(data[0]['representation'], dict) else ['representation']
        # First column: locations
        plt.subplot(2, 1 + len(rep_keys), d_i * (1 + len(rep_keys)) + 1)
        plt.imshow(np.array([np.eye(env.n_locations)[step['location']] 
                             for step in data]))
        plt.title(name + ' locations')        
        # Other columns: representation
        for r_i, rep_key in enumerate(rep_keys):
            plt.subplot(2, 1 + len(rep_keys), d_i * (1 + len(rep_keys)) + 2 + r_i)
            plt.imshow(np.array([step[rep_key] if len(rep_keys)==1 
                                 else step['representation'][rep_key] for step in data]))
            plt.title(name + ' ' + rep_key) 

def plot_replay(env, explore, steps_to_plot=None, max_replays=None):
    """
    Plot walks during explore and escape part of simulation, both the actual locations
    and the locations decoded from representations at each step
    """    
    # If steps_to_plot is not provided: just include all steps with replay
    steps_to_plot = [step_i for step_i, step in enumerate(explore) if len(step['replay']) > 0]  \
        if steps_to_plot is None else steps_to_plot
    # If max_replays is not provided: just take the highest number of replays included
    max_replays = max([len(explore[i]['replay']) for i in steps_to_plot]) \
        if max_replays is None else max_replays
    
    # Create figure with all replays in indicated steps
    plt.figure(figsize=(10, 10))    
    for curr_row, curr_step in enumerate(steps_to_plot):
        for curr_col, curr_replay in enumerate(explore[curr_step]['replay']
                                               [:min(len(explore[curr_step]['replay']), max_replays)]):
            # Create subplot for this replay
            ax = plt.subplot(len(steps_to_plot), max_replays, 
                             curr_row * max_replays + curr_col + 1)
            # Plot environment
            plot_map(env, np.ones(len(env.locations)), ax=ax);
            # Plot walk so far
            plot_walk(env, [exp_step['location'] for exp_step in explore[:(curr_step + 1)]], ax=ax, cmap='Reds');
            # Plot current replay
            if len(curr_replay) > 0:
                plot_walk(env, [rep_step['location'] for rep_step in curr_replay], ax=ax, cmap='Greens');
            # Set title
            plt.title('Step ' + str(curr_step) + ', replay ' + str(curr_col))         

def initialise_axes(ax=None):
    """
    Initialise axes for plotting environment with default values (no axis, square aspect ratio, flipped y-direction)
    """
    # If no axes specified: create new figure with new empty axes
    if ax is None:
        plt.figure()
        ax = plt.axes()
    # Set axes limits to 0, 1 as this is how the positions in the environment are setup
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # Force axes to be square to keep proper aspect ratio
    ax.set_aspect(1)
    # Revert y-axes so y position increases downwards (as it usually does in graphics/pixels)
    ax.invert_yaxis()
    # And don't show any axes
    ax.axis('off')
    # Return axes object
    return ax


def action_patch(location_from, location_to, radius, colour):
    """
    Create single action patch: triangle pointing from origin location to transition location
    """
    # Set patch coordinates                    
    if location_to['id'] == location_from['id']:
        # If this is a transition to self: action will point down (y-axis is reversed so pi/2 degrees is up)
        a_dir = np.pi / 2;
        # Set the patch coordinates to point from this location to transition location (shifted upward for self-action)
        xdat = location_from['x'] + radius * np.array([2 * np.cos((a_dir - np.pi / 6)),
                                                       2 * np.cos((a_dir + np.pi / 6)),
                                                       3 * np.cos((a_dir))])
        ydat = location_from['y'] - radius * 3 + radius * np.array([2 * np.sin((a_dir - np.pi / 6)),
                                                                    2 * np.sin((a_dir + np.pi / 6)),
                                                                    3 * np.sin((a_dir))])
    else:
        # This is not a transition to self. Find out the direction between current location and transitioned location
        xvec = location_to['x'] - location_from['x']
        yvec = location_from['y'] - location_to['y']
        a_dir = np.arctan2(xvec * 0 - yvec * 1, xvec * 1 + yvec * 0);
        # Set the patch coordinates to point from this location to transition location
        xdat = location_from['x'] + radius * np.array([2 * np.cos((a_dir - np.pi / 6)),
                                                       2 * np.cos((a_dir + np.pi / 6)),
                                                       3 * np.cos((a_dir))])
        ydat = location_from['y'] + radius * np.array([2 * np.sin((a_dir - np.pi / 6)),
                                                       2 * np.sin((a_dir + np.pi / 6)),
                                                       3 * np.sin((a_dir))])
    # Return action patch for provided data
    return plt.Polygon(np.stack([xdat, ydat], axis=1), color=colour)
