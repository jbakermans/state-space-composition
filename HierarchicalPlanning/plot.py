#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:42:24 2021

@author: jbakermans
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

def plot_map(env_dict, component=None, values=None, ax=None, min_val=None, max_val=None, 
             num_cols=100, location_cm='viridis', action_cm='Pastel1', 
             do_plot_actions=False, shape='circle', radius=None):
    """
    Plot map of environment where each location is coloured according to array of values with length of #locations
    """
    # If values are not provided, set them to ones everywhere
    values = np.ones(env_dict['n_locations']) if values is None else values
    # If min_val and max_val are not specified: take the minimum and maximum of the supplied values
    min_val = np.min(values) if min_val is None else min_val
    max_val = np.max(values) if max_val is None else max_val
    # Create color map for locations: colour given by value input
    location_cm = cm.get_cmap(location_cm, num_cols)
    # Create color map for actions: colour given by action index
    action_cm = cm.get_cmap(action_cm, env_dict['n_actions'])
    # Calculate colour corresponding to each value
    plotvals = np.floor((values - min_val) / (max_val - min_val) * num_cols) if max_val != min_val \
        else np.ones(values.shape)
    # Calculate radius of location circles based on how many nodes there are
    radius = 2 * (0.01 + 1 / (10 * np.sqrt(env_dict['n_locations']))) if radius is None else radius
    # Initialise empty axis
    ax = initialise_axes(ax)
    # Create empty list of location patches and action patches
    location_patches, action_patches = [], []
    # Now start drawing locations and actions
    for i, location in enumerate(env_dict['locations']):
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
                    locations_to = [env_dict['locations'][loc_to]
                                    for loc_to in np.where(np.array(action['transition']) > 0)[0]]
                    # Create an action patch for each possible transition for this action
                    for loc_to in locations_to:
                        action_patches.append(action_patch(location, loc_to, radius, action_cm(action['id'])))
    # Get components: if not specified, use all
    component = env_dict['components'].keys() if component is None else component
    # After drawing all locations, add component patches
    for location in env_dict['locations']:
        # Create colour for each component
        comp_cm = cm.get_cmap('tab10', 10)
        # Run through components and highlight included locations for each
        for comp_i, comp in enumerate(component):
            if location['components'][comp]['in_comp']:
                # Create square patch for location
                location_patches.append(
                    plt.Rectangle((location['x'] - radius / 2, location['y'] - radius / 2), radius, radius,
                                  linewidth=3, facecolor='none', edgecolor=comp_cm(comp_i))
                    if shape == 'square'
                    else plt.Circle((location['x'], location['y']), radius,
                                    linewidth=3, facecolor='none', edgecolor=comp_cm(comp_i)))
    # Add patches to axes
    for patch in location_patches + action_patches:
        ax.add_patch(patch)
    # Return axes for further use
    return ax

def plot_grid(env_dict, locations=None, ax=None, do_plot_actions=True, components=None,
              values=None, min_val=None, max_val=None, val_cm='viridis', loc_default_col=(1, 1, 1)):
    """ Restricted but neat plotting of square grid graphs """
    val_cm = cm.get_cmap(val_cm)
    # If values are provided: scale them to be between min and max
    min_val = np.min(values) if min_val is None else min_val
    max_val = np.max(values) if max_val is None else max_val    
    plot_vals = None if values is None \
        else np.maximum(0, np.minimum(1, (values - min_val) / (max_val - min_val)))
    # Set locations as optimal policy from environment if not provided
    if locations is None:
        locations = copy.deepcopy(env_dict['locations'])
    else:
        locations = copy.deepcopy(locations)
    # Any locations with multiple non-zero action probabilities: choose random one
    for location in locations:
        # Find highest action probability
        max_probability = max([action['probability'] for action in location['actions']])
        # Collect all non-zero actions
        non_zero_actions = [action for action in location['actions']
                            if action['probability'] == max_probability]
        # Set them all to zero
        for action in location['actions']:
            action['probability'] = 0
        # Then choose one random one and set its probability to 1
        np.random.choice(non_zero_actions)['probability'] = 1
    # Set components to empty if not provided
    components = [] if components is None else components
    # Create color map for locations: colour given by components, if provided
    location_cm = {comp: cm.get_cmap('tab10', 10)(c_i) for c_i, comp in enumerate(components)}
    # Calculate location radius: minimum distance between two nodes
    radius = min([abs(l1[i] - l2[i]) for l1 in locations for l2 in locations 
                  for i in ['x','y'] if abs(l1[i] - l2[i]) > 0]) / 2
    # Initialise empty axis
    ax = initialise_axes(ax)
    location_patches, action_patches = [], []
    # Draw locations
    for i, location in enumerate(locations):
        # Find if this location belongs to any of the requested components
        loc_comp = [key for key, val in location['components'].items()
                    if val['in_comp'] and key in location_cm.keys()]
        # Location colour is set to first matching component, if there is any; 
        # else, the colour provided by input value, if there is any; else, white
        loc_col = location_cm[loc_comp[0]] if len(loc_comp) > 0 else (
            [1, 1, 1] if values is None else val_cm(plot_vals[i]))
        # Create square patch for location
        location_patches.append(
            plt.Rectangle((location['x'] - radius, location['y'] - radius), radius*2, radius*2,
                          linewidth=1, facecolor=loc_col, edgecolor=[0, 0, 0]))
        # And create action patches, but not for walls or rewards
        if len(loc_comp) == 0:
            for a, action in enumerate(location['actions']):
                # Only draw patch if action probability is larger than 0
                if action['probability'] > 0:
                    # Find where this action takes you
                    locations_to = [locations[loc_to] for loc_to in np.where(np.array(action['transition']) > 0)[0]]
                    # Create an action patch for each possible transition for this action
                    for loc_to in locations_to:
                        action_patches.append(action_patch(location, loc_to, radius, (0.5, 0.5, 0.5), gap=False))
    # Add patches to axes
    for patch in (location_patches + (action_patches if do_plot_actions else [])):
        ax.add_patch(patch)
    # Return axes for further use
    return ax       


def plot_actions(locations, field='probability', ax=None, min_val=None, max_val=None,
                 num_cols=100, action_cm='viridis', components=None, loc_default_col=(0, 0, 0),
                 action_radius_factor=1, location_radius_factor=1):
    """
    Plot map of environment where all actions receive a colour according to field (e.g. 'probability' to plot policy)
    """
    # If min_val and max_val are not specified: take the minimum and maximum of the supplied values
    min_val = min([action[field] for location in locations for action in location['actions']]) \
        if min_val is None else min_val
    max_val = max([action[field] for location in locations for action in location['actions']]) \
        if max_val is None else max_val
    # Create color map for actions: colour given by value input
    action_cm = cm.get_cmap(action_cm, num_cols)
    # Set components to empty if not provided
    components = [] if components is None else components
    # Create color map for locations: colour given by components, if provided
    location_cm = {comp: cm.get_cmap('tab10', 10)(c_i) for c_i, comp in enumerate(components)}
    # Calculate radius of location circles based on how many nodes there are
    radius = 1 * (0.01 + 1 / (10 * np.sqrt(len(locations))))
    a_rad = radius * action_radius_factor
    l_rad = radius * location_radius_factor
    # Initialise empty axis
    ax = initialise_axes(ax)
    # Create empty list of location patches and action patches
    location_patches, action_patches = [], []
    # Now start drawing locations and actions
    for i, location in enumerate(locations):
        # Find if this location belongs to any of the requested components
        loc_comp = [key for key, val in location['components'].items()
                    if val['in_comp'] and key in components]
        # Location colour is set to first matching component, if there is any
        loc_col = location_cm[loc_comp[0]] if len(loc_comp) > 0 else loc_default_col
        # Create circle patch for location
        location_patches.append(plt.Circle((location['x'], location['y']), l_rad, color=loc_col))
        # And create action patches
        for a, action in enumerate(location['actions']):
            # Only draw patch if action probability is larger than 0
            if action['probability'] > 0:
                # Calculate colour for this action from colour map
                action_colour = action_cm(int(np.floor((action[field] - min_val) / (max_val - min_val) * num_cols))) \
                    if max_val != min_val else action_cm(0)
                # Find where this action takes you
                locations_to = [locations[loc_to] for loc_to in np.where(np.array(action['transition']) > 0)[0]]
                # Create an action patch for each possible transition for this action
                for loc_to in locations_to:
                    action_patches.append(action_patch(location, loc_to, a_rad, action_colour))
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

def plot_env(env_dict, ax=None, do_plot_actions=True):
    # Copy env_dict: this function is going to change it, but leave input unchanged
    env_dict = copy.deepcopy(env_dict)
    # Plot high level actions
    ax = plot_actions(env_dict['locations'], field='id', 
                      action_cm='tab10', num_cols=10, min_val=0, max_val=9,
                      components=['dummy', 'dummy', 'reward'],
                      location_radius_factor=2, ax=ax)
    # Find all circle patches and make them much much lighter
    for p in ax.patches:
        # Find if path is circle
        if isinstance(p, plt.Circle):
            # Get patch facecolor r,g,b as numpy array
            col = np.array(p.get_facecolor()[:-1])
            # Make it much lighter
            col = col + (1 - col)*0.8
            # And update patch colour
            p.set_facecolor(list(col) + [1])

    # Plot blockade, if available and specified
    if 'components' in env_dict.keys():
        block = [c for c in env_dict['components'].values() if c['type'] == 'block']
        if len(block) > 0 and all([l is not None for l in block[0]['locations']]):
            l1, l2 = block[0]['locations']
            plt.scatter([0.5 * (env_dict['locations'][l1]['x'] + env_dict['locations'][l2]['x'])],
                        [0.5 * (env_dict['locations'][l1]['y'] + env_dict['locations'][l2]['y'])],
                        c='black', marker='x')
    
    # Get low-level scaling from number of high-level locations
    scale = 3*(0.01 + 1 / (10 * np.sqrt(len(env_dict['locations']))))    
    # Then plot low-level actions
    for l_i, loc_high in enumerate(env_dict['locations']):
        # Set location x, y to be centred around high-level location,
        # scaled by high-level radius
        for loc_low in loc_high['env']['locations']:
            loc_low['x'] = loc_high['x'] + scale * (loc_low['x'] - 0.5)
            loc_low['y'] = loc_high['y'] + scale * (loc_low['y'] - 0.5)            
        plot_grid(loc_high['env'], ax=ax, 
                  components=['door0', 'door1', 'reward'], 
                  do_plot_actions=do_plot_actions[l_i] if isinstance(do_plot_actions, list) else do_plot_actions)
        # plot_actions(loc_high['env']['locations'], ax=ax, components=['door0', 'door1', 'reward'],
        #              action_radius_factor=scale, location_radius_factor=scale)

def plot_model(model):
    plt.figure()
    cols = len(model.env['env'].components.keys()) + 1
    rows = 2
    # Plot components
    for i, comp in enumerate(model.env['env'].components.keys()):
        ax = plt.subplot(rows, cols, i + 1)
        plot_map(model.env['env'], component=comp,
                 values=np.array([l['components'][comp]['weight'] 
                                  if 'weight' in l['components'][comp].keys()
                                  else 0
                                  for l in model.env['env'].locations]), 
                 min_val=0, max_val=1, ax=ax)
        plt.title(comp + ' weights')        
        ax = plt.subplot(rows, cols, i + cols + 1)
        plot_actions(model.env['env'].get_policy(name=comp, in_place=False), ax=ax)
        plt.title(comp + ' actions')
    # Plot full environment
    ax = plt.subplot(rows, cols, cols)
    plot_map(model.env['env'], ax=ax)
    plt.title('compositional locations')
    ax = plt.subplot(rows, cols, cols * 2)
    plot_actions(model.env['env'].get_policy(in_place=False), ax=ax)
    plt.title('compositional actions')
                

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


def action_patch(location_from, location_to, radius, colour, gap=True):
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
        # Set location with gap: not from the location centre, but a bit forward in action direction
        if gap:
            # Set the patch coordinates to point from this location to transition location
            xdat = location_from['x'] + radius * np.array([2 * np.cos((a_dir - np.pi / 6)),
                                                           2 * np.cos((a_dir + np.pi / 6)),
                                                           3 * np.cos((a_dir))])
            ydat = location_from['y'] + radius * np.array([2 * np.sin((a_dir - np.pi / 6)),
                                                           2 * np.sin((a_dir + np.pi / 6)),
                                                           3 * np.sin((a_dir))])
        else:
            # No gap: draw arrow right in the location centre
            xdat = location_from['x'] + radius * np.array([np.cos((a_dir - 2 * np.pi / 3)),
                                                           np.cos((a_dir + 2 * np.pi / 3)),
                                                           np.cos((a_dir))])
            ydat = location_from['y'] + radius * np.array([np.sin((a_dir - 2 * np.pi / 3)),
                                                           np.sin((a_dir + 2 * np.pi / 3)),
                                                           np.sin((a_dir))])            
    # Return action patch for provided data
    return plt.Polygon(np.stack([xdat, ydat], axis=1), color=colour)
