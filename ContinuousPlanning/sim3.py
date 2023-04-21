#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:14:57 2021

@author: jbakermans
"""

# This should probably be a Jupyter notebook
import world
import model
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle

# Get experiment, either from running simulations or loading existing results
def get_exp(curr_exp, overwrite=False):
    # Load results if they were already run, else run the experiment
    if os.path.isfile(curr_exp['filename']) and not overwrite:
        with open(curr_exp['filename'], 'rb') as f:
            load_exp = pickle.load(f)
            load_exp['name'] = curr_exp['name']
            load_exp['model_names'] = curr_exp['model_names']
            curr_exp = load_exp
    else:
        # Now run experiment
        curr_exp['results'] = run_exp(curr_exp)
        # Save result
        with open(curr_exp['filename'], 'wb') as f:
            pickle.dump(curr_exp, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Plot result
    plot_exp(curr_exp)
    # Return results for provided experiment
    return curr_exp

# Run simulations required for heatmap experiments
def run_exp(experiment):
    # Intialise results
    results = []    
    for m_i, curr_exp in enumerate(experiment['experiments']):
        # Start with empty results matrix
        curr_result = np.zeros((N, len(experiment[experiment['y']]), 
                                len(experiment[experiment['x']])))
        for x_i, curr_x in enumerate(experiment[experiment['x']]):
            for y_i, curr_y in enumerate(experiment[experiment['y']]):
                for i in range(experiment['N']):                
                    # Find current noise level, walk length, number of replays
                    curr_params = {'noise': [], 'walk': [], 'replay': []}
                    for par in curr_params.keys():
                        curr_params[par] = curr_x if experiment['x'] == par \
                            else curr_y if experiment['y'] == par \
                                else experiment[par]                                
                    # Reset model with correct noise level
                    m.reset(env, 
                            {'experiment': curr_exp,
                             'noise': curr_params['noise'] * np.array([0.5 * 2*np.pi, 0.05]),
                             'print': False,
                             'explore_max_steps': curr_params['walk'],
                             'replay_n': curr_params['replay'],
                             'replay_interval': 4,
                             'replay_reverse': True if curr_exp==5 else False})
                    # Run simulation
                    explore, escape = m.simulate()
                    # Collect results
                    if len(escape) > 0:
                        curr_result[i, y_i, x_i] = \
                            max(0, m.get_goal_dist(escape[-1]['location']) - 
                                m.pars['line_of_sight'])
                print('Finished model ' + experiment['model_names'][m_i] 
                      + ', ' + experiment['x'] + ' = '+ str(curr_x) 
                      + ', ' + experiment['y'] + ' = ' + str(curr_y))
        # Add results to results list
        results = results + [curr_result]
    # Finally, return all results in this experiment
    return results

# Plot resulting heatmap for simulations for Figure 5e,f
def plot_exp(experiment):
    # Get upper limit for plotting
    vmax = max([np.max(np.mean(curr_result, axis=0)) for curr_result in experiment['results']])
                
    # Plot experiment
    fig = plt.figure(figsize=(5, 2.6))
    for m_i, (curr_model, curr_result) in enumerate(zip(experiment['model_names'], experiment['results'])):        
        ax = plt.subplot(1, len(experiment['model_names']), m_i + 1)
        im = ax.imshow(np.mean(curr_result, axis=0),origin='lower', vmin=0, vmax=np.ceil(vmax*10)/10)
        
        ax.set_xticks([i for i in range(len(experiment[experiment['x']]))])
        ax.set_xticklabels([str(round(x, 2)) if i%3==0 else ''
                            for i, x in enumerate(experiment[experiment['x']])])
        ax.set_xlabel(experiment['x'])
        
        ax.set_yticks([i for i in range(len(experiment[experiment['y']]))])
        ax.set_yticklabels([str(round(i, 2)) for i in experiment[experiment['y']]])
        if m_i < 1:
            ax.set_ylabel(experiment['y'])        
        
        ax.set_title(curr_model)        
    fig.suptitle(experiment['name'])
    # Add colourbar
    #fig.tight_layout()
    fig.subplots_adjust(right=0.875)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([i/10 for i in range(int(np.ceil(vmax*10+1)))])

# Decide whether to overwrite existing runs, or load them if encountered
overwrite = False
# Set number of simulations to run
N = 10
# Set noise levels
noise_n = 10
noise_start = 0
noise_stop = 1.0
noise = [noise_start + i * (noise_stop-noise_start)/(noise_n-1) for i in range(noise_n)]
# Set number of replays
replay_n = 10
replay_start = 0
replay_stop = replay_start + 1 * (replay_n - 1)
replay = [int(replay_start + i*(replay_stop-replay_start)/(replay_n-1)) for i in range(replay_n)]
# Set walk length
walk_n = 10
walk_start = 10
walk_stop = walk_start + 10 * (walk_n - 1)
walk = [int(walk_start + i*(walk_stop-walk_start)/(walk_n-1)) for i in range(walk_n)]

# Create environment for all simulations
env = world.World(env_type={'name': 'reward', 'locations': [[0.5, 0.2]]})
# Set fixed home location
env.components['reward']['locations'] = [[0.5, 0.2]]
# Create model
m = model.Model(env)
# Set model experiment numbers
exp = [3, 4, 5]
names = ['PathIntegrate', 'BellmanReplay', 'MemoryReplay']

# Create lists to keep track of experiments
sim_results = []
sim_models = []
sim_model_names = []
sim_names = []

# Create list of all experiments
experiments = []

# Create experiment 2 dictionary
curr_exp = {}
curr_exp['name'] = 'a. Replay reduces path integration noise'
curr_exp['experiments'] = [exp[0], exp[2]]
curr_exp['model_names'] = [names[0], names[2]]
curr_exp['N'] = N
curr_exp['x'] = 'noise'
curr_exp['y'] = 'walk'
curr_exp['noise'] = noise
curr_exp['walk'] = walk
curr_exp['replay'] = 5
curr_exp['filename'] = './sim/exp6/' + str(len(experiments) + 1) + '_' \
    + ''.join([curr_name for curr_name in curr_exp['model_names']]) + '_' \
        + curr_exp['x'].capitalize() + '_' \
            + curr_exp['y'].capitalize() + '_' \
                + '.pkl' 
# Change some values after filename so they appear nicer on plot
# curr_exp['model_names'] = ['Path integrator', 'Memory replay']
# curr_exp['x'] = 'Transition noise'
# curr_exp['y'] = 'Walk length'

# Run experiment (or load if results already exist) and add to experiments list
experiments = experiments + [get_exp(curr_exp, overwrite)]

# Create experiment 4 dictionary
curr_exp = {}
curr_exp['name'] = 'b. Fewer experiences needed than Bellman replay'
curr_exp['experiments'] = [exp[1], exp[2]]
curr_exp['model_names'] = [names[1], names[2]]
curr_exp['N'] = N
curr_exp['x'] = 'noise'
curr_exp['y'] = 'replay'
curr_exp['noise'] = noise
curr_exp['walk'] = 25
curr_exp['replay'] = replay
curr_exp['filename'] = './sim/exp6/' + str(len(experiments) + 1) + '_' \
    + ''.join([curr_name for curr_name in curr_exp['model_names']]) + '_' \
        + curr_exp['x'].capitalize() + '_' \
            + curr_exp['y'].capitalize() + '_' \
                + '.pkl' 
# Change some values after filename so they appear nicer on plot
# curr_exp['model_names'] = ['Bellman replay', 'Memory replay']
# curr_exp['x'] = 'Transition noise'
# curr_exp['y'] = 'Number of replays'

# Run experiment (or load if results already exist) and add to experiments list
experiments = experiments + [get_exp(curr_exp, overwrite)]