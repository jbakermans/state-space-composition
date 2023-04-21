#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 17:56:53 2021

@author: jbakermans
"""
import world
import model
import plot
import copy
import matplotlib.pyplot as plt
import numpy as np

# Create environment from json file in envs directory
env = world.World('./envs/10x10.json')
# Create place model. Pass parameter dict to overwrite default params
place = model.Place(env, {'model':{'rep_transition_noise': 0}, 
                          'sim':{'explore_steps': 25,
                                 'replay_interval': 10,
                                 'replay_n': 1}})
# Run simulation
explore, escape = place.simulate()

# Plot walks
plot.plot_sim(env, place, explore, escape)
# Select replays to plot: first select all steps that have replay at all
steps_to_plot = [step_i for step_i, step in enumerate(explore) if len(step['replay']) > 0]
# Then select just the first two and the last two
steps_to_plot = steps_to_plot[:2] + steps_to_plot[-2:]
# And plot replays
plot.plot_replay(env, explore, steps_to_plot=steps_to_plot, max_replays=4)

# Just for plotting: make map of final q-vals
final_pol = copy.deepcopy(env.locations)
for l_i, location in enumerate(final_pol):
    for a_i, action in enumerate(location['actions']):
        action['probability'] = place.rep_policy(place.rep_encode(l_i))[a_i]
# Then plot learned policy
ax = plot.plot_map(env, np.ones(len(env.locations)));
plot.plot_actions(final_pol, ax=ax);