#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:24:53 2021

@author: jbakermans
"""

import world
import model
import plot

# Create environment from json file in envs directory
env = world.World('./envs/10x10.json')
# Create place model. Pass parameter dict to overwrite default params
hybrid = model.Hybrid(env, {'model':{'rep_transition_noise': {'place': 0.6, 'ovc': 0.6}},
                            'sim':{'print': True,
                                   'explore_steps': 25,
                                   'replay_n': 1,
                                   'replay_interval': 4}})
                          
                          # {'model':{'rep_transition_noise': {'ovc': 0.4, 'place': 0}}})
# Run simulation
explore, escape = hybrid.simulate()

# Plot walks
plot.plot_sim(env, hybrid, explore, escape)
# Select replays to plot: first select all steps that have replay at all
steps_to_plot = [step_i for step_i, step in enumerate(explore) if len(step['replay']) > 0]
# Then select just the first two and the last two
steps_to_plot = steps_to_plot[:2] + steps_to_plot[-2:]
# And plot replays
plot.plot_replay(env, explore, steps_to_plot=steps_to_plot, max_replays=4)