#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 17:59:18 2021

@author: jbakermans
"""
import world
import model
import plot

# Create environment from json file in envs directory
env = world.World('./envs/10x10.json')
# Create place model. Pass parameter dict to overwrite default params
ovc = model.OVC(env, {'model':{'rep_transition_noise': 0.4}})
# Run simulation
explore, escape = ovc.simulate()

# Plot walks
plot.plot_sim(env, ovc, explore, escape)
