#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:11:08 2021

@author: jbakermans
"""

import json
import numpy as np
import copy
import random
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import LineString
import math


class World:
    def __init__(self, env=None, env_type=None):
        # If no environment provided: initialise completely empty environment
        if env is None:
            # Initialise empty environment
            self.init_empty()
        else:
            # Copy provided dictionary
            env = copy.deepcopy(env)
        
        # Create components
        self.env_type, self.components = self.init_components(env_type)
        # Build graph
        self.graph = self.init_graph(self.components)

    def init_empty(self):
        # Create empty environment
        self.xlim = [0, 1]
        self.ylim = [0, 1]
        self.components = {}
        self.graph = {'d': [], 'v': [], 'e': []}
        
    def init_components(self, env_type):
        # Copy env_type dict if provided, or set to empty dict if not
        env_type = {} if env_type is None else dict(env_type)
        # Then initialise world as specified by env_type: dictionary that 
        # specifies world type, which determines which components it contains
        if env_type == {}:
            components = {}
        else:
            if env_type['name'] == 'reward':
                components = self.init_env_reward(env_type)
            elif env_type['name'] == 'wall':
                    components = self.init_env_wall(env_type) 
            elif env_type['name'] == 'walls':
                components = self.init_env_walls(env_type)
            elif env_type['name'] == 's':
                components = self.init_env_s(env_type)
            else:
                print('Component type ' + env_type['name'] + ' not recognised.')
        # Return components
        return env_type, components
                
    def init_graph(self, components):
        # Build graph for components: distances between each component location
        graph = {}
        # Get all border lines that can't be crossed (walls + env boundaries)
        walls = self.get_border_lines(components)
        # Each location gets a node on the graph
        graph['v'] = []
        for name, comp in components.items():
            for l_i, loc in enumerate(comp['locations']):
                # Specify node
                v = {'id': len(graph['v']), 'comp': name, 'loc': loc, 'l_i': l_i,
                     'in': [], 'out': [], 'd': np.inf}
                # Add to list of nodes
                graph['v'].append(v)
        # Initialise adjacency: dist FROM i (row/outer list) TO j (col/inner list)
        graph['a'] = np.eye(len(graph['v']))                        
        # Each pair of nodes gets an edge, with distance or 0 blocked
        graph['e'] = []
        for v_i, v_from in enumerate(graph['v']):
            for v_j, v_to in enumerate(graph['v']):
                # Create line between locations
                line = LineString([(v_from['loc'][0], v_from['loc'][1]),
                                   (v_to['loc'][0], v_to['loc'][1])])
                # Get distance between locations
                d = line.length
                # Find line if line crosses any of the walls, so blocked
                # Line from one corner of wall will intersect with that wall
                # To ignore those, don't include lines that share end points
                # WAS line.crosses, but that doesn't work for overlapping walls
                blocked = any([line.intersects(wall) 
                               and all([c not in wall.coords for c in line.coords])
                               for wall in walls])
                # Specify edge
                e = {'id': len(graph['e']), 'source': v_i, 'target': v_j, 
                     'd': d, 'blocked': blocked}
                # Add edge to edge array
                graph['e'].append(e)
                # And add egde to source and target vertex
                v_from['out'].append(e)
                v_to['in'].append(e)
                # Add distance to adjacency matrix if not blocked
                graph['a'][v_i, v_j] = d if not blocked else 0
        # Do shortest path to find graph distances between all points
        graph['d'] = shortest_path(csgraph=graph['a'], directed=True)
        # Find for each graph node if it is a reward component
        is_reward = [components[v['comp']]['type'] == 'shiny' for v in graph['v']]
        # Finally, find distance to nearest reward component for each node
        for d, v in zip(graph['d'], graph['v']):
            v['d'] = min([dist for dist, reward in zip(d, is_reward) if reward])
        # And return the graph
        return graph
    
    def init_env_reward(self, env_type):
        # Use shiny mechanism to create home in fixed position for easy comparison
        return {'reward': self.init_shiny(pars=env_type)}
    
    def init_env_walls(self, env_type):
        # Start with empty components dictionary
        components = {}
        # Use shiny mechanism to create home in fixed position for easy comparison
        components['reward'] = self.init_shiny()
        # Get environment minimum dim
        min_dim = min([lim[1] - lim[0] for lim in [self.xlim, self.ylim]])
        # Set lims for walls so you can completely block off parts of environment
        wall_xlim = [self.xlim[0] + 0.1*(self.xlim[1] - self.xlim[0]),
                     self.xlim[0] + 0.9*(self.xlim[1] - self.xlim[0])]
        wall_ylim = [self.ylim[0] + 0.1*(self.ylim[1] - self.ylim[0]),
                     self.ylim[0] + 0.9*(self.ylim[1] - self.ylim[0])]        
        # Now create wall, which can go anywhere except shouldn't overlap with reward
        components['obstacle1'] = self.init_wall(
            {'length': min_dim * 0.6, 'xlim': wall_xlim, 'ylim': wall_ylim,
             'avoid': components['reward']['locations']})
        # And let's stick in a second wall that is one step smaller
        components['obstacle2'] = self.init_wall(
            {'length': min_dim * 0.4, 'xlim': wall_xlim, 'ylim': wall_ylim,
             'avoid': components['reward']['locations']})
        # Return components dictionary
        return components
    
    def init_env_wall(self, env_type):
        # Start with empty components dictionary
        components = {}
        # Use shiny mechanism to create home in fixed position for easy comparison
        components['reward'] = self.init_shiny(pars=env_type)
        # Get environment minimum dim
        min_dim = min([lim[1] - lim[0] for lim in [self.xlim, self.ylim]])
        
        # Set center of wall: center of world plus small random offset
        wall_center = [self.xlim[0] + 0.5*(self.xlim[1] - self.xlim[0]),
                       self.ylim[0] + 0.5*(self.ylim[1] - self.ylim[0])]
        wall_center = [w + random.uniform(-0.05, 0.05) * d 
                       for w, d in zip(wall_center, 
                                       [(self.xlim[1] - self.xlim[0]), 
                                        (self.ylim[1] - self.ylim[0])])]
        # Set wall angle: random on 2 pi
        wall_angle = random.uniform(0, 2*np.pi)
        # Set wall length
        wall_length = min_dim * 0.8
        # Now get wall locations: half a length on both sides of center
        wall_locs = [[wall_center[0] + 0.5 * wall_length * np.cos(a), 
                      wall_center[1] + 0.5 * wall_length * np.sin(a)]
                     for a in [wall_angle, wall_angle + np.pi]]
        # Now create wall, which can go anywhere except shouldn't overlap with reward
        components['obstacle1'] = self.init_wall({'length': wall_length,
                                                  'locations': wall_locs})
        # Return components dictionary
        return components    
            
    def init_env_s(self, env_type):
        # Start with empty components dictionary
        components = {}        
        # Use shiny mechanism to create home in fixed position for easy comparison
        components['reward'] = self.init_shiny()
        
        # Choose s-maze orientation: direction of walls
        orientation = random.randint(0,1)
        # Choose offset for both wall areas (1: between 1/6, 1/2; 2: between 1/2, 5/6)
        offset = [1/6, 1/2]; random.shuffle(offset)
        # Set wall length
        l = 0.75 * ([self.xlim, self.ylim][orientation][1] 
                    - [self.xlim, self.ylim][orientation][0])
        # Create empty locations for both walls
        loc_1 = [[[], []], [[], []]]
        loc_2 = [[[], []], [[], []]]
        # Set locations for both walls
        for l_i, curr_loc in enumerate([loc_1, loc_2]):
            # Get current (direction of wall) and other (orthogonal to wall) lims
            currlim = [self.xlim, self.ylim][orientation]
            otherlim = [self.xlim, self.ylim][1-orientation]
            # Get current (direction of wall) and other (orthogonal to wall) coords            
            currcoords = [currlim[0]-0.01, currlim[0] + l] if l_i == 0 \
                else [currlim[1] - l, currlim[1]+0.01]
            othercoords = otherlim[0] + otherlim[1] * \
                random.uniform(offset[l_i], offset[l_i] + 1/3)
            # Assign current and other coordinates to the corresponding wall coords                          
            curr_loc[0][orientation] = currcoords[0]
            curr_loc[1][orientation] = currcoords[1]
            curr_loc[0][1-orientation] = othercoords
            curr_loc[1][1-orientation] = othercoords       
            
        # Now create first s-maze wall
        components['obstacle1'] = self.init_wall({'length': l, 'locations': loc_1})
        # And second s-maze wall
        components['obstacle2'] = self.init_wall({'length': l, 'locations': loc_2})
        # Return components dictionary
        return components

    def init_shiny(self, pars=None):
        # Defaults for root level component dictionary
        defaults = {'type': 'shiny',
                    'locations': None,
                    'xlim': [self.xlim[0] + 0.1*(self.xlim[1] - self.xlim[0]),
                             self.xlim[0] + 0.9*(self.xlim[1] - self.xlim[0])],
                    'ylim': [self.ylim[0] + 0.1*(self.ylim[1] - self.ylim[0]),
                             self.ylim[0] + 0.9*(self.ylim[1] - self.ylim[0])]}        
        # Create shiny component
        new_comp = self.init_pars(pars, defaults)        
        # Set coordinate, if not provided
        new_comp['locations'] = [[random.uniform(start, stop)
                                  for start, stop in [new_comp['xlim'], 
                                                      new_comp['xlim']]]] \
            if new_comp['locations'] is None else new_comp['locations']
        # Return comp
        return new_comp
        
    def init_wall(self, pars=None):
        # Defaults for root level component dictionary
        defaults = {'type': 'wall',
                    'locations': None,
                    'length': min([lim[1] - lim[0] for lim 
                                   in [self.xlim, self.ylim]]) / 2,
                    'xlim': self.xlim,
                    'ylim': self.ylim,
                    'avoid': None,
                    'avoid_radius': max([(self.xlim[1]-self.xlim[0]) * 0.1,
                                         (self.ylim[1]-self.ylim[0]) * 0.1])}        
        # Create wall component
        new_comp = self.init_pars(pars, defaults)
        # Specify locations, if not provided
        while new_comp['locations'] is None:
            # Create first location that is out of range of any env border
            loc_1 = [random.uniform(start, stop)
                     for start, stop in [new_comp['xlim'], new_comp['ylim']]]
            # Make sure that wall fits in some direction
            while all(flatten(
                    [[abs(loc - lim) < new_comp['length'] for lim in lims] 
                     for loc, lims in zip(loc_1, [new_comp['xlim'], new_comp['ylim']])])):
                loc_1 = [random.uniform(start, stop)
                         for start, stop in [new_comp['xlim'], new_comp['ylim']]]
            # Choose second location at random angle from first
            angle = random.uniform(0, 2 * np.pi)
            loc_2 = [l1 + new_comp['length'] * loc 
                     for l1, loc in zip(loc_1, [np.cos(angle), np.sin(angle)])]
            # Make sure wall is fully within environment
            while any([(loc < lim[0]) or (loc > lim[1])
                      for loc, lim in zip(loc_2, [new_comp['xlim'], new_comp['ylim']])]):
                angle = random.uniform(0, 2 * np.pi)
                loc_2 = [l1 + new_comp['length'] * loc 
                         for l1, loc in zip(loc_1, [np.cos(angle), np.sin(angle)])]
            # Set coordinate
            new_comp['locations'] = [loc_1, loc_2]
            # Check if this wall hits any of the locations to be avoided
            if new_comp['avoid'] is not None:
                # Find distance from line to point:
# https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
                if any([abs((loc_2[0] - loc_1[0]) * (loc_1[1] - loc_avoid[1]) -
                            (loc_1[0] - loc_avoid[0]) * (loc_2[1] - loc_1[1]))
                        / np.sqrt((loc_2[0]-loc_1[0])**2 + (loc_2[1] - loc_1[1])**2)
                        < new_comp['avoid_radius'] 
                        for loc_avoid in new_comp['avoid']]):
                    new_comp['locations'] = None
        # Return component
        return new_comp

    def init_pars(self, pars_in=None, defaults=None):
        # If no input parameters provided: set input to empty
        pars_in = {} if pars_in is None else pars_in
        # If no defaults provided: start with empty dictionary
        defaults = {} if defaults is None else defaults
        # Then copy over all values provided, overwriting defaults if existing
        for category in pars_in.keys():
            # If this dict value is a dict itself: copy all it contains
            # (but keep defaults for values it doesn't contain!)
            if isinstance(pars_in[category], dict):
                if category not in defaults:
                    defaults[category] = {}
                for key, val in pars_in[category].items():
                    defaults[category][key] = val
            # If this dict entry is just a value: copy that value
            else:
                defaults[category] = pars_in[category]
        # Return dictionary that combines defaults and provided parameters
        return defaults

    def get_border_lines(self, components=None, include_borders=True):
        # If components not specified: use all env components
        components = self.components if components is None else components
        # Build line segment for each wall
        walls = [LineString([(comp['locations'][0][0], comp['locations'][0][1]), 
                             (comp['locations'][1][0], comp['locations'][1][1])])
                 for comp in components.values() if comp['type'] == 'wall']
        # Also add borders: can't go outside environment
        borders = [LineString([(self.xlim[0], self.ylim[0]), (self.xlim[1], self.ylim[0])]),
                   LineString([(self.xlim[0], self.ylim[0]), (self.xlim[0], self.ylim[1])]),
                   LineString([(self.xlim[0], self.ylim[1]), (self.xlim[1], self.ylim[1])]),
                   LineString([(self.xlim[1], self.ylim[0]), (self.xlim[1], self.ylim[1])])]
        # Return list containing all borders
        return walls + borders if include_borders else walls

    def get_location_target(self, location):
        # Build line segment for each wall, needed to see if locations are blocked
        walls = self.get_border_lines(self.components)
        # Get distance to goal via each graph node
        d = []
        # Distance to goal = distance to graph node + graph node to goal
        for v in self.graph['v']:
            # Create line between locations
            line = LineString([location, (v['loc'][0], v['loc'][1])])
            # See if line is blocked
            blocked = any([line.crosses(wall) for wall in walls])
            # Add distances together if not blocked; else: inf
            d.append(np.inf if blocked else v['d'] + line.length)
        # Choose location with smallest distance
        return [v['loc'] for v in self.graph['v']][d.index(min(d))]
    
    def get_location_direction(self, location):
        # Get target location
        target = self.get_location_target(location)
        # Return angle to location
        return math.atan2(target[1] - location[1], target[0] - location[0])

    def get_grid_locs(self, n_x=20, n_y=20):
        # Get env width and height
        w = self.xlim[1] - self.xlim[0]
        h = self.ylim[1] - self.ylim[0]        
        # Create grid for calculating policy
        grid = np.meshgrid(np.linspace(self.xlim[0] + w / (2*n_x), 
                                       self.xlim[1] - w / (2*n_x), n_x),
                           np.linspace(self.ylim[0] + h / (2*n_x), 
                                       self.ylim[1] - h / (2*n_x), n_y))
        # Reshape grid to flatten it
        grid = [g.reshape(-1) for g in grid]
        # Turn from list of x and y into list of x,y locations
        grid = [loc for loc in zip(*grid)]
        # Return grid
        return grid        

    def get_policy(self, n_x=20, n_y=20):
        # Turn from list of x and y into list of x,y locations
        grid = self.get_grid_locs(n_x, n_y)
        # Calculate direction at each location on grid
        direction = [self.get_location_direction(loc) for loc in grid]
        # Return locations and directions
        return grid, direction
    
    def get_copy(self):
        # Copy now only requires env setup info in env_type
        return World(env_type=self.env_type)
        


def flatten(t):
    return [item for sublist in t for item in sublist]
            