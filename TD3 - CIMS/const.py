# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:40:59 2020

@author: jkenney
"""
import numpy

#   Length of area of interest
grid_length_meters = 121.92

#   Width of the area of interest
grid_width_meters = 6.096

#   Bot geometry
bot_length_meters = 1.347216
bot_width_meters = 0.762

turn_time = 4.0
transition_time = 4.0
case_time = 20.0

#   I had training success with
#       RewardDt            = 1.0
#       RewardCircleRadius  = 25.0
#       dt                  = 1.0
#       timeLimit           = 100.0
#       goodEnoughDistance  = 5.0
#   Time to look ahead for calculating reward
RewardDt            = 1.0 * 0.0
#   The circle where the distance reward goes from negative to positive
RewardCircleRadius  = 25.0
BoundaryRadius      = 125.0
#   The simulation time step
dt          = 0.1
#   The simulation time limit
timeLimit   = 200.0

#   The distance to target to be considered "close enough"
goodEnoughDistance  = 2.5

#   Maximum speed
bot_maxspeed_deck_meters_per_second = 5

#   Maximum acceleration
bot_maxacceleration_meters_per_second_squared = 2
