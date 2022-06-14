# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:27:28 2021

@author: jkenney
"""







import numpy as np
import const as c

# Class to define bot attributes
class Bot(object):
    def __init__(self,beginning,destination):

        #   target
        self.destination = destination

        #   initial position
        self.beginning = beginning

        #   current position
        self.position = self.beginning

        #   velocity and acceleration
        self.velocity = np.zeros(2, dtype=np.float32)
        self.acceleration = np.zeros(2, dtype=np.float32)

        #   current time
        self.t = 0

        #   time step
        self.dt = c.dt
        
    #   How fast is the bot allowed to go?
        self.maxspeed_deck = c.bot_maxspeed_deck_meters_per_second
        self.maxspeed = self.maxspeed_deck
        self.maxacceleration = c.bot_maxacceleration_meters_per_second_squared
        
    #   Misc. lists and checks for various operations like waiting, completing a task, turning, etc...
        self.bot_dimensions = (c.bot_length_meters, c.bot_width_meters)
        
    #   Time, in seconds, to complete an opteration: turning, moving from deck to lift/aisle, picking/placing a case
        self.turn_time = int(c.turn_time/self.dt)
        self.transition_time = int(c.transition_time/self.dt)
        self.case_time = int(c.case_time/self.dt)
    
    #   Definition to update bot's position, velocity, and acceleration
    def update(self):
                    
        #   Updates velocity
        self.position       = np.array(self.position)
        self.velocity       = np.array(self.velocity)
        self.acceleration   = np.array(self.acceleration)
        oldVelocity         = self.velocity
        self.velocity       += (self.acceleration * self.dt)
        
        #   Limiting velocity to maxspeed
        mag_v = np.linalg.norm(self.velocity)
        if mag_v > self.maxspeed:
            scale = self.maxspeed / mag_v
            self.velocity       = self.velocity * scale
            self.acceleration   = (self.velocity - oldVelocity) / self.dt
        
        #   Updates position
        self.position += (oldVelocity * self.dt) + (self.acceleration * self.dt * self.dt / 2.0)
            
        #   Updates bot's clock
        self.t += self.dt
                
    def action(self, action):
        #   Ensure the action is compatible with the bot dynamics, and then update the bot state
        self.acceleration = np.array(action)
        accelerationMagnitude = np.linalg.norm(self.acceleration)
        
        if accelerationMagnitude > self.maxacceleration:
            self.acceleration = self.maxacceleration * action / accelerationMagnitude
            
        self.update()