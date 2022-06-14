# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:41:15 2021

@author: jkenney
"""







import const as c
import Bot
import matplotlib.pyplot as plt
import numpy as np
import random

from gym import Env
from gym.spaces import Box

ReachTarget         = False
GoodEnougDistance   = c.goodEnoughDistance

class Env(Env):
    def __init__(self):

        #   Time to find the target
        self.time_limit = c.timeLimit

        #   Action limits
        low = -c.bot_maxacceleration_meters_per_second_squared
        high = c.bot_maxacceleration_meters_per_second_squared
        
        self.action_space = Box(low=np.array([low,low], dtype=np.float32),high=np.array([high,high], dtype=np.float32),shape=(2,),dtype=np.float32)

        #   Observation limite
        obs_shape = np.zeros(5)
        obs_low = np.zeros(obs_shape.shape[0])
        # obs_low.fill(-np.inf)
        obs_low[0] = -1.0
        obs_low[1] = -1.0
        obs_low[2] = -1.0
        obs_low[3] = -1.0
        obs_low[4] = 0.0
        obs_high = np.zeros(obs_shape.shape[0])
        # obs_high.fill(np.inf)
        obs_high[0] = 1.0
        obs_high[1] = 1.0
        obs_high[2] = 1.0
        obs_high[3] = 1.0
        obs_high[4] = 1.0

        self.observation_space = Box(low = obs_low, high = obs_high, shape = obs_shape.shape,dtype=np.float32)

        #   Set up the environment
        self.state = self.reset()
        
        self.bots = []
    
    def reset(self):
        #   Set up the environment:
        #       The position is randomly within a circle - the destination is always (0, 0)
        #           The particular arrangement favors the center more than the periphery.
        radius  = np.random.uniform(0.0, c.grid_length_meters)
        theta   = np.random.uniform(0.0, 2.0 * np.pi)
        psn = np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=np.float32)
        dst = np.array([0.0, 0.0], dtype=np.float32)
        #       The velocity is randomly in a circle - also favors center more than periphery
        radius  = np.random.uniform(0.0, c.bot_maxspeed_deck_meters_per_second)
        theta   = np.random.uniform(0.0, 2.0 * np.pi)
        velX    = radius * np.cos(theta)
        velY    = radius * np.sin(theta)

        #   The robot
        self.bot = Bot.Bot(psn, dst)

        #   The target (always (0,0) currently)
        self.dock = self.bot.destination

        #   The vector to the target
        del_x = self.dock[0] - self.bot.position[0]
        del_y = self.dock[1] - self.bot.position[1]

        #   The bot's copy of velocity
        self.bot.velocity[0]    = velX
        self.bot.velocity[1]    = velY

        #   The observation space is;
        #       - vector to the target
        #       - bot velocity
        #       - proportion of time remaining
        self.state = np.array([del_x / c.BoundaryRadius,
                               del_y / c.BoundaryRadius,
                               self.bot.velocity[0] / c.bot_maxspeed_deck_meters_per_second,
                               self.bot.velocity[1] / c.bot_maxspeed_deck_meters_per_second,
                               self.bot.t / self.time_limit], dtype=np.float32)
        
        return self.state

    def step(self, action):
        #   Perform one simulation time step
        #       Our current speed
        velocityMagnitude   = np.linalg.norm(self.bot.velocity)
        #       Our current direction to the target and its magnitude
        goodDirection       = self.dock - self.bot.position
        currentDistance     = np.linalg.norm(goodDirection)

        #   The distance a bot can travel in one time step
        if ReachTarget:
            #   Train to reach the target
            reachedTargetDistance  = self.bot.maxspeed * self.bot.dt
        else:
            #   Train to get close to the target
            reachedTargetDistance  = GoodEnougDistance
            

        #   Current position and velocity
        rewardPosition  = np.array(self.bot.position)
        rewardVelocity  = np.array(self.bot.velocity)

        #   Distance required to come to a full stop when traveling at maximum speed
        stopDistance = (c.bot_maxspeed_deck_meters_per_second * c.bot_maxspeed_deck_meters_per_second
                        / (2.0 * c.bot_maxacceleration_meters_per_second_squared))

        #   Calculate the effect of the action on the bot
        self.bot.action(action)

        #   Calculate the distance from the target if we continue the present action for c.RewardDt time
        rewardAcceleration  = np.array(self.bot.acceleration)
        rewardPosition      += (rewardVelocity * c.RewardDt) + (rewardAcceleration * c.RewardDt * c.RewardDt / 2.0)
        rewardDistance      = np.linalg.norm(self.dock - rewardPosition)

        #   Calculate the new vector to the target
        del_x = self.dock[0] - self.bot.position[0]
        del_y = self.dock[1] - self.bot.position[1]

        #   Update the observation state
        self.state = np.array([del_x / c.BoundaryRadius,
                               del_y / c.BoundaryRadius,
                               self.bot.velocity[0] / c.bot_maxspeed_deck_meters_per_second,
                               self.bot.velocity[1] / c.bot_maxspeed_deck_meters_per_second,
                               self.bot.t/self.time_limit], dtype=np.float32)

        #   We don't believe we have crashed
        crashed = False

        if currentDistance <= reachedTargetDistance:
            #   We are within a minimum distance to the target - we have docked
            docked = True

            #   The maximum positive reward  is +2.0, so make the reward for finishing at least that large
            reward = 6.0 * (self.time_limit / self.bot.dt)\
                     * ((c.RewardCircleRadius / reachedTargetDistance)
                        + c.RewardCircleRadius - reachedTargetDistance - 1)

            if ReachTarget:
                #   Encourage reaching the target at zero velocity
                #       At this distance, stoppableVelocity is the larges value which will enable use to stopped
                stoppableVelocity = np.max([np.sqrt(2.0 * c.bot_maxacceleration_meters_per_second_squared * reachedTargetDistance),
                                            0.001])
                #   Penalize the robot for having leftover speed
                velocityFactor = 1.0 - (0.5 * np.sqrt(np.max([velocityMagnitude / stoppableVelocity, 4])))
            else:
                #   Just get close to the target
                velocityFactor = 1.0

            reward *= velocityFactor

        elif currentDistance > c.BoundaryRadius:
            #   Stop when we are too far away from the target
            #       Make sure the penalty is larger than from accumulated time penalties
            reward  = 6.0 * (self.time_limit / self.bot.dt)\
                     * ((c.RewardCircleRadius / c.BoundaryRadius)
                        + c.RewardCircleRadius - c.BoundaryRadius)
            docked  = False
            crashed = True
        else:
            #   Make the reward/penalty pay us for being close to the target.
            reward = (c.RewardCircleRadius / c.BoundaryRadius) + c.RewardCircleRadius - currentDistance

            velocityFactor = 1.0
            if ReachTarget:
                #   Encourage slowing down if we want to reach our target
                if currentDistance < stopDistance:
                    #   If we are within the stopping distance for our maximum velocity , make sure that we are slowing down
                    stoppableVelocity = np.max([np.sqrt(2.0 * c.bot_maxacceleration_meters_per_second_squared * currentDistance),
                                                0.001])
                    velocityFactor = 1.0 - (0.5 * np.sqrt(np.max([velocityMagnitude / stoppableVelocity, 4])))

            reward *= velocityFactor

            # if velocityMagnitude > 0.0:
            #     #   Give a reward for the extent that the velocity points toward the target
            #     reward += np.dot(rewardVelocity, goodDirection) / (velocityMagnitude * currentDistance)

            docked = False

        #   Penalize system for taking too long to finish.
        reward *= (1.0 - (0.5 * np.sign(reward) * self.bot.t / self.time_limit))

        #   Check the elapsed time
        if self.bot.t >= self.time_limit:
            out_of_time = True
        else:
            out_of_time = False
        
        info = {}

        #   Check ending condition
        if docked or crashed or out_of_time:
            done = True
        else:
            done = False

        return self.state, reward, done, info

    def render(self, line1, reward, action):
        #   Give some environmental status
        #       matplotlib is pretty restrictive, I have got it to work with one line indicating:
        #           -   the reward
        #           -   the position
        #           -   the velocity
        #           -   the acceleration

        x = [20.0 * action[0],                                              #   the acceleration * 20
             0.0,                                                           #   return to the origin
             0.0,                                                           #   The reward * 10 (in the y direction)
             0.0,                                                           #   return to the origin
             self.bot.position[0] - self.dock[0],                           #   the position
             self.bot.velocity[0] + self.bot.position[0] - self.dock[0]]    #   the velocity from the position

        y = [20.0 * action[1],                                              #   the acceleration * 20
             0.0,                                                           #   return to the origin
             ( reward * 1.0 ),                                             #   the reward * 10 (in the y-direction)
             0.0,                                                           #   return to the origin
             self.bot.position[1] - self.dock[1],                           #   the position
             self.bot.velocity[1] + self.bot.position[1] - self.dock[1]]    #   the velocity from the position

        if line1 == []:
            #   First pass through, set up the plot elements
            plt.ion()
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            line1, = ax.plot(x, y, 'k-+')
            ax.set_xlim([-c.grid_length_meters/1, c.grid_length_meters/1])
            ax.set_ylim([-c.grid_length_meters/1, c.grid_length_meters/1])
            ax.grid()
            plt.show()

        #   subsequent passes, update the data in the line
        line1.set_xdata(x)
        line1.set_ydata(y)
        plt.pause(0.01)

        #   return the line
        return line1



