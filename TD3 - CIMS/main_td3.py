# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:38:48 2021

@author: jkenney
"""







import const as c
import numpy as np
from td3_tf2 import Agent
from CIMS_env import Env

#   Are we evaluating performance (True) or learning (False)?
watchPerformance = False
#   Are we restarting learning (True) or continuing learning (False)?
restartLearning = False

if __name__ == '__main__':
    #   Setup particle environment
    env = Env()

    #   Standard deviation on action space during random warmup period
    warmNoise   = c.bot_maxacceleration_meters_per_second_squared / 1.5

    #   Standard deviation of proportional noise during learning period
    learnNoise  = c.bot_maxacceleration_meters_per_second_squared * 0.1

    #   Total number of target find episodes to run
    episodes = 250000
    
    layer_width_1 = 512
    
    batch_size = 2048
    # Adjust to the  number of iterations to spend warming up
    #   Each state->action->state->reward is an iteration
    if restartLearning and (not watchPerformance):
        warmupIterations = 64 * 1024
    else:
        warmupIterations = 4 * 1024

    #   Setup up the number of episodes to evaluate for whether it is we should save the current model
    numberToAverage = 10
    waitToSave      = 20

    if watchPerformance:
        #   We are looking at the existing performance
        #       Don't ever update the model
        updateActorIntervalPerNIterations = 2000000000
        learnNoise = 0.0
    else:
        #   We are learning
        #       Update the model every N iterations
        updateActorIntervalPerNIterations = 2

    #   Largely copied from the Lunar Lander/Bipedal Walker tutorials
    agent = Agent(alpha=0.001, beta=0.001,
                  input_dims=env.observation_space.shape, tau=0.005,
                  env=env, batch_size=batch_size, layer1_size=layer_width_1,
                  n_actions=env.action_space.shape[0], warmup=warmupIterations,
                  warmupNoise=warmNoise, learningNoise=learnNoise,
                  update_actor_interval=updateActorIntervalPerNIterations)

    #   Use the rewards to determine whether we should save the current model
    bestReward = env.reward_range[0]
    reward_history = []

    if (not restartLearning) or watchPerformance:
        #   We are continuing to learn, so load the previous model
        agent.load_models()

    #   This is necessary to use matplotlib's continual plotting of the environment
    line1           = []
    iterationIndex  = 0

    #   Make sure that we actually command the range of actions by tracking the largest action commanded
    maxAction       = 0.0

    for i in range(episodes):
        #   Work through target finding episodes

        #   Setup a new target episode
        observation = env.reset()

        #   Allow conditional exit from the episode
        done = False

        #   Track the accumulated reward in the episode
        episodeReward = 0.0

        #   Debug
        psn = observation[0:2]
        vel = observation[2:3]
        
        while not done:
            #   We are still seeking the target.

            #   Get the next action
            action = agent.choose_action(observation, not watchPerformance)

            #   Get the next state/reward from the selected action
            observation_, reward, done, info = env.step(action)

            #   Plot the current environment conditions
            line1 = env.render(line1, reward, action)

            #   Track the maximum action commanded
            if (np.abs(action[0]) > maxAction) and (iterationIndex > warmupIterations):
                maxAction = np.abs(action[0])
                print("Max Action: ", maxAction)
            if (np.abs(action[1]) > maxAction) and (iterationIndex > warmupIterations):
                maxAction = np.abs(action[1])
                print("Max Action: ", maxAction)

            #   We have completed an iteration
            iterationIndex += 1

            if not watchPerformance:
                #   We are learning, so remember the tuple and learn from it
                agent.remember(observation, action, reward, observation_, done)
                agent.learn(True)

            #   Add the current states reward
            episodeReward += reward

            #   Keep a copy of the current state
            observation = observation_

        #   Store the episode's total reward
        reward_history.append(episodeReward)

        if np.abs(episodeReward) < 0.001:
            print("0 Reward Position: ", psn)
            print("0 Reward Velocity: ", vel)

        if i == 0:
            #   Start with a best reward that is half of the first episode's reward
            bestReward = 0.5 * episodeReward

        #   Find the running average of reward
        rewardRunningAverage = np.mean(reward_history[-numberToAverage:])

        if (rewardRunningAverage > bestReward) and (i > waitToSave) and (iterationIndex > warmupIterations):
            #   The running average is better than our best, and we have waited long enough to have a reasonable
            #   sample of rewards
            bestReward = rewardRunningAverage
            if not watchPerformance:
                #   We are learning, so save the model
                agent.save_models()
        elif iterationIndex < warmupIterations:
            #   We are waiting to finish the warm up period
            print("Iteration:", iterationIndex)
        else:
            if not watchPerformance:
                #   We are learning, move the best reward so that we will eventually beat it.
                #       For this reward function, bots starting further from the target have lower total reward,
                #       so a few episodes starting close to the target will prevent further learning.
                if bestReward > 0.0:
                    #   Positive rewards shrink in magnitude
                    bestReward *= 0.99
                else:
                    #   Negative rewards increase in magnitude
                    bestReward *= 1.01

        print('Episode: ' + str(i)
              + ' Episode reward: ' + str(episodeReward)
              + ' Average reward: ' + str(rewardRunningAverage)
              + ' Best reward: ' + str(bestReward))

    if not watchPerformance:
        #   We are learning, save the last model
        agent.save_models()