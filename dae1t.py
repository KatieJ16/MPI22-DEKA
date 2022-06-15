# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:26:21 2022

@author: David A. Edwards adapted from Katherine Johnston's code.
"""

# Code dae1t.  This code computes a single trajectory for a single value of lambda.   Moreover, it stores the iterates for x, v, and a.

# Import the pacakges we need.
import numpy as np
import matplotlib.pyplot as plt

    
# Just like in Pascal, functions have to be defined ahead of any script.  So we just define a main function which is the script, and call it at the end.

def main():
    
    # Variables:
        # amax: maximum acceleration of robot
        # dt: time step
        # epsilon: target radius
        # i: step variable
        # lam_list: list of lambda values to test
        # n_steps: number of steps to run code
        # num: looping variabe for simulation
        # rstop: tolerance for radius
        # v: velocity vector of robot
        # vmax: maximum velocity of robot
        # v0: initial velocity of robot
        # v0max: maximum CARTESIAN COORDINATE of initial velocity
        # vstop: tolerance for velocity
        # x: position vector of robot
        # x0: initial position of robot
        # x0max: maximum CARTESIAN COORDINATE of initial condition


    # This generates the range of lambdas.  Here the first argument is the starting value, the third argument is the step, and the second argument is the "stopping" value, which is NOT included in the range.  So this is [0,0.1,...,1].
    # lam_list = np.arange(0,1.1, .1) #the list of lambdas to test
    # This tells us whether to plot the solutions or not.
    to_plot = True
    # to_plot = False
    
    # Maximum velocity of robot (m/s)
    vmax = 4.0
    # Maximum acceleration of robot (m/s^2)
    amax = 2.0
    # maximum CARTESIAN COORDINATE of initial condition
    x0max = 1000
    # maximum CARTESIAN COORDINATE of initial velocity
    v0max = 4
    #target radius
    rstop = 1
    #target velocity
    vstop = 0.01
    # Number of steps to run simulation
    n_steps = 10000
    
    
    # For testing purposes, seed the random number generator so we always get the same things:
        
    # np.random.seed(0)
    
    #get initial start spot
    step_n_list = list()
    # Create a random vector with two positions with magnitude less than x0max.
    # x0 = np.random.uniform(-x0max,x0max, 2)
    x0 = np.random.uniform(-x0max,x0max, 2)
    # Create a random vector with two positions with velocity less than v0max.
    # v0 = np.random.uniform(-v0max,v0max, 2)
    v0 = np.random.uniform(-v0max,v0max, 2)
    # This line will never execute as long as vmax^2>2v0max^2.
    if np.linalg.norm(v0) > vmax:
        v0 = v0/np.linalg.norm(v0) * vmax
    
    lam = 0.7
        
    x = x0
    v = v0
    
    #plot the start
    if to_plot:
        plt.plot(x[0], x[1], '*')
        plt.plot(0,0,'k*')
        plt.arrow(x[0], x[1], v[0], v[1])
    x_list = list()
    v_list = list()
    a_list = list()
    
    # Keep track of iterations.
    i = 1
    
    #loop over every timestep.  Stop only when x and v are small.
    while((np.linalg.norm(x) > rstop) or (np.linalg.norm(v) > vstop) and i<n_steps):
        #update v
        
        x,v,a = advance(x,v,lam,amax,vmax)
        
        i = i + 1
        
        # If we are plotting, add the new values of the norm to the list.

        x_list.append(np.linalg.norm(x))
        v_list.append(np.linalg.norm(v))
        a_list.append(np.linalg.norm(a))
        
        if to_plot:
            plt.plot(x[0], x[1], '.')
        
    
    #record number of timesteps
    print(i)
    step_n_list.append(i)
        
    # plt.xlim([-.1,.1])
    # plt.ylim([-.1, .1])
       
    if to_plot:
        # make the origin star on top
        plt.plot(0,0,'k*')
        
        # #plot circle 
        # angle = np.linspace( 0 , 2 * np.pi , 150 ) 
        
        # radius = 0.5
        # x = radius * np.cos( angle ) 
        # y = radius * np.sin( angle ) 
        
        # plt.plot( x, y, 'k' ) 
        
        plt.title("Path with lambda = " +str(lam))
        
        plt.show()
        plt.semilogy(x_list, label = "x")
        plt.plot(v_list, label = "v")
        # plt.plot(a_list, label = "a")
        plt.legend()
        plt.show()
    
    # plt.semilogy(lam_list, step_n_list)
    # plt.xlabel("lambda")
    # plt.ylabel("timesteps to origin (Max at 10^4)")
    # plt.title("Timesteps to origin for 10 different initial conditions")

def advance(x,v,lam,amax,vmax):
# This function determines the new values of x and v given the previous values.  It just does one time step.

# Called by: main
# Calls: none.

# Input variables:
    # amax: maximum acceleration
    # lam: lambda
    # v: velocity at previous time step
    # vmax: maximum speed
    # x: position at previous time step
    
# Internal variables:
    # rd: decleration distance
    
    # time step
    dt = 0.1
    
# Calculate the new acceleration direction, given the value of lambda.
    a = ((1-lam)*v+lam*x)
# Then set its magnitude to amax.  Note there is no deceleration radius anymore.
    a = -a/np.linalg.norm(a) * amax
    # IMPORTANT: It is possible that this will have to be adjusted so that it sets to a smaller value at the end, when x is near 0 and v is near 0 so it doesn't oscillate.
    
    #update v
    v = v + a * dt
    # If the speed is greater than vmax, renormalize
    if np.linalg.norm(v) > vmax:
        v = vmax*v/np.linalg.norm(v)
    
    #take a step in space
    x = x + v * dt
    
    return x,v,a

# Execute the code.
main()