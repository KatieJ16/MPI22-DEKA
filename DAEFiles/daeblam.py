# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:26:21 2022

@author: David A. Edwards adapted from Katherine Johnston's code.

Last Revised: 6/15/22.
"""

# Code daeblam.  This code computes the number of iterates used for convergence vs. lambda for a given number of initial condition.

# Import the pacakges we need.
import numpy as np
import matplotlib.pyplot as plt

    
# Just like in Pascal, functions have to be defined ahead of any script.  So we just define a main function which is the script, and call it at the end.

def main():

# Variables:
    # dt: time step
# lam_list: list of lambda values to test
# lamgrid: number of lambda points to test
# numplot: looping variable for simulation
# numic: number of simulations to do
# step_n_list: list of iterations for each lambda
# to_plot: logical variable for if we are producing plots
# v0: initial velocity of robot
# v0max: maximum CARTESIAN COORDINATE of initial velocity
    # vmax: maximum velocity of robot
# x0: initial position of robot
# x0max: maximum CARTESIAN COORDINATE of initial condition

# How many lambdas to test.
    lamgrid = 10
# This generates the range of lambdas.  Here the first argument is the starting value, the third argument is the step, and the second argument is the "stopping" value, which is NOT included in the range.  So this is [0.1,...,0.9].
    lam_list = np.arange(1/(lamgrid+1),1,1/(lamgrid+1))
# This tells us whether to plot the solutions or not.
    to_plot = True
    # to_plot = False
    
    # Define time step.
    dt = 0.1
    # maximum CARTESIAN COORDINATE of initial condition
    x0max = 1000
    # maximum CARTESIAN COORDINATE of initial velocity
    v0max = 4
    # Maximum velocity of robot (m/s)
    vmax = 4.0
    
    # Number of initial conditions to check.
    numic = 10
     
    for numplot in range(numic):
        #Create a list where we will store the number of iterations versus lambda.
        step_n_list = list()
        # For testing purposes, seed the random number generator so we always get the same things:
            
        # np.random.seed(0)
        
        # Create a random vector with two positions with magnitude less than x0max.
        x0 = np.random.uniform(-x0max,x0max, 2)
        # Create a random vector with two positions with velocity less than v0max.
        v0 = np.random.uniform(-v0max,v0max, 2)
        if np.linalg.norm(v0) > vmax:
            v0 = v0/np.linalg.norm(v0) * vmax
            
        # Now test each lambda.  Here lam_list is already a range, so the for loop will do each version.
        for lam in lam_list:
        
            i = niter(x0,v0,lam,vmax,dt)
            
            step_n_list.append(i)
            
        # Convert our list of iterates into the performance factor
        step_n_list = dt * np.array(step_n_list) * vmax/np.linalg.norm(x0)
               
        if to_plot:
            plt.plot(lam_list, step_n_list)
        
    if to_plot:
        plt.xlabel("$\lambda$")
        plt.ylabel("Performance factor")
        plt.title("Performance factor for %d initial conditions" % numic)
        # IMPORTANT: In order for the plot to display when you are also saving it, the commands must be in this order:
        plt.savefig('lamvary.pdf')

    
def niter(x0,v0,lam,vmax,dt):
# This function calculates the number of iterations needed to get to the origin from x0 and v0 given lambda.

# Called by: main
# Calls: advance.

# Input variables:
    # lam: value of lambda
    # vmax: maximum velocity of robot
    # v0: initial velocity of robot
    # x0: initial position of robot


# Internal variables:
    # a: acceleration vector
    # amax: maximum acceleration of robot
    # epsilon: target radius
    # i: step variable
    # lam_list: list of lambda values to test
    # n_steps: number of steps to run code
    # num: looping variabe for simulation
    # rstop: tolerance for radius
    # v: velocity vector of robot

    # vstop: tolerance for velocity
    # x: position vector of robot


    # Maximum acceleration of robot (m/s^2)
    amax = 2.0
    #target radius
    rstop = 1
    #target velocity
    vstop = 0.01
    # Number of steps to run simulation
    n_steps = 10000
    
    x = x0
    v = v0
    
    # Keep track of iterations.
    i = 1
    
    #Loop over every time step.  If we haven't done the maximum number of steps (i<nsteps, so that's with an AND), then we continue as long as one of the stopping conditions isn't satisfied (r>rs OR v>vs).
    while(((np.linalg.norm(x) > rstop) or (np.linalg.norm(v) > vstop)) and i<n_steps):
        #update v
        
        x,v,a = advance(x,v,lam,amax,vmax,dt)
        
        i = i + 1
    
    #record number of timesteps
    print(i)
    
    return i

def advance(x,v,lam,amax,vmax,dt):
# This function determines the new values of x and v given the previous values.  It just does one time step.

# Called by: niter
# Calls: none.

# Input variables:
    # amax: maximum acceleration
    # dt: time step
    # lam: lambda
    # v: velocity at previous time step
    # vmax: maximum speed
    # x: position at previous time step
    
# Internal variables:
    # a: acceleration vector
    

    
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