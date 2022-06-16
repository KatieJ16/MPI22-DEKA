# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:26:21 2022

@author: David A. Edwards adapted from Katherine Johnston's code.

Last revised on 6/16/22
"""

# Code dae1t.  This code computes a single trajectory for a single value of lambda.   Moreover, it stores the iterates for x, v, and a, as well as how aligned x and v are.

# Import the pacakges we need.
import numpy as np
import matplotlib.pyplot as plt
# This library gives trig functions.
import math

    
# Just like in Pascal, functions have to be defined ahead of any script.  So we just define a main function which is the script, and call it at the end.

def main():
    
    # Variables:
        # alignlist: list of alignments of x and v
        # alval: alignment value
        # amax: maximum acceleration of robot
        # dt: time step
        # epsilon: target radius
        # i: step variable
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
    
    # Create a random vector with two positions with magnitude less than x0max.
    # x0 = np.random.uniform(-x0max,x0max, 2)
    x0 = np.random.uniform(-x0max,x0max, 2)
    # Create a random vector with two positions with velocity less than v0max.
    # v0 = np.random.uniform(-v0max,v0max, 2)
    v0 = np.random.uniform(-v0max,v0max, 2)
    # This line will never execute as long as vmax^2>2v0max^2.
    if np.linalg.norm(v0) > vmax:
        v0 = v0/np.linalg.norm(v0) * vmax
        
        # x0 = np.array([678.15415613,-817.65612229])
        # v0 = np.array([2.34036646,-0.21327138])
    
    lam = 0.55
        
    x = x0
    v = v0
    
    #plot the start
    if to_plot:
        plt.plot(x[0], x[1], '*')
        plt.plot(0,0,'k*')
        # Plot the initial velocity.  Has to be larger because of the scale.
        plt.arrow(x[0], x[1], 20*v[0], 20*v[1])
    x_list = list()
    v_list = list()
    a_list = list()
    alignlist = list()
    
    # Keep track of iterations.
    i = 1
    
    #Loop over every time step.  If we haven't done the maximum number of steps (i<nsteps, so that's with an AND), then we continue as long as one of the stopping conditions isn't satisfied (r>rs OR v>vs).
    while(((np.linalg.norm(x) > rstop) or (np.linalg.norm(v) > vstop)) and i<n_steps):
        #update v
        
        x,v,a = advance(x,v,lam,amax,vmax)
        
        i = i + 1
        
        # Add the new values of the norm to the list.

        x_list.append(np.linalg.norm(x))
        v_list.append(np.linalg.norm(v))
        a_list.append(np.linalg.norm(a))
        
        # Next we track how aligned x and v are
        alval = np.dot(x,v)/np.linalg.norm(x)/np.linalg.norm(v)
        alignlist.append(alval)
        
        # If we are plotting, go ahead and add the point now.
        
        if to_plot:
            plt.plot(x[0], x[1], '.')
    
    #record number of timesteps
    print(i)
       
    if to_plot:
        # make the origin star on top
        plt.plot(0,0,'k*')
        
        # Give the iterate plot a title and labels
        plt.title("Path with $\lambda=$" +str(lam))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        
        # IMPORTANT: In order for the plot to display when you are also saving it, the commands must be in this order:
        plt.savefig('iterplot.pdf')
        plt.show()
       
        # Start the x and v figure.
        plt.semilogy(x_list, label = "$r$")
        plt.plot(v_list, label = "$v$")
        # plt.plot(a_list, label = "a")
        plt.xlabel('Iterate')
        plt.legend()
        plt.title("Radius and velocity vs. iterate")
        # IMPORTANT: In order for the plot to display when you are also saving it, the commands must be in this order:
        plt.savefig('xvplot.pdf')
        plt.show()
        
        # Start the alignment figure
        plt.plot(alignlist)
        plt.xlabel('Iterate')
        plt.ylabel('Alignment')
        plt.title("Alignment vs. iterate")
        plt.savefig('align.pdf')
        plt.show()


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
# # Calculate the new acceleration direction, given the value of lambda.
#     a = np.array([math.cos(lam),math.sin(lam)])*amax

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