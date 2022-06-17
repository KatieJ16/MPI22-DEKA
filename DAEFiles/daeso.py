# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:26:21 2022

@author: David A. Edwards adapted from Katherine Johnston's code.

Last Revised: 6/16/22.
"""

# Code daesopt.  This code generates a stochastic optimization solution for a particular initial condition and penalty function.

# Import the pacakges we need.
import numpy as np
import matplotlib.pyplot as plt

    
# Just like in Pascal, functions have to be defined ahead of any script.  So we just define a main function which is the script, and call it at the end.

def main():

# Variables:
    # flag: variable selecting cost function to use
# pfac: performance factor
# v0: initial velocity of robot
# v0max: maximum CARTESIAN COORDINATE of initial velocity
    # vmax: maximum velocity of robot
# x0: initial position of robot
# x0max: maximum CARTESIAN COORDINATE of initial condition

    # maximum CARTESIAN COORDINATE of initial condition
    x0max = 1000
    # maximum CARTESIAN COORDINATE of initial velocity
    v0max = 4
    # Maximum velocity of robot (m/s)
    vmax = 4.0
    
    # For testing purposes, seed the random number generator so we always get the same things:
        
    # np.random.seed(0)
    
    # Create a random vector with two positions with magnitude less than x0max.
    x0 = np.random.uniform(-x0max,x0max, 2)
    # Create a random vector with two positions with velocity less than v0max.
    v0 = np.random.uniform(-v0max,v0max, 2)
    if np.linalg.norm(v0) > vmax:
        v0 = v0/np.linalg.norm(v0) * vmax
    
    for flag in range(3):
    # Recall that this does 0,1,2, so we have to add 1
        pfac = plotgen(x0,v0,flag+1)
        print("flag=%d, p=%.2f" % (flag+1,pfac))


def plotgen(x0,v0,flag):
# This function generates a plot for a particular choice of flag

# Input variables:
    # flag: variable specifying cost function
    # v0: initial velocity
    # x0: initial position

# Output variable:
    # titer: number of iterations

# Internal variables:
    # bestfiter: best value of f in an iteration
    # besti: best value of i in an iteration
    # bestlist: list of F values
    # bestv: best value of v in an iteration
    # bestx: best value of x in an iteration
    # dt: time step
    # exitflag: this stays at 0 if the code terminates without switching to convergence mode (hasn't been happening)
    # f: value of cost function
    # ftol: best value of f to date
    # lam_list: list of lambda values to test
    # lamgrid: number of lambda points to test
    # n_steps: number of steps to run code
    # numplot: looping variable for simulation
    # numic: number of simulations to do
    # step_n_list: list of iterations for each lambda
    # times: array of times of each iteration
    # to_plot: logical variable for if we are producing plots
        # vmax: maximum velocity of robot
    # x0max: maximum CARTESIAN COORDINATE of initial condition
    # xz: stored value of x0

# How many lambdas to test.
    lamgrid = 10
# This generates the range of lambdas.  Here the first argument is the starting value, the third argument is the step, and the second argument is the "stopping" value, which is NOT included in the range.  So this is [0.1,...,0.9].
    lam_list = np.arange(1/(lamgrid+1),1,1/(lamgrid+1))
# This tells us whether to plot the solutions or not.
    to_plot = True
    # to_plot = False
    
    # Define time step.
    dt = 0.1
    # Number of steps to run simulation
    n_steps = 10
    # maximum CARTESIAN COORDINATE of initial condition
    x0max = 1000
    # Maximum velocity of robot (m/s)
    vmax = 4.0
    
    xz = x0
    
    #Create a list where we will store the best F for each t*
    bestlist = list()

    # plot the start
    if to_plot:
        plt.plot(x0[0], x0[1], '*')
        plt.plot(0,0,'k*')
        # Plot the initial velocity.  Has to be larger because of the scale.
        plt.arrow(x0[0], x0[1], 20*v0[0], 20*v0[1])

    # Set maximum tolerance
    ftol = 2*x0max
    
    # Initialize iteration count
    titer = 0
    # Initialize exitflag variable
    exitflag = 0
    
    # Do this loop until you get to the origin.
    while ftol>0:
    # Now test each lambda.  Here lam_list is already a range, so the for loop will do each version.
        
        # Reset the best fit for this interval of time.
        bestfiter = 2*x0max
        for lam in lam_list:
        
            f,x,v,i = seestep(x0,v0,lam,vmax,dt,flag,n_steps)
                        
            # If f in this iteration is better than the best in the interval:
            if f<bestfiter:
                bestfiter = f
                bestlam = lam
                bestx = x
                bestv = v
                besti = i
        
        # Add the best value of F to the list.
        bestlist.append(bestfiter)
        # Add the number of iterations used to our running total.
        titer = titer + besti
    
        print(bestfiter,x,v)
        
        ftol = bestfiter
        
        #Then reset our "initial" condition to the best x, v
        x0 = bestx
        v0 = bestv
        
        # If we are plotting, go ahead and add the point now.
        
        if to_plot:
            plt.plot(bestx[0], bestx[1], '.')

    # Convert our list of iterates into the performance factor
    titer = dt * titer * vmax/(np.linalg.norm(xz))

    # Convert iterates to time steps.  (This is actually a bit of a kludge, because the last one may not be the full 10 iterates, but it's within a second.)
    times = np.array(range(len(bestlist)))*dt*n_steps
       
    if to_plot:
        # make the origin star on top
        plt.plot(0,0,'k*')
        
        # Give the iterate plot a title and labels.  Note that the variables have to be contained in parentheses because we have more than one.
        plt.title("Path, $F$=(4.%d), $p$=%.2f" % (flag,titer))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        
        # IMPORTANT: In order for the plot to display when you are also saving it, the commands must be in this order:
        plt.savefig('sof%d.pdf' % flag)
        plt.show()
        
        # Start the alignment figure
        plt.semilogy(times,bestlist)
        plt.xlabel('$t$ (s)')
        plt.ylabel('$F$')
        plt.title("$F$ vs. $t$")
        plt.savefig('fplot%d.pdf' %flag)
        plt.show()
        
    return titer

def seestep(x0,v0,lam,vmax,dt,flag,n_steps):
# This function returns the penalty function and the corresponding x and v for a given value of lambda and 10 time steps.

# Called by: main
# Calls: advance, penalty.

# Input variables:
    # lam: value of lambda
    # n_steps: number of steps to run code
    # vmax: maximum velocity of robot
    # v0: initial velocity of robot
    # x0: initial position of robot

# Internal variables:
    # a: acceleration vector
    # amax: maximum acceleration of robot
    # epsilon: target radius
    # lam_list: list of lambda values to test
    # num: looping variabe for simulation
    # rstop: tolerance for radius
    # vstop: tolerance for velocity
    
# Output variables:
    # f: penalty value
    # i: step variable
    # v: velocity vector of robot
    # x: position vector of robot

    # Maximum acceleration of robot (m/s^2)
    amax = 2.0
    #target radius
    rstop = 1
    #target velocity
    vstop = 0.01

    
    x = x0
    v = v0
    
    # Keep track of iterations.
    i = 1
    
    #Loop over every time step.  If we haven't done the maximum number of steps (i<nsteps, so that's with an AND), then we continue as long as one of the stopping conditions isn't satisfied (r>rs OR v>vs).
    while(((np.linalg.norm(x) > rstop) or (np.linalg.norm(v) > vstop)) and i<n_steps):
        #update v
        
        x,v,a = advance(x,v,lam,amax,vmax,dt)
        # if tt == 1:
        #     print("x=",x,np.linalg.norm(x),"v=",v,np.linalg.norm(v))
        # # print((np.linalg.norm(x) > rstop),(np.linalg.norm(v) > vstop),i<n_steps)
        
        i = i + 1
    
    # Compute the penalty
    f = penalty(np.linalg.norm(x),np.linalg.norm(v),rstop,flag)
    
    # If we have converged because we have reached the stopping condition, set f to 0:
    if ((np.linalg.norm(x) < rstop) and (np.linalg.norm(v) < vstop)):
        f = 0
    
    return f,x,v,i

def penalty(r,vel,rstop,flag):
# This code computes the local penalty.

# Called by: seestep.
# Calls: none.

# Input variables:
    # flag: flag telling which case we are in
    # rstop: Stopping radius
    # r: radius
    # vel: velocity
    
# Output variable:
    # f: penalty function
    
# In all the cases, f has the radius in it
    f = r
    
    # Second case: change to velocity 
    if (flag==2 and r<rstop):
        f = vel
        
    # Third case: add velocity penalty
    if (flag==3):
        if r>rstop:
            f = f + vel**(-1)
        else:
            f = f + vel
    
    return f

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
    
    # We introduce a kludge.  If x and v are parallele,
# Calculate the new acceleration direction, given the value of lambda.
    a = ((1-lam)*v+lam*x)
    # if tt==1:
    #     print("a=",a, end=",")
# Then set its magnitude to amax.  Note there is no deceleration radius anymore.
    a = -a/np.linalg.norm(a) * amax
    # if np.linalg.norm(x) < vmax**2/2/amax:
    #     a=-a
    # if tt==1:
    #     print("norma=",a)
    #     print("xold=",x,", vold=",v)
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