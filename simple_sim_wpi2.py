# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:26:21 2022

@author: kjohnst
"""

import numpy as np
import matplotlib.pyplot as plt

lam_list = np.arange(0,1.1, .1) #the list of lambdas to test
to_plot = False


for num in range(10): #we are repeating everything 10 times
    #get initial start spot
    step_n_list = list()
    x = np.random.uniform(0.5,1, 2)
    v = np.random.uniform(0,1, 2)
    if np.linalg.norm(v) > 1.0:
        v = v/np.linalg.norm(v)
    x0 = x
    v0 = v
     
     
    #run for every lambda 
    for lam in lam_list:
        
        #set beginning values
        x = x0
        v = v0
        
        #plot the start
        # print("v = ", v)
        if to_plot:
            plt.plot(x[0], x[1], '*')
            plt.plot(0,0,'k*')
            plt.arrow(x[0], x[1], v[0], v[1])
        
        a = -x / np.linalg.norm(x)
        
        # x_list = list()
        # v_list = list()
        # a_list = list()
        
        # x_list.append(np.linalg.norm(x))
        # v_list.append(np.linalg.norm(v))
        # a_list.append(np.linalg.norm(a))
        

        #setting some parameters
        epsilon = 0.01
        n_steps = 10000
        dt = 0.1
        
        
        #loop over every timestep
        for i in range(1, n_steps):
            #update v
            v = v + a * dt
            if np.linalg.norm(v) > 1.0:
                v = v/np.linalg.norm(v)
                
            
            #take a step in space
            x = x + v * dt
            # x_list.append(np.linalg.norm(x))
            # v_list.append(np.linalg.norm(v))
            # a_list.append(np.linalg.norm(a))
            
            #get the direction of a
            a = ((1-lam)*v+lam*x)
            
            #get magnitude of a
            a = -a/np.linalg.norm(a) #outside circle
            if np.linalg.norm(x) < 1/2: #hit the breaks, inside circle
                a = a/np.linalg.norm(a)
            if to_plot:
                plt.plot(x[0], x[1], '.')
            
                
            #check if we are close enough to origin then stop
            if np.linalg.norm(x) < epsilon:
                print("DONE")
                break
        
        #record number of timesteps
        print(i)
        step_n_list.append(i)
            
        # plt.xlim([-.1,.1])
        # plt.ylim([-.1, .1])
           
        if to_plot:
            # make the origin star on top
            plt.plot(0,0,'k*')
            
            #plot circle 
            angle = np.linspace( 0 , 2 * np.pi , 150 ) 
            
            radius = 0.5
            x = radius * np.cos( angle ) 
            y = radius * np.sin( angle ) 
            
            plt.plot( x, y, 'k' ) 
            
            plt.show()
        # plt.semilogy(x_list, label = "x")
        # plt.plot(v_list, label = "v")
        # plt.plot(a_list, label = "a")
        # plt.legend()
    
    plt.semilogy(lam_list, step_n_list)
    