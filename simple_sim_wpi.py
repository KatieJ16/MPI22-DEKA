# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:26:21 2022

@author: kjohnst
"""

import numpy as np
import matplotlib.pyplot as plt

#set beginning values
x = np.random.uniform(0,1, 2)
v = np.random.uniform(0,1, 2)
if np.linalg.norm(v) > 1.0:
    v = v/np.linalg.norm(v)

#plot the start
print("v = ", v)
plt.plot(x[0], x[1], '*')
plt.plot(0,0,'k*')
plt.arrow(x[0], x[1], v[0], v[1])#, arrowprops=dict(arrowstyle="->"))

a = -x / np.linalg.norm(x)

print("a = ", a)

#setting some parameters
epsilon = 0.01
n_steps = 200
dt = 0.1

#loop over every timestep
for i in range(n_steps):
    #update v
    v = v + a * dt
    if np.linalg.norm(v) > 1.0:
        v = v/np.linalg.norm(v)
        
    print("v = ", np.linalg.norm(v))
        
    #take a step in space
    x = x + v * dt
    #acceleration always goes to the origin
    a = -x / np.linalg.norm(x)
    plt.plot(x[0], x[1], '.')
    
    #when to hit breaks
    # if np.linalg.norm(x) < 1/2:
    #     a = x / np.linalg.norm(x)
        
    #check if we are close enough to origin then stop
    if np.linalg.norm(x) < epsilon:
        print("DONE")
        break
    
    
# make the origin star on top
print("v = ", v)
plt.plot(0,0,'k*')

#plot circle 
angle = np.linspace( 0 , 2 * np.pi , 150 ) 

radius = 0.5
x = radius * np.cos( angle ) 
y = radius * np.sin( angle ) 

# plt.plot( x, y, 'k' ) 

plt.title("Path when lambda = 1")
plt.savefig("lambda1.pdf")

    