# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:26:21 2022

@author: kjohnst
"""

import numpy as np
import matplotlib.pyplot as plt

#set beginning values
x = np.random.uniform(0.5,1, 2)
v = np.random.uniform(0,1, 2)
x0 = x
v0 = v
if np.linalg.norm(v) > 1.0:
    v = v/np.linalg.norm(v)

#plot the start
print("v = ", v)
plt.plot(x[0], x[1], '*')
plt.plot(0,0,'k*')
plt.arrow(x[0], x[1], v[0], v[1])#, arrowprops=dict(arrowstyle="->"))

a = -x / np.linalg.norm(x)

x_list = list()
v_list = list()
a_list = list()

x_list.append(np.linalg.norm(x))
v_list.append(np.linalg.norm(v))
a_list.append(np.linalg.norm(a))

print("a = ", a)

#setting some parameters
epsilon = 0.01
n_steps = 100
dt = 0.1

#loop over every timestep
for i in range(1, n_steps):
    #update v
    v0 = v
    v = v + a * dt
    if np.linalg.norm(v) > 1.0:
        v = v/np.linalg.norm(v)
        
    
    #take a step in space
    x0 = x
    x = x + v * dt
    x_list.append(np.linalg.norm(x))
    v_list.append(np.linalg.norm(v))
    a_list.append(np.linalg.norm(a))
    print("v = ", v)
    print("x = ", x)
    print("np.dot(-x, v) -np.linalg.norm(x)  * np.linalg.norm(v) = ", np.dot(-x, v) -np.linalg.norm(x)  * np.linalg.norm(v))
    # if np.abs(np.dot(-x, v) -np.linalg.norm(x)  * np.linalg.norm(v))< epsilon:
    #     a = -v/np.linalg.norm(v)
    #     print("BREAK")
    #     break
    #acceleration always goes to the origin
    # a = -x / np.linalg.norm(x)
    
    a = (v+x)#(x - x0 - v*(dt))/((dt)**2) * 2
    
    print("a = ", a)
    a = -a/np.linalg.norm(a)
    if np.linalg.norm(x) < 1/2:
        a = a/np.linalg.norm(a)
    plt.plot(x[0], x[1], '.')
    
    # #when to hit breaks
    # if np.linalg.norm(x) < 1/2:
    #     a = x / np.linalg.norm(x)
        
    #check if we are close enough to origin then stop
    if np.linalg.norm(x) < epsilon:
        print("DONE")
        break
    
# plt.xlim([-.1,.1])
# plt.ylim([-.1, .1])
    
# make the origin star on top
print("v = ", v)
plt.plot(0,0,'k*')

#plot circle 
angle = np.linspace( 0 , 2 * np.pi , 150 ) 

radius = 0.5
x = radius * np.cos( angle ) 
y = radius * np.sin( angle ) 

plt.plot( x, y, 'k' ) 

plt.show()
plt.semilogy(x_list)
plt.plot(v_list)
plt.plot(a_list)


    