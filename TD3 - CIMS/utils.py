# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:41:01 2021

@author: jkenney
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 episode rewards')
    plt.savefig(figure_file)