# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:40:42 2021

@author: jkenney
"""

import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras import initializers

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = Dense(self.fc1_dims, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001))
        # self.bn1 = BatchNormalization()
        # self.ac1 = ReLU()
        self.fc2 = Dense(self.fc1_dims, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001))
        # self.bn2 = BatchNormalization()
        # self.ac2 = ReLU()
        self.fc3 = Dense(self.fc1_dims, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001))
        # self.bn3 = BatchNormalization()
        # self.ac3 = ReLU()
        self.fc4 = Dense(self.fc1_dims, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001))
        # self.bn4 = BatchNormalization()
        # self.ac4 = ReLU()
        
        self.q = Dense(1, activation=None)

    def call(self, state, action, training):
        q1_action_value = self.fc1(tf.concat([state, action], axis=1))
        # q1_action_value = self.bn1(q1_action_value, training=training)
        # q1_action_value = self.ac1(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        # q1_action_value = self.bn2(q1_action_value, training=training)
        # q1_action_value = self.ac2(q1_action_value)
        q1_action_value = self.fc3(q1_action_value)
        # q1_action_value = self.bn3(q1_action_value, training=training)
        # q1_action_value = self.ac3(q1_action_value)
        q1_action_value = self.fc4(q1_action_value)
        # q1_action_value = self.bn4(q1_action_value, training=training)
        # q1_action_value = self.ac4(q1_action_value)

        q = self.q(q1_action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims, n_actions, name, chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = Dense(self.fc1_dims, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001))
        # self.bn1 = BatchNormalization()
        # self.ac1 = ReLU()
        self.fc2 = Dense(self.fc1_dims, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001))
        # self.bn2 = BatchNormalization()
        # self.ac2 = ReLU()
        self.fc3 = Dense(self.fc1_dims, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001))
        # self.bn3 = BatchNormalization()
        # self.ac3 = ReLU()
        self.fc4 = Dense(self.fc1_dims, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001))
        # self.bn4 = BatchNormalization()
        # self.ac4 = ReLU()
        self.mu = Dense(self.n_actions, activation='tanh')


    def call(self, state, training):
        prob = self.fc1(state)
        # prob = self.bn1(prob, training=training)
        # prob = self.ac1(prob)
        prob = self.fc2(prob)
        # prob = self.bn2(prob, training=training)
        # prob = self.ac2(prob)
        prob = self.fc3(prob)
        # prob = self.bn3(prob, training=training)
        # prob = self.ac3(prob)
        prob = self.fc4(prob)
        # prob = self.bn4(prob, training=training)
        # prob = self.ac4(prob)

        mu = self.mu(prob)

        return mu