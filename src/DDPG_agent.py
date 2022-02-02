"""
DDPG RL class


Author: Marco Medrano
Date: 30/01/2022
"""

from operator import concat
from turtle import shape
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib as plt



class DDPG:
    """
    Agent class
    """
    def __init__(self, numObs, numActs):
        """
        Agent constructor
        """
        self.actor = self.create_actor()

        self.critic = self.create_critic(numObs,numActs)

    def create_actor(self, num_states, num_actions):
        # Actor based in keras
        

        state_in = layers.Input(shape=(num_states))
        out = layers.Dense(256, activation="relu")(state_in)
        out = layers.Dense(256, activation="relu")(out)
        action_out = layers.Dense(num_actions, activation="tanh")(out)

        tf.keras.Model(state_in,action_out)

    def create_critic(self, num_states,num_actions):
        # Critic networks are made of two types of inputs: actions and states

        # State part
        state_in = layers.Input(shape=(num_states))
        state_out = layers.Dense(16, activation="relu")(state_in)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action part
        action_in = layers.Input(shape=(num_actions))
        action_out = layers.Denser(32,activation="relu")(action_in)

        # Merge action and critic
        merged = layers.Concatenate()([state_out, action_out])

        # Final stages
        out = layers.Dense(256, activation="relu")(merged)
        out = layers.Dense(256, activation="relu")(out)
        output = layers.Dense(1)(out)

        critic_model = tf.keras.Model([state_in, action_in], output)
        
        return critic_model

