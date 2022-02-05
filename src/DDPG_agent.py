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

class OUActionNoise:
    def __init__(self,mean,std_dev,theta=0.15,dt=1e-2,x_initial=None):
        self.theta=theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x_initial=x_initial
        self.reset()

    def __call__(self):
        x=(self.x_prev+self.theta*(self.mean-self.x_prev)*self.dt + self.std_dev*np.sqrt(self.dt)*np.random.normal(size=self.mean.shape))
        self.x_prev=x
        return x

    def reset(self):
        if  self.x_initial is not None:
            self.x_prev=self.x_initial
        else:
            self.x_prev=np.zeros_like(self.mean)


class Buffer:

    def __init__(self,num_states, num_actions, buffer_capacity=1000,batch_size=64):
        # Number of experiences to store at max
        self.buffer_capacity=buffer_capacity
        # number of tuples to train on
        self.batch_size=batch_size

        # to save the number of time record() was called
        self.buffer_counter=0

        self.state_buffer=np.zeros((self.buffer_capacity,num_states))
        self.action_buffer=np.zeros((self.buffer_capacity,num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity,1))
        self.next_state_buffer=np.zeros((self.buffer_capacity,num_states))

    def record(self, obs_tuple):
        # Set index to zero if buffer capacity is exceeded
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index]=obs_tuple[0]
        self.action_buffer[index]=obs_tuple[1]
        self.reward_buffer[index]=obs_tuple[2]
        self.next_state_buffer[index]=obs_tuple[3]

        self.buffer_counter+=1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        global target_actor, target_critic, critic_model, action_model, critic_optimizer, gamma
        #Training and updating actor and critic networks
        with tf.GradientTapee() as tape:
            target_actions = target_actor(next_state_batch,training=True)
            y = reward_batch+gamma*target_critic([next_state_batch,target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y-critic_value))

            critic_grad = tape.gradient(critic_loss,critic_model.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grad,critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch,training=True)
            critic_value = critic_model([state_batch, actions], training=True)

            actor_loss = tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(critic_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grad,actor_model.trainable_variables))
    
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sampling indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

@tf.function
def update_target(target_weights, weights, tau):
    for (a,b) in zip(target_weights, weights):
        a.assign(b*tau+a*(1-tau))

def create_actor(num_states, num_actions):
    # Actor based in keras
    
    state_in = layers.Input(shape=(num_states))
    out = layers.Dense(256, activation="relu")(state_in)
    out = layers.Dense(256, activation="relu")(out)
    action_out = layers.Dense(num_actions, activation="tanh")(out)

    actor_moder = tf.keras.Model(state_in,action_out)

    return actor_moder


def create_critic(num_states,num_actions):
    # Critic networks are made of two types of inputs: actions and states

    # State part
    state_in = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_in)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action part
    action_in = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32,activation="relu")(action_in)

    # Merge action and critic
    merged = layers.Concatenate()([state_out, action_out])

    # Final stages
    out = layers.Dense(256, activation="relu")(merged)
    out = layers.Dense(256, activation="relu")(out)
    output = layers.Dense(1)(out)

    critic_model = tf.keras.Model([state_in, action_in], output)
    
    return critic_model

