"""
Script to a fast test of the DDPG algorithm

Author: Marco Medrano
Date: 01/02/2022
"""

import gym
import argparse
import tensorflow as tf
import numpy as np
from DDPG_agent import OUActionNoise
import DDPG_agent
import matplotlib.pyplot as plt


class testEnv:

    def __init__(self):
        """
        To make quick tests of the RL algoritms, we're using the 
        FetchPickAndPlace-v1 environment provided by Gym
        """
        
        # Create the env
        self.env = gym.make('FetchPickAndPlace-v1')
        self.env.reset()

        # Env parameters
        num_states = 25
        num_actions = 4

        # Create DDPG agent
        # agent = DDPG(num_states,num_actions)


    def run(self, num_episodes=20, num_steps=100):
        """
        
        """
        
        
        for i_episode in range(num_episodes):
            observation = self.env.reset()
            for t in range(num_steps):
                self.env.render()
                
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

        self.env.close()


def policy(state, noise_object):
    global actor_model
    #global lower_bound, upper_bound

    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Add noise to the action
    sampled_actions = sampled_actions.numpy() + noise

    # put action within bounds
    legal_action = sampled_actions#np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


def main():

    # Get input parameters
    parser = argparse.ArgumentParser(description="Test env for RL algoritms in Intelligent Manipulator")    
    parser.add_argument('--num_episodes',default=20,type=int)
    parser.add_argument('--num_steps',default=100,type=int)
    args = parser.parse_args()

    # Use input parameters
    num_episodes = args.num_episodes
    num_steps = args.num_steps

    # Create env
    env = gym.make('FetchPickAndPlace-v1')
    env.reset()

    # Env parameters
    num_states = 25
    num_actions = 4


    # Noise generator
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_dev=float(std_dev)*np.ones(1))

    actor_model = DDPG_agent.create_actor(num_states,num_actions)
    critic_model = DDPG_agent.create_critic(num_states,num_actions)

    target_actor = DDPG_agent.create_actor(num_states,num_actions)
    target_critic = DDPG_agent.create_critic(num_states,num_actions)


    # Making the weights equa initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr=0.001

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    total_episodes = 100
    # Discount factor for future rewards
    gamma = 0.99
    # value to update networks
    tau=0.005

    buffer=DDPG_agent.Buffer(num_states, num_actions, 500, 32)

    ep_reward_list = []
    avg_reward_list = []

    # Model execution
    for ep in range(total_episodes):

        prev_state = env.reset()["observation"]
        episodic_reward = 0

        while True:
            env.render()

            tf_pre_state = tf.expand_dims(tf.convert_to_tensor(prev_state),0)

            action = policy(tf_pre_state, ou_noise)
            # Recieve state and reward from environment
            state, reward, done, info = env.step(action)

            state = state[["observation"]]

            buffer.record((prev_state, action, reward, state))
            episodic_reward +=reward
            
            buffer.learn()
            DDPG_agent.update_target(target_actor.variables, actor_model.variables, tau)
            DDPG_agent.update_target(target_critic.variables, critic_model.variables, tau)

            if done:
                break

            prev_state=state
        
        ep_reward_list.append(episodic_reward)

        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg reward is : {}".format(ep,avg_reward))
        avg_reward_list.append(avg_reward)
    
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic reward")
    plt.show()

    # Save weights
    actor_model.save_weights("PickAndPlace_actor.h5")
    critic_model.save_weights("PickAndPlace_critic.h5")

    target_actor.save_weights("PickAndPlace_target_actor.h5")
    target_critic.save_weights("PickAndPlace_target_critic.h5")



if __name__=="__main__":
    main()
