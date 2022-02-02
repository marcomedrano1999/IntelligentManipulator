"""
Script to a fast test of the DDPG algorithm

Author: Marco Medrano
Date: 01/02/2022
"""

import gym
import argparse
from DDPG_agent import DDPG

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
        agent = DDPG(num_states,num_actions)


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
    env = testEnv()

    # Train env
    env.run(num_episodes,num_steps)


if __name__=="__main__":
    main()