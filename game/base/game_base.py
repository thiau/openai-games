"""
Base Module for OpenAI gaming
"""

import gym
from game import Dqn


class Game():
    """ A base class for OpenAI games """

    def __init__(self, game_name):
        self.game_name = game_name

    def create_env(self):
        env = gym.make(self.game_name)
        action_space = env.action_space.n
        observation_space = env.observation_space.shape[0]

        return env, action_space, observation_space
