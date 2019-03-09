import gym
import matplotlib.pyplot as plt
from game import Dqn
from game.base.game_base import Game
import sys


class MountainCarGame(Game):

    def __init__(self):
        self.game_base = Game("MountainCar-v0")

    def play(self):
        env, action_space, observation_space = self.game_base.create_env()
        brain = Dqn(observation_space, action_space, 0.9)

        observation = env.reset()

        reward = 0
        scores = list()
        max_position = -.4
        max_count = 0

        gradient_reward = 5

        observation = env.reset()

        for episode in range(10000):
            print("EPISODE: {episode}".format(episode=episode), end="\r")
            # observation = env.reset()
            # for step in range(200):
            while True:
                env.render()
                action = brain.update(reward, observation)
                scores.append(brain.score())
                observation, reward, done, info = env.step(action)

                if observation[0] > max_position:
                    max_count += 1
                    max_position = observation[0]

                    print("{count} - reached new highest and got a bonus of {bonus}".format(
                        count=max_count, bonus=gradient_reward))

                    gradient_reward += 1
                    reward = gradient_reward

                if done:
                    if observation[0] >= 0.5:
                        reward = gradient_reward + 1
                        print("YOU DID IT")
                        print("EPISODE {episode}".format(episode=episode))
                        plt.plot(scores)
                        plt.show()
                        sys.exit(-1)
                    break


# def play():
#     env = gym.make("MountainCar-v0")
#     actions = env.action_space.n
#     space = env.observation_space.shape[0]

#     brain = Dqn(space, actions, 0.9)

#     env.reset()

#     reward = 0
#     scores = list()
#     max_position = -.4
#     max_count = 0

#     gradient_reward = 5

#     observation = env.reset()

#     for episode in range(10000):
#         print("EPISODE: {episode}".format(episode=episode), end="\r")
#         # observation = env.reset()
#         # for step in range(200):
#         while True:
#             env.render()
#             action = brain.update(reward, observation)
#             scores.append(brain.score())
#             observation, reward, done, info = env.step(action)

#             if observation[0] > max_position:
#                 max_count += 1
#                 max_position = observation[0]

#                 print("{count} - reached new highest and got a bonus of {bonus}".format(
#                     count=max_count, bonus=gradient_reward))

#                 gradient_reward += 1
#                 reward = gradient_reward

#             if done:
#                 if observation[0] >= 0.5:
#                     reward = gradient_reward + 1
#                     print("YOU DID IT")
#                     print("EPISODE {episode}".format(episode=episode))
#                     plt.plot(scores)
#                     plt.show()
#                     sys.exit(-1)
#                 break

#     env.close()
