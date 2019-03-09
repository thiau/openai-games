import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Network Architecture

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action

        neurons = 100  # number of neurons

        self.fc1 = nn.Linear(input_size, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# Experience Reply

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()  # last state, new state, last action, last reward

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = list()
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Creates the Torch Tensor
        # input_size is the size of the tensor, which is the state
        # of the environment: sensors, orientations
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.batch_size = 100

    def select_action(self, state):
        temperature = 7  # certainty of the action / exploration level
        volatile_state = Variable(state, volatile=True) * temperature
        q_values = self.model(volatile_state)
        probs = F.softmax(q_values)
        action = probs.multinomial()
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(
            1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state,
                          new_state,
                          torch.LongTensor([int(self.last_action)]),
                          torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > self.batch_size:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(
                self.batch_size)
            self.learn(batch_state, batch_next_state,
                       batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)

    def save(self):
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, "models/last_brain.pth")

    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("* Loading the Model")
            checkpoint = torch.load("models/last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("* Done")
        else:
            print("* Model not found")
