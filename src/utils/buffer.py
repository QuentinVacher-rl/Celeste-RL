import numpy as np

class Buffer:

    def __init__(self, size, action_size, state_size) -> None:
        self.size = size
        self.state_size = state_size
        self.action_size = action_size
        self.states = np.zeros((size, state_size))
        self.actions_probs = np.zeros((size, action_size))
        self.actions = np.zeros((size, 1))
        self.values = np.zeros((size, 1))
        self.rewards = np.zeros((size, 1))
        self.dones = np.zeros((size, 1))

        self.index = 0
        self.full = False

    def insert_data(self, state, action, action_prob, value, reward, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.actions_probs[self.index] = action_prob
        self.values[self.index] = value
        self.rewards[self.index] = reward
        self.dones[self.index] = done

        self.index += 1
        self.full = self.index == self.size

    def reset(self):
        self.index = 0
        self.full = False
        self.states = np.zeros((self.size, self.state_size))
        self.actions_probs = np.zeros((self.size, self.action_size))
        self.actions = np.zeros((self.size, 1))
        self.values = np.zeros((self.size, 1))
        self.rewards = np.zeros((self.size, 1))
        self.dones = np.zeros((self.size, 1))
