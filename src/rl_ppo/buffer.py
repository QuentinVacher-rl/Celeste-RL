import numpy as np

class Buffer:

    def __init__(self, size, action_size, state_size, image_size, size_histo) -> None:
        self.size = size
        self.state_size = state_size
        self.action_size = action_size
        self.image_size = image_size
        if self.image_size is not None:
            self.images = np.zeros((size+1, (size_histo+1)*image_size[0], image_size[1], image_size[2]))
        self.states = np.zeros((size, state_size))
        self.actions = np.zeros((size, action_size))
        self.log_prob = np.zeros((size, action_size))
        self.values = np.zeros((size, 1))
        self.rewards = np.zeros((size, 1))
        self.dones = np.zeros((size, 1))

        self.index = 0
        self.full = False

    def insert_data(self, state, image, action, log_prob, value, reward, done):
        self.states[self.index] = state
        if self.image_size is not None:
            self.images[self.index] = image
        self.actions[self.index] = action
        self.log_prob[self.index] = log_prob
        self.values[self.index] = value
        self.rewards[self.index] = reward
        self.dones[self.index] = done

        self.index += 1
        self.full = self.index == self.size

    def reset(self):
        self.index = 0
        self.full = False
        self.states.fill(0)
        if self.image_size is not None:
            self.images.fill(0)
        self.actions.fill(0)
        self.log_prob.fill(0)
        self.values.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
