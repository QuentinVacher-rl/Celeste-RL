
import os
import torch

class ReplayBuffer:

    def __init__(self, size, action_size, state_size, image_size, size_histo, file_save=None):
        self.size = size
        self.state_size = state_size
        self.action_size = action_size
        self.image_size = image_size
        self.size_histo = size_histo
        self.file_save = file_save
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_device = torch.device("cpu")
        self.states = torch.zeros((size+1, state_size), device=self.device)
        self.actions_probs = torch.zeros((size, action_size), device=self.device)
        self.rewards = torch.zeros((size, 1), device=self.device)
        self.terminated = torch.zeros((size, 1), dtype=torch.bool, device=self.device)
        if self.image_size is not None:
            self.images = torch.zeros((size+1, (size_histo+1)*image_size[0], image_size[1], image_size[2]), device=self.image_device)

        self.index = 0
        self.current_size = 0

    def insert_data(self, state, new_state, image, new_image, actions_probs, reward, terminated,):
        self.states[self.index] = torch.tensor(state, device=self.device)
        self.states[self.index+1] = torch.tensor(new_state, device=self.device)
        self.actions_probs[self.index] = torch.tensor(actions_probs, device=self.device)
        self.rewards[self.index] = torch.tensor(reward, device=self.device)
        self.terminated[self.index] = torch.tensor(terminated, device=self.device)
        if self.image_size is not None:
            self.images[self.index] = torch.tensor(image, device=self.image_device)
            self.images[self.index+1] = torch.tensor(new_image, device=self.image_device)

        self.index = (self.index+1) % self.size
        self.current_size = min(self.current_size + 1, self.size - 1)

    def sample_data(self, batch_size):
        max_index = min(self.index, self.size)

        sampled_index = torch.randint(0, max_index, size=(batch_size,), device=self.device)

        sampled_states = self.states[sampled_index]
        sampled_new_states = self.states[sampled_index+1]
        sampled_actions_probs = self.actions_probs[sampled_index]
        sampled_rewards = self.rewards[sampled_index]
        sampled_terminated = self.terminated[sampled_index]
        if self.image_size is not None:
            sampled_images = self.images[sampled_index.to(self.image_device)].to(self.device)
            sampled_new_images = self.images[sampled_index.to(self.image_device)+1].to(self.device)
        else:
            sampled_images = None
            sampled_new_images = None

        return sampled_states, sampled_new_states, sampled_images, sampled_new_images, sampled_actions_probs, sampled_rewards, sampled_terminated

    def reset(self):
        self.states = torch.zeros((self.size+1, self.state_size), device=self.device)
        self.actions_probs = torch.zeros((self.size, self.action_size), device=self.device)
        self.rewards = torch.zeros((self.size, 1), device=self.device)
        self.terminated = torch.zeros((self.size, 1), dtype=torch.bool, device=self.device)
        if self.image_size is not None:
            self.images = torch.zeros((self.size+1, (self.size_histo+1)*self.image_size[0], self.image_size[1], self.image_size[2]), device=self.image_device)

    def save(self):
        if not os.path.isdir(self.file_save):    
            os.makedirs(self.file_save)

        torch.save(self.states, "{}/states.pt".format(self.file_save))
        torch.save(self.actions_probs, "{}/actions.pt".format(self.file_save))
        torch.save(self.rewards, "{}/rewards.pt".format(self.file_save))
        torch.save(self.terminated, "{}/dones.pt".format(self.file_save))
        if self.image_size is not None:
            torch.save(self.images, "{}/images.pt".format(self.file_save))


    def load(self):
        self.states = torch.load("{}/states.pt".format(self.file_save))
        self.actions_probs = torch.load("{}/actions.pt".format(self.file_save))
        self.rewards = torch.load("{}/rewards.pt".format(self.file_save))
        self.terminated = torch.load("{}/dones.pt".format(self.file_save))
        if self.image_size is not None:
            self.images = torch.load("{}/images.pt".format(self.file_save))