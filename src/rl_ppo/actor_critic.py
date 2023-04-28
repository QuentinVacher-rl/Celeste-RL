"""ActorCritic network for ppo class file
"""

import numpy as np

import torch
import torch.nn as nn

from rl_ppo.config_ppo import ConfigPPO


class ActorCritic(nn.Module):
    def __init__(self, action_size, observation_size, image_size, histo_size, device, config: ConfigPPO):
        super(ActorCritic, self).__init__()


        self.state_size = observation_size
        self.action_size = action_size
        self.size_image = image_size

        if self.size_image is not None:
            self.base_image = nn.Sequential(
                nn.Conv2d((histo_size+1)*self.size_image[0], 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 16, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )
             # Divide and minus three times by 2 because maxpooling, multiply by 16 with 16 output filter
            size_output_image = int(16 * np.prod(np.trunc(np.trunc(np.trunc(np.trunc((self.size_image[1:3] - 2)/2-2)/2-2)/2-2)/2)))
        else:
            size_output_image = 0

        self.base = nn.Sequential(
            nn.Linear(size_output_image + observation_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

        self.mu_actor = nn.Linear(config.hidden_size, action_size)

        self.var_actor = nn.Sequential(
            nn.Linear(config.hidden_size, action_size),
            nn.Softplus()
        )

        self.critic = nn.Linear(config.hidden_size, 1)

        self.noise_value = 1e-6

        
        self.device = device
        self.to(device)

    def forward(self, x, y):
        if self.size_image is not None:
            y = self.base_image(y)
            x = torch.cat((x, y), dim=1)
        else:
            x = self.base(x)

        return self.mu_actor(x), self.var_actor(x), self.critic(x)
