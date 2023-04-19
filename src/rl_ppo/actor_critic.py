"""ActorCritic network for ppo class file
"""

import torch
import torch.nn as nn

from rl_ppo.config_ppo import ConfigPPO


class ActorCritic(nn.Module):
    def __init__(self, action_size, observation_size, image_size, device, config: ConfigPPO):
        super(ActorCritic, self).__init__()

        self.to(device)

        self.state_size = observation_size
        self.action_size = action_size
        self.size_image = image_size

        if self.size_image is not None:
            self.base_image = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.Flatten()
            )
             # Divide to times by 4 because maxpooling, multiply by 16 with 16 output filter
            size_output_image = int(self.size_image[1] * self.size_image[2] / 4 / 4 * 16)
        else:
            size_output_image = 0

        self.base = nn.Sequential(
            nn.Linear(size_output_image + observation_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

        self.mu_actor = nn.Sequential(
            nn.Linear(config.hidden_size, action_size),
            nn.Tanh()
        )

        self.var_actor = nn.Sequential(
            nn.Linear(config.hidden_size, action_size),
            nn.Softplus()
        )

        self.critic = nn.Linear(config.hidden_size, 1)

        self.mode = "train"


    def forward(self, x, y):

        y = self.base_image(y)
        x = torch.cat((x, y), dim=1)
        x = self.base(x)

        return self.mu_actor(x), self.var_actor(x), self.critic(x)
