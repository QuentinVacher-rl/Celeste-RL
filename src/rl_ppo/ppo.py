"""Multiple DQN class file
"""

from rl_ppo.buffer import Buffer
from rl_ppo.actor_critic import ActorCritic
from rl_ppo.config_ppo import ConfigPPO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from config import Config


# Définir le modèle de réseau neuronal (Q-value)

class PPO(nn.Module):
    def __init__(self, config_ppo: ConfigPPO, config_env: Config):
        super(PPO, self).__init__()

        self.device = config_env.device
        self.to(self.device)

        self.action_mode = "Continuous"

        self.config = config_ppo

        self.save_file = self.config.file_save + "/network.pt"
        self.discount_factor = self.config.discount_factor
        self.standardize = self.config.standardize

        self.coef_entropy = self.config.coef_entropy
        self.coef_critic = self.config.coef_critic

        self.nb_epochs = self.config.nb_epochs

        self.state_size = config_env.observation_size
        self.action_size = config_env.action_size.shape[0]
        self.image_size = config_env.size_image

        self.buffer = Buffer(size=config_env.max_steps, action_size=self.action_size, state_size=self.state_size, image_size=self.image_size)

        self.clip_value = self.config.clip_value

        self.actor_critic = ActorCritic(self.action_size, self.state_size, self.image_size, self.device, self.config)

        if self.config.restore:
            self.load_model()

        self.optimizer = torch.optim.Adam(params=list(self.actor_critic.parameters()).copy(), lr=self.config.lr)




    def choose_action(self, state, image):

        # Formate state
        image = torch.tensor(image, dtype=torch.float32).permute(0, 3, 1, 2)
        state = torch.tensor(state, dtype=torch.float32)

        mu, var, value = self.actor_critic(state, image)

        dist = tdist.Normal(mu, torch.sqrt(var))
        action_probs = torch.clip(dist.sample(), -1, 0.999)
        log_prob = dist.log_prob(action_probs)

        #action = 0 if action_prob < 0 else 1

        return action_probs, log_prob.detach(), value.detach()


    def insert_data(self, state, image, actions, log_prob, value, reward, done):
        self.buffer.insert_data(state, image, actions, log_prob, value, reward, done)

    def train(self, episode: int):

        returns = self.get_expected_returns()

        old_actions = torch.tensor(self.buffer.actions)

        old_action_log_probs = torch.tensor(self.buffer.log_prob)
        old_advantage = returns - torch.tensor(self.buffer.values)

        for _ in range(self.nb_epochs):

            t_states = torch.tensor(self.buffer.states, requires_grad=True, dtype=torch.float32)
            t_images = torch.tensor(self.buffer.images, requires_grad=True, dtype=torch.float32).permute(0, 3, 1, 2)
            mu, var, values = self.actor_critic(t_states, t_images)
            

            distribution = tdist.Normal(mu, torch.sqrt(var))
            entropy = distribution.entropy()
            action_log_probs = distribution.log_prob(old_actions)

            ratio = torch.exp(action_log_probs - old_action_log_probs)

            unclipped_loss = ratio * old_advantage
            clipped_loss = torch.clip(ratio, 1-self.clip_value, 1+self.clip_value) * old_advantage

            actor_loss = torch.min(unclipped_loss, clipped_loss)
            critic_loss = F.mse_loss(values, returns, reduction="none")

            loss = torch.mean(-actor_loss + self.coef_critic * critic_loss + self.coef_entropy * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.reset()


    def save_model(self):
        torch.save(self.actor_critic.state_dict(), "network/model.pt")
    
    def load_model(self):
        model_state_dict = torch.load("network/model.pt")
        self.actor_critic.load_state_dict(model_state_dict)

    def get_expected_returns(self):

        t_rewards = torch.tensor(self.buffer.rewards).flip(0)
        t_dones = torch.tensor(self.buffer.dones).flip(0)
        returns = torch.zeros(t_rewards.size())

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        discounted_sum = 0
        for index in range(t_rewards.shape[0]):
            if t_dones[index]:
                discounted_sum = 0
            reward = t_rewards[index]
            discounted_sum = reward + self.discount_factor * discounted_sum
            returns[index] = discounted_sum
        returns = returns.flip(0)

        if self.standardize:
            returns = ((returns - torch.mean(returns)) / 
                    (torch.std(returns) + 10e-7))

        return returns

