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

        self.action_mode = "Continuous"

        self.device = config_env.device
        self.to(self.device)

        self.config = config_ppo

        self.save_file = self.config.file_save + "/network.pt"
        self.discount_factor = self.config.discount_factor
        self.standardize = self.config.standardize

        self.coef_entropy = self.config.coef_entropy
        self.coef_critic = self.config.coef_critic

        self.nb_epochs = self.config.nb_epochs

        self.size_histo = config_env.histo_image
        self.state_size = config_env.observation_size
        self.action_size = config_env.action_size.shape[0]
        self.use_image = config_env.use_image
        self.image_size = config_env.size_image if self.use_image else None

        self.buffer = Buffer(size=self.config.size_buffer, action_size=self.action_size, state_size=self.state_size, image_size=self.image_size, size_histo=self.size_histo)

        self.clip_value = self.config.clip_value

        self.actor_critic = ActorCritic(self.action_size, self.state_size, self.image_size, self.size_histo, self.device, self.config)

        if self.config.restore:
            self.load_model()

        self.optimizer = torch.optim.Adam(params=list(self.actor_critic.parameters()).copy(), lr=self.config.lr)

        self.action_raw = None
        self.log_prob = None
        self.value = None



    def choose_action(self, state, image):

        # Formate state
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        if self.use_image:
            image = torch.tensor(image, dtype=torch.float, device=self.device)
        mu, var, value = self.actor_critic(state, image)
        probs = tdist.Normal(mu, var)
        actions = probs.sample()
        action = torch.tanh(actions).to(self.device)

        self.action_raw = actions.cpu().detach().numpy()
        self.log_prob = probs.log_prob(actions).cpu().detach().numpy()
        self.value = value.cpu().detach().numpy()
        return action.cpu().detach().numpy()

    def insert_data(self, state, new_state, image, new_image, action, reward, terminated, truncated):
        self.buffer.insert_data(state, image, self.action_raw, self.log_prob, self.value, reward, terminated or truncated)

    def train(self):

        if not self.buffer.full:
            return

        returns = self.get_expected_returns()

        old_actions = torch.tensor(self.buffer.actions, device=self.device)

        old_action_log_probs = torch.tensor(self.buffer.log_prob, device=self.device)
        old_advantage = returns - torch.tensor(self.buffer.values, device=self.device)

        for _ in range(self.nb_epochs):

            t_states = torch.tensor(self.buffer.states, requires_grad=True, dtype=torch.float32, device=self.device)
            if self.use_image:
                t_images = torch.tensor(self.buffer.images, requires_grad=True, dtype=torch.float32, device=self.device)
            else:
                t_images = None

            mu, var, values = self.actor_critic(t_states, t_images)


            distribution = tdist.Normal(mu, var)
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
        torch.save(self.actor_critic.state_dict(), self.save_file)

    def load_model(self):
        self.actor_critic.load_state_dict(torch.load(self.save_file))


    def get_expected_returns(self):

        t_rewards = torch.tensor(self.buffer.rewards, device=self.device).flip(0)
        t_dones = torch.tensor(self.buffer.dones, device=self.device).flip(0)
        returns = torch.zeros(t_rewards.size(), device=self.device)

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

