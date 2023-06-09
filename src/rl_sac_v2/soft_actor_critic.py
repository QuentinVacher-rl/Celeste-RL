
from rl_sac_v2.networks import ActorNetwork, CriticNetwork
from rl_sac_v2.replay_buffer import ReplayBuffer
from rl_sac_v2.config_sac import ConfigSac
from config import Config

import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim

class ActorCritic():

    def __init__(self, config_sac: ConfigSac, config_env: Config) -> None:



        self.action_mode = "Continuous"

        self.config = config_sac
        self.size_histo = config_env.histo_image
        self.action_size = config_env.action_size.shape[0]
        self.state_size = config_env.observation_size

        self.size_image = config_env.size_image if self.config.use_image_train else None
        self.use_image = config_env.use_image

        self.gamma = self.config.discount_factor
        self.tau = self.config.tau
        self.batch_size = self.config.batch_size


        self.actor = ActorNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config)

        self.critic_1 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="critic_1")
        self.critic_2 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="critic_2")
        self.target_critic_1 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="target_critic_1")
        self.target_critic_2 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="target_critic_2")

        self.memory = ReplayBuffer(self.config.size_buffer, self.action_size, self.state_size, self.size_image, self.size_histo)


        self.target_entropy = -1 * self.action_size * torch.ones(1, device=self.actor.device)
        init_entropy_coef = self.config.init_entropy
        self.log_entropy_coef = torch.log(init_entropy_coef*torch.ones(1, device=self.actor.device)).requires_grad_(True)
        self.entropy_coef_optimizer = optim.Adam([self.log_entropy_coef], lr=self.config.lr)

        if self.config.restore_networks:
            self.load_model()


        self.update_network_parameters(init=True)

    def insert_data(self, state, new_state, image, new_image, actions_probs, reward, terminated, truncated):
        self.memory.insert_data(state, new_state, image, new_image, actions_probs, reward, terminated)

    def choose_action(self, state, image):
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)        
        if self.use_image:
            image = torch.tensor(image, dtype=torch.float).to(self.actor.device)
        action, _ = self.actor.sample_normal(state, image)
        return action.cpu().detach().numpy()

    def update_network_parameters(self, init=False):
        tau = 1 if init else self.tau

        target_critic_1_params = dict(self.target_critic_1.named_parameters())
        critic_1_params = dict(self.critic_1.named_parameters())

        for name in critic_1_params:
            critic_1_params[name] = tau*critic_1_params[name].clone() + \
                    (1-tau)*target_critic_1_params[name].clone()

        self.target_critic_1.load_state_dict(critic_1_params)


        target_critic_2_params = dict(self.target_critic_2.named_parameters())
        critic_2_params = dict(self.critic_2.named_parameters())

        for name in critic_2_params:
            critic_2_params[name] = tau*critic_2_params[name].clone() + \
                    (1-tau)*target_critic_2_params[name].clone()

        self.target_critic_2.load_state_dict(critic_2_params)

    def train(self):

        if self.memory.current_size < self.batch_size:
            return 0

        if self.memory.index % self.config.frequency_training != 0:
            return

        obs_arr, new_obs_arr, img_arr, new_img_arr, action_arr, reward_arr, dones_arr = self.memory.sample_data(self.batch_size)

        action, probs = self.actor.sample_normal(obs_arr, img_arr)


        entropy_coef = torch.exp(self.log_entropy_coef.detach())
        entropy_coef_loss = -(self.log_entropy_coef * (probs + self.target_entropy).detach()).mean()
        self.entropy_coef_optimizer.zero_grad()
        entropy_coef_loss.backward()
        self.entropy_coef_optimizer.step()

        with torch.no_grad():

            next_action, next_probs = self.actor.sample_normal(new_obs_arr, new_img_arr)

            next_q_values_1 = self.target_critic_1(new_obs_arr, next_action, new_img_arr)
            next_q_values_2 = self.target_critic_2(new_obs_arr, next_action, new_img_arr)
            next_q_values, _ = torch.min(torch.cat((next_q_values_1, next_q_values_2), dim=1), dim=1, keepdim=True)

            next_q_values = next_q_values - entropy_coef*next_probs

            target_q_values = reward_arr + (1-dones_arr) * self.gamma * next_q_values

        current_q_values_1 = self.critic_1(obs_arr, action_arr, img_arr)
        current_q_values_2 = self.critic_2(obs_arr, action_arr, img_arr)

        critic_loss_1 = 0.5 * F.mse_loss(current_q_values_1, target_q_values)
        critic_loss_2 = 0.5 * F.mse_loss(current_q_values_2, target_q_values)
        critic_loss = critic_loss_1 + critic_loss_2

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()


        q_values_1 = self.critic_1(obs_arr, action, img_arr)
        q_values_2 = self.critic_2(obs_arr, action, img_arr)
        q_values, _ = torch.min(torch.cat((q_values_1, q_values_2), dim=1), dim=1, keepdim=True)

        actor_loss = (entropy_coef*probs - q_values).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return np.round(entropy_coef.item(), 4)
    

    def save_model(self):
        self.actor.save_model()
        self.critic_1.save_model()
        self.critic_2.save_model()
        self.target_critic_1.save_model()
        self.target_critic_2.save_model()
        torch.save(self.log_entropy_coef, self.config.file_save_network + "/entropy.pt")

    def load_model(self):
        self.actor.load_model()
        self.critic_1.load_model()
        self.critic_2.load_model()
        self.target_critic_1.load_model()
        self.target_critic_2.load_model()
        self.log_entropy_coef = torch.load(self.config.file_save_network + "/entropy.pt")
