"""Config of PPO file
"""

class ConfigPPO:
    """Class for the config of PPO
    """

    def __init__(self) -> None:


        self.discount_factor = 0.995
        self.standardize = True
        self.coef_entropy = 0.01
        self.coef_critic = 0.5

        self.nb_epochs = 8

        self.clip_value = 0.2

        self.noise_value = 0.6

        self.tau = 0.005
        self.batch_size = 512
        self.epoch = 1

        self.lr = 0.001
        self.hidden_size = 512

        self.restore = True
        self.file_save = "src/rl_ppo/network"

