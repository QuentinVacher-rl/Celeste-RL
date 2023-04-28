"""Config of PPO file
"""

class ConfigPPO:
    """Class for the config of PPO
    """

    def __init__(self) -> None:

        self.size_buffer = 150

        self.discount_factor = 0.99
        self.standardize = True

        self.coef_entropy = 0.05
        self.coef_critic = 0.5

        self.nb_epochs = 8

        self.clip_value = 0.2

        self.lr = 0.001
        self.hidden_size = 512

        self.restore = False
        self.file_save = "src/rl_ppo/network"

