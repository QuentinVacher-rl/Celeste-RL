"""Config of SAC file
"""

class ConfigSac:
    """Class for the config of SAC
    """

    def __init__(self) -> None:


        self.discount_factor = 0.993
        self.tau = 0.005
        self.batch_size = 512
        self.epoch = 1

        self.lr = 0.001
        self.hidden_size = 512
        self.size_buffer = 1000_000

        self.alpha = 2
        self.restore = False
        self.file_save = "src/rl_sac/network"

        self.noise_value = 1e-6
