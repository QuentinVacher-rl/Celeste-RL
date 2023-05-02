"""Config of SAC file
"""

class ConfigSac:
    """Class for the config of SAC
    """

    def __init__(self) -> None:


        self.discount_factor = 0.99
        self.tau = 0.005
        self.batch_size = 512
        self.epoch = 1

        self.lr = 3e-4
        self.hidden_size = 1024
        self.size_buffer = 1000_000

        self.alpha = 2
        self.restore_networks = True
        self.restore_memory = False
        self.file_save_network = "src/rl_sac/network"
        self.file_save_memory = "src/rl_sac/memory"

        self.noise_value = 1e-6
