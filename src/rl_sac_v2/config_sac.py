"""Config of SAC file
"""

class ConfigSac:
    """Class for the config of SAC
    """

    def __init__(self) -> None:



        self.use_image_train = True
        self.only_image_actor = True

        self.discount_factor = 0.99
        self.tau = 0.005
        self.batch_size = 256000
        self.epoch = 1
        self.frequency_training = 1

        self.lr = 3e-4
        self.hidden_size = 1024
        self.size_buffer = 10_000

        

        self.init_entropy = 2
        self.restore_networks = False
        self.file_save_network = "src/rl_sac_v2/network"
        self.file_save_memory = "src/rl_sac_v2/memory"

        self.noise_value = 1e-6
