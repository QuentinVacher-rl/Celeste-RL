"""Config of SAC file
"""

class ConfigSac:
    """Class for the config of SAC
    """

    def __init__(self) -> None:


        self.activate_supervised_learning_init = False
        self.size_supervised_buffer = 65_536
        self.partition_buffer = 10
        self.nb_epochs_supervised_per_partitions = 8_192
        self.supervised_batch_size = 512

        self.save_supervised_buffer = False
        self.use_image_train = False
        self.use_image_supervised_buffer = False

        self.discount_factor = 0.99
        self.tau = 0.005
        self.batch_size = 512
        self.epoch = 1
        self.frequency_training = 1

        self.lr = 3e-4
        self.hidden_size = 1024
        self.size_buffer = 1000_000

        self.alpha = 2
        self.restore_networks = True
        self.file_save_network = "src/rl_sac/network"
        self.file_save_memory = "src/rl_sac/memory"

        self.noise_value = 1e-6
