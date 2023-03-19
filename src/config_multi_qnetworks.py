"""Config of Multiple DQN file
"""

class ConfigMultiQnetworks:
    """Class for the config of Multiple  DQN
    """

    def __init__(self) -> None:

        # Episode to start learning
        self.start_learning = 50

        # Learn each *this value* of episodes
        self.nb_episode_learn = 1

        # Copy the targer network each *this value* of episodes
        self.nb_episode_copy_target = 5

        # Batch size when training
        self.batch_size = 512

        # Batch size of the fitting method
        self.mini_batch_size = 128

        # Capacity of the memory
        self.memory_capacity = 20_000

        # True if the networks are restored
        self.restore_networks = False

        # Gamma value
        self.discount_factor = 0.9


        # Epsilon configuration
        self.init_epsilon = 0.5
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.08

        # Learning rate configuration
        self.init_lr = 0.005
        self.nb_episode_lr_decay = 3000
        self.lr_decay = 0.5
        self.min_lr = 0.001
