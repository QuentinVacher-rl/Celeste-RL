"""Config of Multiple DQN file
"""

class ConfigActorCritic:
    """Class for the config of Multiple  DQN
    """

    def __init__(self) -> None:

        # Learn each *this value* of episodes
        self.nb_episode_learn = 1

        # True if the networks are restored
        self.restore_networks = False

        # Gamma value
        self.discount_factor = 0.95

        # Standardize the return reward
        self.standardize = False

        # Learning rate configuration
        self.init_lr = 0.01
        self.nb_episode_lr_decay = 3000
        self.lr_decay = 0.5
        self.min_lr = 0.001
