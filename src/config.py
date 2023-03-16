"""File for the general config
"""

import numpy as np

class Config:
    """Class for the general config
    """

    def __init__(self) -> None:

        # GLOBAL CONFIG

        # -------------------------------------------

        # Total number of episode
        self.num_episodes = 100_000

        # Max step per episode
        self.max_steps = 200

        # -------------------------------------------



        # FUNCTIONNING ENVIRONNEMENT CONFIGURATION

        # -------------------------------------------

        # Number of frames per action
        self.nb_frame_action = 5

        # Coordinates maximal and minimal of x in 1st screen
        self.x_max = 308
        self.x_min = 12

        # Coordinates maximal and minimal of x in 1st screen
        self.y_max = 195
        self.y_min = 16

        # Basic waiting time (equal to 1 frame)
        self.sleep = 0.017

        # Tas file to run for init the first screen
        self.init_tas_file = "console load 1 1\n   95\n\n# end\n   1"

        # Path of TAS file
        self.path_tas_file = "D:\\Taffe\\perso\\Celeste\\file.tas"

        # Size of the screenshoted image after pooling
        self.size_image = np.array([45,80, 3])

        # -------------------------------------------



        # INTERACTION ENVIRONNEMENT CONFIGURATION

        # -------------------------------------------

        # Action size vector
        self.action_size = np.array([9, 9, 2, 2])
        # 9 for dashes in each direction
        # 9 for right, left, up, down + diagonals
        # 2 for jump
        # 2 for hold/climb

        # Base size of observation
        self.base_observation_size = 6

        # True if the action are given in the observation
        self.give_former_actions = True

        # If True, the base size of observation is bigger
        if self.give_former_actions:
            self.base_observation_size = (self.base_observation_size + len(self.action_size))

        # Quantity of former iteration state and action (if action given) put if the observation vector
        self.histo_obs = 3

        # Calculate the real size of observation
        self.observation_size = (self.histo_obs + 1) * self.base_observation_size

        # True if the goal coordinate are given
        self.give_goal_coords = True

        # If True, add 4 because the coordinate are 2 X value and 2 Y value
        if self.give_goal_coords:
            self.observation_size += 4

        # List of all goal reward
        self.list_step_reward = [
            [[80, 100], [108, 128]],
            [[140, 180], [60, 80]],
            [[250, 270], [36, 56]],
            [[250, 290], [-30, 0]],
        ]

        # Init goal reward
        self.reward_step = 1

        # True if the image is used for learning
        self.use_image = True

        # -------------------------------------------



        # METRICS CONFIGURATION

        # -------------------------------------------

        self.val_test = 50

        # -------------------------------------------
