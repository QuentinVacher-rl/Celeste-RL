"""File for the general config
"""

import numpy as np

from utils.screen_info import ScreenInfo
import torch

class Config:
    """Class for the general config
    """

    def __init__(self) -> None:

        # GLOBAL CONFIG

        # -------------------------------------------

        # Total number of episode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.nb_learning_step = 10_000

        # Max step per episode
        self.max_steps = 150

        # Train episode per learning step
        self.nb_train_episode = 100

        # Test episode per learning step
        self.nb_test_episode = 10

        # -------------------------------------------



        # FUNCTIONNING ENVIRONNEMENT CONFIGURATION

        # -------------------------------------------

        # Number of frames per action
        self.nb_frame_action = 5

        self.screen_used = [1]

        # Tas file to run for init the first screen
        self.init_tas_file = "console load 1 {}\n   38\n\n# end\n   1"

        self.screen_info = [
            ScreenInfo(
                screen_id=1,
                start_position = [[19, 144], [90, 128], [160, 80],  [260, 56]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=308, x_min=12,
                y_max=195, y_min=0,
                list_step_reward=[[[80, 100], [108, 128]], [[140, 180], [60, 80]], [[250, 270], [36, 56]]],
                goal=[[ 250, 280], [0, 0]]
            ),
            ScreenInfo(
                screen_id=2,
                start_position = [[264, -24], [360, -50], [445, -85], [530, -80], [530, -160]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=540, x_min=252,
                y_max=0, y_min=-190,
                list_step_reward=[[[350, 370], [-60, -40]], [[430, 460], [-100, -70]], [[515, 540], [-85, -64]]],
                goal=[[ 516, 540], [-190, -190]]
            )
        ]

        # Basic waiting time (equal to 1 frame)
        self.sleep = 0.017

        # Path of TAS file
        self.path_tas_file = "C:\\Code python\\Celeste\\file.tas"

        # Reduction factor of image screen
        self.reduction_factor = 8

        # True to indicate that size image could be wrong
        self.allow_change_size_image = False
        # Size of the screenshoted image after pooling
        self.size_image = np.array([3, 68, 120])

        # -------------------------------------------



        # INTERACTION ENVIRONNEMENT CONFIGURATION

        # -------------------------------------------

        # Action size vector
        self.action_size = np.array([3,3,2,3,2])
        # 9 for dashes in each direction
        # 9 for right, left, up, down + diagonals
        # 2 for jump
        # 2 for hold/climb

        # Base size of observation
        self.base_observation_size = 6

        # True if the action are given in the observation
        self.give_former_actions = False

        # If True, the base size of observation is bigger
        if self.give_former_actions:
            self.base_observation_size = self.base_observation_size + len(self.action_size)

        # Quantity of former iteration state and action (if action given) put if the observation vector
        self.histo_obs = 0

        # Calculate the real size of observation
        self.observation_size = (self.histo_obs + 1) * self.base_observation_size

        # True if the goal coordinate are given
        self.give_goal_coords = False

        # If True, add 4 because the coordinate are 2 X value and 2 Y value
        if self.give_goal_coords:
            self.observation_size += 4

        # True if the index of the screen is give
        self.give_screen_value = False

        # Actualise the obseration size
        if self.give_screen_value:
            self.observation_size += 1

        # Reward for death
        self.reward_death = -0.5

        # Reward when step reached
        self.reward_step_reached = 0

        # Reward chen screen passed
        self.reward_screen_passed = 100

        # Reward when nothing append
        self.natural_reward = -0.5

        # True if the image is used for learning
        self.use_image = False


        # -------------------------------------------



        # METRICS CONFIGURATION

        # -------------------------------------------

        self.val_test = 100

        self.color_graph = {
            "Death": "#1f77b4",
            "Level passed": "#2ca02c",
            "Unfinished": "#ff7f0e",
            "Step 1": "#d62728",
            "Step 2": "#9467bd",
            "Step 3": "#8c564b"
        }

        self.limit_restore = 3000

        # -------------------------------------------
