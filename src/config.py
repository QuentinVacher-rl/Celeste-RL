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
        self.y_min = 0

        # Basic waiting time (equal to 1 frame)
        self.sleep = 0.017

        # Tas file to run for init the first screen
        self.init_tas_file = "console load 1 1\n   95\n\n# end\n   1"

        # Path of TAS file
        self.path_tas_file = "C:\\Code python\\Celeste\\file.tas"


        # Size of the screenshoted image after pooling
        self.size_image = np.array([45, 80, 3])

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
            self.base_observation_size = self.base_observation_size + len(self.action_size)

        # Quantity of former iteration state and action (if action given) put if the observation vector
        self.histo_obs = 3

        # Calculate the real size of observation
        self.observation_size = (self.histo_obs + 1) * self.base_observation_size

        # True if the goal coordinate are given
        self.give_goal_coords = True

        # If True, add 4 because the coordinate are 2 X value and 2 Y value
        if self.give_goal_coords:
            self.observation_size += 4

        # List of all step reward
        self.list_step_reward = [
            [[80, 100], [108, 128]],
            [[140, 180], [60, 80]],
            [[250, 270], [36, 56]]
        ]

        # Reward for death
        self.reward_death = 0

        # Reward when step reached
        self.reward_step_reached = 1

        # Reward chen screen passed
        self.reward_screen_passed = 50

        # Reward when nothing append
        self.natural_reward = 0

        # Coordonates of the goal
        self.goal = [[ 250, 280], [0, 0]]

        # True if the image is used for learning
        self.use_image = False

        # -------------------------------------------



        # METRICS CONFIGURATION

        # -------------------------------------------

        self.val_test = 50

        self.color_graph = {
            "Death": "#1f77b4",
            "Level passed": "#2ca02c",
            "Unfinished": "#ff7f0e",
            "Step 1": "#d62728",
            "Step 2": "#9467bd",
            "Step 3": "#8c564b"
        }

        # -------------------------------------------
