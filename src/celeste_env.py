"""Celeste environnement
"""

import time
import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy.signal import convolve2d
import dxcam

from config import Config

class CelesteEnv():
    """Class for celest environnement
    """

    def __init__(self, config: Config):

        # Create config
        self.config = config

        # Action and observation size
        self.action_size = config.action_size
        self.observation_size = config.observation_size

        # Observation vector
        self.observation = np.zeros(self.observation_size)

        # Index to start the information of position, speed and stamina (decay of 4 if the goal coords are given)
        self.index_start_obs = 4 if config.give_goal_coords else 0

        # Tas file (line) send to Celeste
        self.current_tas_file = ""

        # Current step
        self.current_step = 0

        # index of the reward step
        self.reward_step = config.reward_step

        # inGame iteration
        self.game_step = 0

        # Current screen (panel)
        self.current_screen = 1


        # Position x and y of Madeline
        self.pos_x = 0
        self.pos_y = 0

        # True if Madeline is dead
        self.dead = False

        # True if the screen is paster
        self.screen_pasted = False

        # True if the reward step is reached
        self.step_reached = False

        # True if the mid-reward step is reached
        self.mid_step_reached = False

        # Object initiate for screen shot
        if config.use_image:
            self.camera = dxcam.create()


    def step(self, actions):
        """Step method

        Args:
            actions (np.array): Array of actions

        Returns:
            tuple: New state, reward, done and info
        """
        # Incremente the step
        self.current_step += 1

        # Init the frame with the quantity of frames/step
        frame_to_add = f"   {self.config.nb_frame_action}"

        # Add the corresponding actions (See CelesteTAS documentation for explanation)
        if actions[0] == 1 or actions[0] == 2 or actions[0] == 8:
            frame_to_add += ",R"
        if actions[0] == 4 or actions[0] == 5 or actions[0] == 6:
            frame_to_add += ",L"
        if actions[0] == 2 or actions[0] == 3 or actions[0] == 4:
            frame_to_add += ",U"
        if actions[0] == 6 or actions[0] == 7 or actions[0] == 8:
            frame_to_add += ",D"

        if actions[0] != 0:
            frame_to_add += ",X"

        if actions[1] == 1 or actions[1] == 2 or actions[1] == 8:
            frame_to_add += ",R"
        if actions[1] == 4 or actions[1] == 5 or actions[1] == 6:
            frame_to_add += ",L"
        if actions[1] == 2 or actions[1] == 3 or actions[1] == 4:
            frame_to_add += ",U"
        if actions[1] == 6 or actions[1] == 7 or actions[1] == 8:
            frame_to_add += ",D"

        if actions[2] == 1:
            frame_to_add += ",J"

        if actions[3] == 1:
            frame_to_add += ",G"

        # Add the frame to the current tas file
        self.current_tas_file += frame_to_add + "\n"

        # Sometimes there is a Exception PermissionError access to the file
        # The while / Try / Except is to avoid the code crashing because of this
        changes_make = False
        while not changes_make:
            try:
                # Rewrite the tas file with the frame
                with open("file.tas", "w+", encoding="utf-8") as file:
                    file.write(frame_to_add + "\n\n# end\n   1")
                changes_make = True
            except PermissionError:
                # If error, wait 1 second
                time.sleep(1)

        # Run the tas file
        requests.get("http://localhost:32270/tas/playtas?filePath={}".format(self.config.path_tas_file))

        # Fast Forward to the end of the action to save execution time
        requests.get("http://localhost:32270/tas/sendhotkey?id=FastForwardComment")


        # Get observation and done info
        observation, done= self.get_madeline_info()

        # Roll the observation array (because we want to save an historic of the former actions and observations)
        self.observation[self.index_start_obs:] = np.roll(self.observation[self.index_start_obs:], self.config.base_observation_size, axis=0)

        # If the actions are put in the observation vector
        if self.config.give_former_actions:

            # Add them to the observation vector
            observation[self.config.base_observation_size - len(self.action_size):] = actions

        # Now add the current observation
        self.observation[self.index_start_obs:self.config.base_observation_size+self.index_start_obs] = observation

        # Create the observation vector
        obs_vect = [self.observation[np.newaxis, ...]]

        # If the image of the game is used
        if self.config.use_image:

            # Get the array of the screen
            screen_obs = self.get_image_game()

            # Create the observation vector with the screen in
            obs_vect = [screen_obs, self.observation[np.newaxis, ...]]


        # Available actions avec 1 on every possible actions, 0 on every impossible ones
        available_actions = [np.ones(current_action) for current_action in self.action_size]

        # if the dash is not available
        if observation[5] == 0:
            available_actions[0][1:] = 0

        # Get the reward
        reward = self.get_reward()

        # No info passed but initiate for convention
        info = {}

        return obs_vect, reward, done, available_actions, info

    def reset(self, level:int=1):
        """Reset the environnement

        Args:
            level (int): Current screen. Defaults to 1.

        Returns:
            tuple: New state, reward
        """

        # Init the screen
        self.current_screen = level

        # Init the current tas file
        self.current_tas_file = self.config.init_tas_file

        # Init the current step
        self.current_step = 0

        # True if Madeline is dead
        self.dead = False

        # True if the screen is paster
        self.screen_pasted = False

        # True if the reward step is reached
        self.step_reached = False

        # Wait a bit to avoid problems
        time.sleep(0.017*3)

        # Write the tas file
        with open(file="file.tas", mode="w", encoding="utf-8") as file:
            file.write(self.config.init_tas_file)

        # Run it
        requests.get("http://localhost:32270/tas/playtas?filePath={}".format(self.config.path_tas_file))

        # Wait a bit, again..
        time.sleep(0.06)

        # Fast Forward
        requests.get("http://localhost:32270/tas/sendhotkey?id=FastForwardComment")

        # Init the game step
        # With the init tas file given, Madeline take it first action at the 6st frame
        self.game_step = 0
        while self.game_step != 6:
            time.sleep(0.06)

            # Wait, always wait

            # Init the l_text
            l_text = [""]

            # If Timer is in the text, that mean Celeste have rightfully start
            # Sometimes it has start but the delay between FastForward and getting info is too short
            while "Timer" not in "".join(l_text):

                # Get the information of the game
                response = requests.get("http://localhost:32270/tas/info")

                # Get the corresponding text
                l_text = BeautifulSoup(response.content, "html.parser").text.split("\n")

            # Normally Timer appear on -8 index, but sometimes there is a "Cursor" info that can crash the code
            # If "Timer" is not on -8 index, we just check all the index
            if "Timer" in l_text[-8]:
                # Get game step
                self.game_step = int(l_text[-8].replace(")","").split("(")[1])
            else:
                for line in l_text:
                    if "Timer" in line:
                        # Get game step
                        self.game_step = int(line.replace(")","").split("(")[1])

        # Strange behaviour but it work better when game step is reset to 0 and will be update in the next function
        self.game_step = 0

        # Get the observation of Madeline, no use for Done because it can not be True at reset
        observation, _ = self.get_madeline_info()

        # If the goal coords are given, put it in the observation vector
        if self.config.give_goal_coords:
            # Get the two coords for X and Y (coords create a square goal)
            reward_goal_x = np.array(self.config.list_step_reward[self.config.reward_step][0])
            reward_goal_y = np.array(self.config.list_step_reward[self.config.reward_step][1])

            # Make sure to normalize the values
            self.observation[0:2] = (reward_goal_x - self.config.x_min) / (self.config.x_max - self.config.x_min)
            self.observation[2:4] = (reward_goal_y - self.config.y_min) / (self.config.y_max - self.config.y_min)

        # Insert the observation
        self.observation[self.index_start_obs:self.config.base_observation_size+self.index_start_obs] = observation

        # Create the observation vector
        obs_vect = [self.observation[np.newaxis, ...]]

        # If the image of the game is used
        if self.config.use_image:

            # Get the array of the screen
            screen_obs = self.get_image_game()

            # Create the observation vector with the screen in
            obs_vect = [screen_obs, self.observation[np.newaxis, ...]]

        # Available actions avec 1 on every possible actions, 0 on every impossible ones
        available_actions = [np.ones(current_action) for current_action in self.action_size]


        return obs_vect, available_actions, False

    def render(self):
        """Render method
        """
        # Do not know if I will need it anytime


    def get_image_game(self):
        """Get a np array of the current screen

        Returns:
            np.array: array of the current screen
        """
        # Coordonates correspond to the celeste window when place on the automatic render left size on windows
        # So region is for me, you have to put the pixels square of your Celeste Game
        frame = self.camera.grab(region=(1,266,959,804))

        # Sometimes the frame is None because it need more time to refresh, so wait..
        while frame is None:
            time.sleep(0.05)
            frame = self.camera.grab(region=(1,266,959,804))

        # I am not sure exactly but it is not RGB but RBG or something like that so you have to invert two columns
        # It is not really usefull for the IA, only if you want to save the screen
        frame = frame[:, :, ::-1]

        # This line is used for save the screen, you have to import cv2 also
        #cv2.imwrite('screen.png', frame)


        # Definition of block size to reduce the size of the image
        block_size = 12

        # Calcul new shape
        height, width, _ = frame.shape
        new_h = (height + block_size - 1) // block_size * block_size
        new_w = (width + block_size - 1) // block_size * block_size

        # add padding
        top_pad = (new_h - height) // 2
        bottom_pad = new_h - height - top_pad
        left_pad = (new_w - width) // 2
        right_pad = new_w - width - left_pad
        frame_padded = np.pad(frame, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')

        # Definition of convolution mask for max pooling
        mask = np.ones((block_size, block_size, 3))
        mask /= block_size * block_size

        # Calcul of max pooling with the convolution method
        new_img = np.zeros((new_h // block_size, new_w // block_size, 3), dtype=np.uint8)
        for color in range(3):
            pooled = convolve2d(frame_padded[:, :, color], mask[:, :, color], mode='valid')
            new_img[:, :, color] = np.round(np.abs(pooled[::block_size, ::block_size]))

        # Convertion of the datatype
        new_img = new_img.astype(np.uint8)

        # Delete the padding
        new_img = new_img[top_pad//block_size:new_h//block_size-top_pad//block_size,
                        left_pad//block_size:new_w//block_size-left_pad//block_size]

        # Normalize the screen
        screen = new_img / 255

        # Add a new axis (for the RL model)
        screen = screen[np.newaxis, ...]
        return screen


    def get_madeline_info(self):
        """Get the observation of madeline

        Args:
            reset (bool): True if the method is used during reset. Defaults to False.

        Returns:
            np.array, bool: observation and done
        """
        # Init the observation vector
        observation = np.zeros(self.config.base_observation_size)

        # Save the former game step
        former_game_step = self.game_step

        # While the times are the same
        while former_game_step == self.game_step:

            # Wait a bit
            time.sleep(0.03)

            # Get the info of Celeste
            response = requests.get("http://localhost:32270/tas/info")

            # Get the corresponding text
            l_text = BeautifulSoup(response.content, "html.parser").text.split("\n")

            # Normally Timer appear on -8 index, but sometimes there is a "Cursor" info that can crash the code
            # If "Timer" is not on -8 index, we just check all the index
            if "Timer" in l_text[-8]:
                # Get game step
                self.game_step = int(l_text[-8].replace(")","").split("(")[1])
            else:
                for line in l_text:
                    if "Timer" in line:
                        # Get game step
                        self.game_step = int(line.replace(")","").split("(")[1])


        # Get the observation information, not gonna detail those part because it is just the string interpretation
        # Run "http://localhost:32270/tas/info" on a navigator to understand the information gotten

        # Position is on index 11
        pos = l_text[11].split(' ')[-2:]
        self.pos_x = float(pos[0].replace(",",""))
        self.pos_y = float(pos[1].replace("\r", ""))

        # Normalise the information
        observation[0] = (self.pos_x  - self.config.x_min) / (self.config.x_max - self.config.x_min)
        observation[1] = (self.pos_y  - self.config.y_min) / (self.config.y_max - self.config.y_min)

        # Only speed is get for now, maybe velocity will be usefull later
        speed = l_text[12].split()
        observation[2] = float(speed[1].replace(",", ""))/10
        observation[3] = float(speed[2])/10

        # Stamine
        observation[4] = float(l_text[14].split()[1])/110

        # Disponibility of dash
        dispo_dash = ("CanDash" in l_text[15] or "CanDash" in l_text[16])
        observation[5] = 1 if dispo_dash else 0

        # Init done at False
        done = False

        # Reset mid step reached at False
        self.mid_step_reached = False

        # If reward step is reached
        if (
                self.pos_x >= self.config.list_step_reward[self.reward_step][0][0]
                and self.pos_x <= self.config.list_step_reward[self.reward_step][0][1]
                and self.pos_y >= self.config.list_step_reward[self.reward_step][1][0]
                and self.pos_y <= self.config.list_step_reward[self.reward_step][1][1]
        ):
            done = True
            self.step_reached = True

        # If mid reward step is reached
        elif (
                self.reward_step > 0
                and self.pos_x >= self.config.list_step_reward[self.reward_step-1][0][0]
                and self.pos_x <= self.config.list_step_reward[self.reward_step-1][0][1]
                and self.pos_y >= self.config.list_step_reward[self.reward_step-1][1][0]
                and self.pos_y <= self.config.list_step_reward[self.reward_step-1][1][1]
        ):
            self.mid_step_reached = True

        # If dead
        if "Dead" in "".join(l_text[-10:-8]):
            done = True
            self.dead = True

        # If screen pasted. Only on screen 1 for now, will be change later
        if "[2]" in l_text[-8] and self.current_screen == 1:
            self.screen_pasted = True

        return observation, done

    def get_reward(self):
        """Get the reward

        Returns:
            int: Reward
        """
        # If dead reward is 0
        if self.dead:
            return 0

        # If step reached, reward is 50
        if self.step_reached:
            return 50

        # If mid step reached, reward is 1
        elif self.mid_step_reached:
            return 1

        # Else reward is 0
        return 0
