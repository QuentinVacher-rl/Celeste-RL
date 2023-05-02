"""Celeste environnement
"""

import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
import dxcam
import cv2

from config import Config

class CelesteEnv():
    """Class for celest environnement
    """

    def __init__(self, config: Config):

        # Create config
        self.config = config

        # True if the environnement is testing, else it is training
        self.is_testing = True

        # Init max step at config value
        self.max_steps = self.config.max_steps

        # Info about the current screen
        self.screen_info = self.config.screen_info[0]

        # Action and observation size
        self.action_size = config.action_size
        self.observation_size = config.observation_size

        # Observation vector
        self.observation = np.zeros(self.observation_size)
        self.screen_obs = np.zeros(((self.config.histo_obs+1)*self.config.size_image[0], self.config.size_image[1], self.config.size_image[2]))

        # Index to start the information of position, speed and stamina (decay of 4 if the goal coords are given)
        self.index_start_obs = 0
        if config.give_goal_coords:
            self.index_start_obs += 4
        if config.give_screen_value:
            self.index_start_obs += 1

        # Tas file (line) send to Celeste
        self.current_tas_file = ""

        # Current step
        self.current_step = 0

        # inGame iteration
        self.game_step = 0

        # Position x and y of Madeline
        self.pos_x = 0
        self.pos_y = 0

        # True if Madeline is dead
        self.dead = False

        # True if the screen is pasted
        self.screen_passed = False

        # True if the wrong screen is pasted
        self.wrong_screen_passed = False



        # True if maddeline is dashing in the current step
        self.is_dashing = False

        self.action_mode = "Continuous"

        # Object initiate for screen shot
        if config.use_image or config.video_best_screen:
            self.camera = dxcam.create(output_idx=0, output_color="BGR")

        # Object initiate for create video during test
        if config.video_best_screen:
            self.screens = dict()

    def set_action_mode(self, action_mode):
        self.action_mode = action_mode

    def step(self, actions):
        """Step method

        Args:
            actions (np.array): Array of actions

        Returns:
            tuple: New state, reward, done and info
        """
        # Incremente the step
        self.current_step += 1


        # Action 0
        if self.action_mode == "Continuous":
            actions = np.trunc((actions.reshape(-1) + 1) * self.action_size / 2)


        # Init the frame with the quantity of frames/step
        frame_to_add_l1 = "   1"
        frame_to_add_l2 = f"   {self.config.nb_frame_action-1}"

        # Add the corresponding actions (See CelesteTAS documentation for explanation)
        if actions[0] == 2:
            frame_to_add_l1 += ",R"
            frame_to_add_l2 += ",R"
        if actions[0] == 0:
            frame_to_add_l1 += ",L"
            frame_to_add_l2 += ",L"
        if actions[1] == 2:
            frame_to_add_l1 += ",U"
            frame_to_add_l2 += ",U"
        if actions[1] == 0:
            frame_to_add_l1 += ",D"
            frame_to_add_l2 += ",D"

        if actions[2] == 1:
            frame_to_add_l1 += ",X"
            frame_to_add_l2 += ",X"
            self.is_dashing = True
        else:
            self.is_dashing = False

        if actions[3] == 1:
            frame_to_add_l1 += ",J"

        if actions[3] == 2:
            frame_to_add_l1 += ",J"
            frame_to_add_l2 += ",J"

        if actions[4] == 1:
            frame_to_add_l1 += ",G"
            frame_to_add_l2 += ",G"

        # Add the frame to the current tas file
        self.current_tas_file += frame_to_add_l1 + "\n" + frame_to_add_l2 + "\n" 


        # Sometimes there is a Exception PermissionError access to the file
        # The while / Try / Except is to avoid the code crashing because of this
        changes_make = False
        while not changes_make:
            try:
                # Rewrite the tas file with the frame
                with open("file.tas", "w+", encoding="utf-8") as file:
                    file.write(frame_to_add_l1 + "\n" + frame_to_add_l2 + "\n\n# end\n   1")
                changes_make = True
            except PermissionError:
                # If error, wait 1 second
                time.sleep(1)

        # Run the tas file
        requests.get("http://localhost:32270/tas/playtas?filePath={}".format(self.config.path_tas_file), timeout=5)

        # Fast Forward to the end of the action to save execution time
        requests.get("http://localhost:32270/tas/sendhotkey?id=FastForwardComment", timeout=5)


        # Get observation and done info
        observation, terminated, fail_see_death= self.get_madeline_info()

        # Roll the observation array (because we want to save an historic of the former actions and observations)
        self.observation[self.index_start_obs:] = np.roll(self.observation[self.index_start_obs:], self.config.base_observation_size, axis=0)

        # If the actions are put in the observation vector
        if self.config.give_former_actions:

            # Add them to the observation vector
            observation[self.config.base_observation_size - len(self.action_size):] = actions / self.action_size

        # Now add the current observation
        self.observation[self.index_start_obs:self.config.base_observation_size+self.index_start_obs] = observation

        # Create the observation vector
        obs_vect = np.array(self.observation[np.newaxis, ...])

        # If the image of the game is used
        screen_obs = None
        if self.config.use_image:

            # Get the array of the screen
            screen_obs = self.get_image_game()

            self.screen_obs = np.roll(self.screen_obs, 3, axis=0)
            self.screen_obs[0:3] = screen_obs

            screen_obs = np.array(self.screen_obs[np.newaxis, ...])

        if self.is_testing and self.config.video_best_screen:
            self.screen_image()

        truncated = False
        if self.current_step == self.max_steps and not terminated:
            truncated = True


        # Get the reward
        reward = self.get_reward()

        # No info passed but initiate for convention
        info = {"fail_death": fail_see_death}

        return obs_vect, screen_obs, reward, terminated, truncated, info

    def reset(self, test=False):
        """Reset the environnement

        Args:
            test (bool): True if this is a test

        Returns:
            tuple: New state, reward
        """

        self.is_testing = test

        if self.config.video_best_screen and self.is_testing:
            self.screens.clear()

        # Init the screen by choosing randomly in the screen used if not testing
        if not self.is_testing:
            self.screen_info = self.config.screen_info[np.random.choice(self.config.screen_used, p=self.config.prob_screen_used)]
        # If testing and checking all the screen, start with the first screen
        elif not self.config.one_screen:
            self.screen_info = self.config.screen_info[self.config.screen_used[0]]

        # Init the current tas file
        if self.is_testing:
            self.current_tas_file = self.screen_info.get_true_start()
        elif self.config.start_pos_only:
            self.current_tas_file = self.screen_info.get_true_start()
        else:
            self.current_tas_file = self.screen_info.get_random_start()


        # Init the current step
        self.current_step = 0

        # Init max step at config value
        self.max_steps = self.config.max_steps

        # True if Madeline is dead
        self.dead = False

        # True if the screen is pasted
        self.screen_passed = False

        # True if the wrong screen is pasted
        self.wrong_screen_passed = False

        # Wait a bit to avoid problems
        time.sleep(self.config.sleep*3)

        # Write the tas file
        with open(file="file.tas", mode="w", encoding="utf-8") as file:
            file.write(self.current_tas_file)

        # Run it
        requests.get("http://localhost:32270/tas/playtas?filePath={}".format(self.config.path_tas_file), timeout=0.5)

        # Wait a bit, again..
        time.sleep(self.config.sleep)

        # Fast Forward
        requests.get("http://localhost:32270/tas/sendhotkey?id=FastForwardComment", timeout=0.5)

        # Init the game step
        # With the init tas file given, Madeline take it first action at the 6st frame
        self.game_step = 0
        while self.game_step == 0:
            time.sleep(self.config.sleep*3)

            # Wait, always wait

            # Init the l_text
            l_text = [""]

            # If Timer is in the text, that mean Celeste have rightfully start
            # Sometimes it has start but the delay between FastForward and getting info is too short
            while "Timer" not in "".join(l_text):

                # Get the information of the game
                response = requests.get("http://localhost:32270/tas/info", timeout=5)

                # Get the corresponding text
                l_text = BeautifulSoup(response.content, "html.parser").text.split("\n")

            # Normally Timer appear on -8 index, but sometimes there is a "Cursor" info that can crash the code
            # If "Timer" is not on -8 index, we just check all the index
            if len(l_text) > 8 and "Timer" in l_text[-8]:
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
        observation, _, _ = self.get_madeline_info(reset=True)

        # If the goal coords are given, put it in the observation vector
        if self.config.give_goal_coords:
            # Get the two coords for X and Y (coords create a square goal)
            reward_goal_x = np.array(self.screen_info.goal[0])
            reward_goal_y = np.array(self.screen_info.goal[1])

            # Make sure to normalize the values
            self.observation[0:2] = self.screen_info.normalize_x(reward_goal_x)
            self.observation[2:4] = self.screen_info.normalize_y(reward_goal_y)

        if self.config.give_screen_value:
            screen_value = self.screen_info.screen_value
            index = self.index_start_obs - 1
            self.observation[index] = screen_value / self.config.max_screen_value

        # Insert the observation
        for index in range(self.config.histo_obs+1):
            index_start = self.index_start_obs + self.config.base_observation_size * index
            index_end = self.index_start_obs + self.config.base_observation_size * (index + 1)
            self.observation[index_start:index_end] = observation

        # Create the observation vector
        obs_vect = np.array(self.observation[np.newaxis, ...])

        # If the image of the game is used
        screen_obs = None
        if self.config.use_image:

            # Get the array of the screen
            screen_obs = self.get_image_game()

            # Duplicate to match historic size
            for index in range(self.config.histo_obs+1):
                self.screen_obs[index*3:(index+1)*3] = screen_obs

            screen_obs = np.array(self.screen_obs[np.newaxis, ...])

        return obs_vect, screen_obs, False, False

    def change_next_screen(self):
        """Change the screen Maddeline is in.
        To use only if it is node the last screen.
        """
        # Init the screen by choosing randomly in the screen used
        self.screen_info = self.config.screen_info[self.screen_info.screen_value + 1]

        # Add the necessary step to the next screen
        self.max_steps += self.current_step

        # If the goal coords are given, put it in the observation vector
        if self.config.give_goal_coords:
            # Get the two coords for X and Y (coords create a square goal)
            reward_goal_x = np.array(self.screen_info.goal[0])
            reward_goal_y = np.array(self.screen_info.goal[1])

            # Make sure to normalize the values
            self.observation[0:2] = self.screen_info.normalize_x(reward_goal_x)
            self.observation[2:4] = self.screen_info.normalize_y(reward_goal_y)

        if self.config.give_screen_value:
            screen_value = self.screen_info.screen_value
            index = self.index_start_obs - 1
            self.observation[index] = screen_value / self.config.max_screen_value

    def render(self):
        """Render method
        """
        # Do not know if I will need it anytime

    def screen_image(self):
        """Screen the current image of the game
        """
        # Capture the current image
        screen = self.camera.grab(region=self.config.region)

        while screen is None:
            time.sleep(0.05)
            screen = self.camera.grab(region=self.config.region)

        # Add the screen
        self.screens[self.game_step] = screen

    def get_image_game(self, normalize:bool=True):
        """Get a np array of the current screen

        Args:
            normalize (bool): True to normalize array, default=1

        Returns:
            np.array: array of the current screen
        """
        # Coordonates correspond to the celeste window when place on the automatic render left size on windows
        # So region is for me, you have to put the pixels square of your Celeste Game
        frame = self.camera.grab(region=self.config.region)

        # Sometimes the frame is None because it need more time to refresh, so wait..
        while frame is None:
            time.sleep(0.05)
            frame = self.camera.grab(region=self.config.region)

        # I am not sure exactly but it is not RGB but RBG or something like that so you have to invert two columns
        # It is not really usefull for the IA, only if you want to save the screen
        frame = frame[:, :, ::-1]

        # Add a new axis (for the RL model)
        frame = frame[np.newaxis, ...]

        # Definition of pooling size to reduce the size of the image
        pooling_size = self.config.reduction_factor

        frame = tf.nn.max_pool(frame, ksize=pooling_size, strides=pooling_size, padding='SAME')
        frame = tf.transpose(frame, perm=[0, 3, 1 ,2]).numpy()

        # Normalize the screen
        if normalize:
            frame = frame / 255

        return frame


    def get_madeline_info(self, reset=False):
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

        step_searched = former_game_step + self.config.nb_frame_action

        step_searched_dash = step_searched - self.is_dashing * 3

        # Save first try, wait only if first try is wrong
        nb_try = 1

        # While the times are the same
        while self.game_step != step_searched and self.game_step != step_searched_dash and nb_try < 11:

            # Wait a bit
            time.sleep(self.config.sleep)

            nb_try += 1

            # Get the info of Celeste
            response = requests.get("http://localhost:32270/tas/info", timeout=5)

            # Get the corresponding text
            text_row = BeautifulSoup(response.content, "html.parser").text
            l_text = text_row.split("\n")

            # Normally Timer appear on -8 index, but sometimes there is a "Cursor" info that can crash the code
            # If "Timer" is not on -8 index, we just check all the index
            if len(l_text) > 8 and "Timer" in l_text[-8]:
                # Get game step
                self.game_step = int(l_text[-8].replace(")","").split("(")[1])
            else:
                for line in l_text:
                    if "Timer" in line:
                        # Get game step
                        self.game_step = int(line.replace(")","").split("(")[1])

        if self.game_step == 0:
            fail_see_death = True
            return None, None, fail_see_death

        # Get the observation information, not gonna detail those part because it is just the string interpretation
        # Run "http://localhost:32270/tas/info" on a navigator to understand the information gotten

        # Init done at False
        done = False
        self.screen_passed = False

        for line in l_text:

            if "Pos" in line:
                # get position
                pos = line.split(' ')[-2:]
                self.pos_x = float(pos[0].replace(",",""))
                self.pos_y = float(pos[1].replace("\r", ""))

                # Normalise the information
                observation[0] = self.screen_info.normalize_x(self.pos_x)
                observation[1] = self.screen_info.normalize_y(self.pos_y)

            if "Speed" in line:
                # Only speed is get for now, maybe velocity will be usefull later
                speed = line.split()
                observation[2] = float(speed[1].replace(",", ""))/6
                observation[3] = float(speed[2])/6

            if "Stamina" in line:
                # Stamina
                observation[4] = float(line.split()[1])/110

            # By default, 0
            if "Wall-L" in line:
                observation[5] = 0.5
            elif "Wall-R" in line:
                observation[5] = 1

            if "StNormal" in line:
                observation[6] = 0
            elif "StDash" in line:
                observation[6] = 0.5
            elif "StClimb" in line:
                observation[6] = 1

            # By default 0
            if "CanDash" in line:
                observation[7] = 1

            if "Coyote" in line:
                observation[8] = int(line.split("Coyote")[1][1]) / 5 # 5 is max value of coyotte

            if "Jump" in line: # If more than 10
                if line.split("Jump")[1][2].isnumeric():
                    value = int(line.split("Jump")[1][1:3])
                else:
                    value = int(line.split("Jump")[1][1])
                observation[9] = value / 14 # 14 is max value of jump

            if "DashCD" in line:
                if line.split("DashCD")[1][2].isnumeric():
                    value = int(line.split("DashCD")[1][1:3])
                else:
                    value = int(line.split("DashCD")[1][1])
                observation[10] = value / 11 # 14 is max value of jump

            # If dead
            if "Dead" in line:
                done = True
                self.dead = True

            if "Timer" in line:
                # If screen pasted. Only on screen 1 for now, will be change later
                if f"[{self.screen_info.next_screen_id}]" in line:
                    self.screen_passed = True
                    if self.config.one_screen:
                        done = True

                    else:
                        if self.screen_info.screen_value == self.config.max_screen_value:
                            done = True

                        else:
                            self.change_next_screen()

                # Else if the current screen id is not in text, then the wrong screen as been pasted
                elif f"[{self.screen_info.screen_id}]" not in line:
                    self.wrong_screen_passed = True
                    done = True


        return observation, done, None

    def get_reward(self):
        """Get the reward

        Returns:
            int: Reward
        """
        # If dead
        if self.dead:
            return self.config.reward_death

        # If screen passed
        if self.screen_passed:
            return self.config.reward_screen_passed

        if self.wrong_screen_passed:
            return self.config.reward_wrong_screen_passed

        # Else reward is natural reward
        return self.config.natural_reward * np.square(self.screen_info.distance_goal(self.pos_x, self.pos_y) / 0.7)

    def save_video(self):
        """Save the saved video of this episode
        """

        # Write the images in the file
        all_frames = list(self.screens)
        last_frame = all_frames[-1]
        current_frame_index = 0

        # Configuration
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = self.config.fps
        size = (self.screens[last_frame].shape[1], self.screens[last_frame].shape[0])

        # Object creation VideoWriter
        out = cv2.VideoWriter("result.mp4", fourcc, fps, size)


        for frame in range(last_frame+1):
            current_frame = all_frames[current_frame_index]
            out.write(self.screens[current_frame])

            if frame > current_frame:
                current_frame_index += 1


        # Close the file
        out.release()

    def controls_before_start(self):
        """Controls before the start of the test: 
        - Make sure that the init tas file work
        - Check the screen
        """

        # Reset the environnement
        self.reset()

        # Save the image
        if self.config.use_image:
            screen = self.get_image_game(normalize=False)[0]
            cv2.imwrite('screen.png', np.rollaxis(screen, 0, 3))

            # If the image shape is not the same as the one in config
            if screen.shape[0] != self.config.size_image[0] or screen.shape[1] != self.config.size_image[1]:

                # Change i if allowed
                if self.config.allow_change_size_image:
                    self.config.size_image = np.array(screen.shape)
                    self.screen_obs = np.zeros(((self.config.histo_obs+1)*self.config.size_image[0], self.config.size_image[1], self.config.size_image[2]))

                # Else raise error
                else:
                    print("ERROR SIZE IMAGE")
