"""File for the class of the screen info
"""

import random as r
import numpy as np

class ScreenInfo:
    """Class of screen info
    """

    def __init__(self, screen_id: str, screen_value: int, start_position: list, first_frame: int, tas_file: str,
                 x_max: float, x_min: float, y_max :float, y_min: float,
                 goal: list, next_screen_id: str):

        # Id of the screen
        self.screen_id = screen_id

        # Value of the screen
        self.screen_value = screen_value

        # First frame to start
        self.first_frame = first_frame

        # Tas file to init screen
        self.init_tas_file = tas_file

        # X and Y max and min
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min

        self.max_distance = np.sqrt(np.square(self.x_max - self.x_min) + np.square(self.x_max - self.x_min))

        # Position where madeline can start
        self.start_position = start_position

        # Goal to reached
        self.goal = goal

        # Id of the next screen (for if multiple possible next screen)
        self.next_screen_id = next_screen_id

    def normalize_x(self, value: float):
        """Normalize the value on x
        """
        return (value - self.x_min) / (self.x_max - self.x_min)

    def normalize_y(self, value: float):
        """Normalize the value on y
        """
        return (value - self.y_min) / (self.y_max - self.y_min)

    def get_random_start(self):
        """Get random start
        """
        coords = r.sample(self.start_position, 1)[0]
        return self.init_tas_file.format(str(coords[0]) + " " + str(coords[1]))

    def get_true_start(self):
        """Get the true start
        """
        coords = self.start_position[0]
        return self.init_tas_file.format(str(coords[0]) + " " + str(coords[1]))
    
    def distance_goal(self, pos_x, pos_y):
        """Calcul the distance with the coordonates from the goal
        """
        
        goal_x = np.mean(self.goal[0])
        goal_y = np.mean(self.goal[1])
        distance = np.sqrt(np.square(goal_x - pos_x) + np.square(goal_y - pos_y))
        norm_distance = distance / self.max_distance
        return norm_distance

