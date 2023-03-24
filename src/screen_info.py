"""File for the class of the screen info
"""

class ScreenInfo:
    """Class of screen info
    """

    def __init__(self, screen_id: int, first_frame: int, tas_file: str,
                 x_max: float, x_min: float, y_max :float, y_min: float,
                 list_step_reward: list, goal: list):

        # Id of the screen
        self.screen_id = screen_id

        # First frame to start
        self.first_frame = first_frame

        # Tas file to init screen
        self.init_tas_file = tas_file.format(screen_id)

        # X and Y max and min
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min

        # Step reward to reached
        self.list_step_reward = list_step_reward

        # Goal to reached
        self.goal = goal

    def normalize_x(self, value: float):
        """Normalize the value on x
        """
        return (value - self.x_min) / (self.x_max - self.x_min)

    def normalize_y(self, value: float):
        """Normalize the value on y
        """
        return (value - self.y_min) / (self.y_max - self.y_min)
