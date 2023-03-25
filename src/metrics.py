"""Metrics class file
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from config import Config


class Metrics:
    """Class for metrics
    """

    def __init__(self, config: Config) -> None:

        # Save config
        self.config = config

        # Array of shape 3 to indicate the time Madeline pass the level, died, or juste finished with max iteration
        self.info_level = {
            "Unfinished": [0],
            "Step 1": [0],
            "Step 2": [0],
            "Step 3": [0],
            "Death": [0],
            "Level passed": [0]
        }

        # list to store all the rewards gotten
        self.all_reward = list()

        # Max reward gotten on each range of val test
        self.max_mean_reward = -1 * np.inf

        # Get the current time
        self.init_time = time.time()

        # Quantity of step passed
        self.nb_total_step = 0

        # Counter for restore model
        self.counter_restore = 0

    def insert_metrics(self, reward: list(), episode: int):
        """Insert metrics given

        Args:
            reward (list): list of rewards of the episode
            episode (int): current episode

        Returns:
            bool: True if there is a new max reward, else False
        """
        # Actualise the total number of steps
        self.nb_total_step += len(reward)

        # Get the mean of the reward
        mean_reward = np.mean(reward)

        # Add the reward to all the reward
        self.all_reward.append(mean_reward)

        # Check death
        if reward[-1] == self.config.reward_death and len(reward) < self.config.max_steps:
            self.info_level["Death"][-1] += 1

        # Else check level passed
        elif reward[-1] == self.config.reward_screen_passed:
            self.info_level["Level passed"][-1] += 1

        # Else maddeline did not finished
        else:
            self.info_level["Unfinished"][-1] += 1

        # Check for step reached
        for index in range(3):
            # Check is the index is in the reward list
            if (index + 1) * self.config.reward_step_reached in reward:
                self.info_level[f"Step {index + 1}"][-1] += 1



        # Init the new max reward to False
        new_max_reward = False

        restore = False


        # Only print graph is the episode is multiple of value to print
        if episode % self.config.val_test == 0:

            # Shape rewards to simplify calculs
            reshape_rewards = np.array(self.all_reward).reshape(-1, self.config.val_test)

            # If we get a new max mean reward
            if np.mean(reshape_rewards[-1]) > self.max_mean_reward:

                # Save the new max
                self.max_mean_reward = np.mean(reshape_rewards[-1])

                # Set the value to True to save the model
                new_max_reward = True
                self.counter_restore = 0

            else:
                self.counter_restore += 1
                if self.counter_restore == self.config.limit_restore:
                    restore = True
                    self.counter_restore = 0

            # Do not print the graphs if it is the first iteration (because graphs would be empty)
            if episode != self.config.val_test:

                # Print the graphs
                self.print_result(reshape_rewards)

            # Add a new step to the info level
            for value in self.info_level.values():
                value.append(0)

        return new_max_reward, restore

    def print_step(self, episode: int):
        """Print the metrics infos at the current episode

        Args:
            episode (int): current episode
            epsilon (float): current epsilon
        """

        # Get time
        time_spend = time.strftime("%H:%M:%S", time.gmtime(np.round(time.time() - self.init_time)))

        # Get the current episode compare to the value test
        print_reward = episode % self.config.val_test if episode % self.config.val_test != 0 else self.config.val_test
        end = "\n" if episode % self.config.val_test == 0 else "\r"

        # Print the graph
        print("Time : {}, episode : {}, reward last {} ep {}, reward ep {}, max mean reward {}, nb step {}   ".format(
            time_spend,
            episode,
            print_reward,
            np.round(np.mean(self.all_reward[-print_reward:]), 2),
            np.round(self.all_reward[-1], 2),
            np.round(self.max_mean_reward, 2),
            self.nb_total_step
        ), end=end)

    def print_result(self, rewards: np.ndarray):
        """Generate a graph with the results
        """
        mean_curve = np.mean(rewards, axis=1)
        max_curve = np.max(rewards, axis=1)
        min_curve = np.min(rewards, axis=1)

        # Calculer la somme cumulée de curve mean
        global_current_mean = np.cumsum(mean_curve)

        # Calculer la moyenne cumulée de la valeur
        global_current_mean /= np.arange(1, len(mean_curve)+1)


        # Créer une figure et un axe pour le premier graphique
        _, axs = plt.subplots(1, 2, figsize = (20,10))

        # Tracer les courbes pour le premier graphique
        axs[0].plot(max_curve, label="max", color="blue")
        axs[0].plot(mean_curve, label="mean", color="green")
        axs[0].plot(min_curve, label="min", color="red")
        axs[0].plot(global_current_mean, label="Global", color="orange")

        # Ajouter les titres et labels d'axes pour le premier graphique
        axs[0].set_title("Données max, mean, et min")
        axs[0].set_xlabel("Index")
        axs[0].set_ylabel("Valeur")

        # Ajouter une légende pour le premier graphique
        axs[0].legend(loc="upper left")

        # Tracer les courbe pour le deuxième graphique
        for key, value in self.info_level.items():
            percentage_value = np.array(value) * 100 / self.config.val_test
            axs[1].plot(percentage_value, label=key, color=self.config.color_graph[key])

        # Ajouter les titres et labels d'axes pour le deuxième graphique
        axs[1].set_title("Données graphe win")
        axs[1].set_xlabel("Index")
        axs[1].set_ylabel("Valeur graphe win")

        # Ajouter une légende pour le deuxième graphique
        axs[1].legend(loc="upper left")

        # Enregistrer le graphique 2 dans un fichier png
        plt.savefig("result.png")
        plt.close()
