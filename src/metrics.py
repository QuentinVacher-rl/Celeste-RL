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

        # Array of shape 3 to indicate the time Madeline pass the level, died, or juste finished with max iteration
        self.level_passed = np.zeros(3)

        # list to store all the rewards gotten
        self.all_reward = list()

        # calculated data for the graohs
        self.data_for_print = list()

        # Index to print the graphs
        self.val_test = config.val_test

        # Max reward gotten on each range of val test
        self.max_mean_reward = 0

        # Get the current time
        self.init_time = time.time()

    def done_with_level_finished(self):
        """Increase the number of level finished
        """
        self.level_passed[0] += 1

    def done_with_death(self):
        """Increase the number of death
        """
        self.level_passed[2] += 1

    def done_with_time(self):
        """Increase the number of episodes ended with time
        """
        self.level_passed[1] += 1


    def insert_metrics(self, reward: float, episode: int, nb_step_done: int):
        """Insert metrics given

        Args:
            reward (float): mean reward of the episode
            episode (int): current episode
            nb_step_done (int): quantity of steps done

        Returns:
            bool: True if there is a new max reward, else False
        """
        # Get the mean of the reward
        reward /= nb_step_done

        # Add the reward to all the reward
        self.all_reward.append(reward)

        # Init the new max reward to False
        new_max_reward = False

        # Only print graph is the episode is multiple of value to print
        if episode % self.val_test == 0:

            # Calcul the purcentage of the array
            self.level_passed = self.level_passed * 100 / self.val_test

            # Calul the different metrics # TODO modify metrics calculation
            self.data_for_print.append([
                np.max(self.all_reward[-self.val_test:]),
                np.mean(self.all_reward[-self.val_test:]),
                np.min(self.all_reward[-self.val_test:]),
                self.level_passed
            ])

            # If we get a new max mean reward
            if np.mean(self.all_reward[-self.val_test:]) > self.max_mean_reward:

                # Save the new max
                self.max_mean_reward = np.mean(self.all_reward[-self.val_test:])

                # Set the value to True to save the model
                new_max_reward = True

            # Do not print the graphs if it is the first iteration (because graphs would be empty)
            if episode != self.val_test:

                # Print the graphs
                self.print_result()

            # Reset the level passed
            self.level_passed = np.zeros(3)

        return new_max_reward

    def print_step(self, episode: int, epsilon: float):
        """Print the metrics infos at the current episode

        Args:
            episode (int): current episode
            epsilon (float): current epsilon
        """

        # Get time
        time_spend = time.strftime("%H:%M:%S", time.gmtime(np.round(time.time() - self.init_time)))

        # Get the current episode compare to the value test
        print_reward = episode % self.val_test if episode % self.val_test != 0 else self.val_test
        end = "\n" if episode % self.val_test == 0 else "\r"

        # Print the graph
        print("Time : {}, episode : {}, reward last {} ep {}, reward ep {}, max mean reward {}, epsilon {}   ".format(
            time_spend,
            episode,
            print_reward,
            np.round(np.mean(self.all_reward[-print_reward:]), 2),
            np.round(self.all_reward[-1], 2), np.round(self.max_mean_reward, 2),
            np.round(epsilon,3)
        ), end=end)

    def print_result(self):
        """Generate a graph with the results
        """
        # TODO change all this
        data_max = np.zeros(len(self.data_for_print))
        data_mean = np.zeros(len(self.data_for_print))
        data_min = np.zeros(len(self.data_for_print))
        global_current_mean = np.zeros(len(self.data_for_print))


        data_win = np.zeros(len(self.data_for_print))
        data_loose = np.zeros(len(self.data_for_print))
        data_not_finished = np.zeros(len(self.data_for_print))
        for index, data in enumerate(self.data_for_print):
            data_max[index] = data[0]
            data_mean[index] = data[1]
            data_min[index] = data[2]
            data_win[index] = data[3][0]
            data_not_finished[index] = data[3][1]
            data_loose[index] = data[3][2]

        # Calculer la somme cumulée de data mean
        global_current_mean = np.cumsum(data_mean)

        # Calculer la moyenne cumulée de a
        global_current_mean /= np.arange(1, len(data_mean)+1)


        # Créer une figure et un axe pour le premier graphique
        _, axs = plt.subplots(1, 2, figsize = (20,10))

        # Tracer les courbes pour le premier graphique
        axs[0].plot(data_max, label="max", color="blue")
        axs[0].plot(data_mean, label="mean", color="green")
        axs[0].plot(data_min, label="min", color="red")
        axs[0].plot(global_current_mean, label="Global", color="orange")

        # Ajouter les titres et labels d'axes pour le premier graphique
        axs[0].set_title("Données max, mean, et min")
        axs[0].set_xlabel("Index")
        axs[0].set_ylabel("Valeur")

        # Ajouter une légende pour le premier graphique
        axs[0].legend(loc="upper left")

        # Tracer la courbe pour le deuxième graphique
        axs[1].plot(data_not_finished, label="not finish", color="orange")
        axs[1].plot(data_loose, label="loos", color="red")
        axs[1].plot(data_win, label="Win", color="green")

        # Ajouter les titres et labels d'axes pour le deuxième graphique
        axs[1].set_title("Données graphe win")
        axs[1].set_xlabel("Index")
        axs[1].set_ylabel("Valeur graphe win")

        # Ajouter une légende pour le deuxième graphique
        axs[1].legend(loc="upper left")

        # Enregistrer le graphique 2 dans un fichier png
        plt.savefig("result.png")
        plt.close()
