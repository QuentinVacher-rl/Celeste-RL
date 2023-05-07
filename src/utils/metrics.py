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
            "Death": [0],
            "Level passed": [0]
        }

        # Best reward ever getten on test
        self.best_scren = 0
        self.best_run = 0


        # list to store all the testing rewards gotten
        self.all_reward = list()

        # List to store all the training rewards gotten
        self.train_reward = list()

        # Max reward gotten on each range of val test
        self.max_mean_reward = -1 * np.inf

        # Get the current time
        self.init_time = time.time()

        # Quantity of step passed
        self.nb_total_test_step = 0
        self.nb_total_train_step = 0

        # Counter for restore model
        self.counter_restore = 0

        # Number terminated for train part
        self.nb_terminated_train = 0

    def insert_metrics(self, learning_step: int, reward: list(), episode: int, max_steps_ep: int, last_step):
        """Insert metrics given

        Args:
            learning_step (int): step of the learning
            reward (list): list of rewards of the episode
            episode (int): current episode
            max_steps_ep (int): Max step of this episode (can variate with multiple screens)

        Returns:
            bool: True if there is a new max reward, else False
        """

        # Actualise the total number of test steps
        self.nb_total_test_step += len(reward)

        # Get the mean of the reward
        mean_reward = np.sum(reward)

        # Add the reward to all the reward
        self.all_reward.append(mean_reward)


        # Check death
        if reward[-1] == self.config.reward_death and len(reward) < max_steps_ep:
            self.info_level["Death"][-1] += 1

        # Else check level passed
        elif reward[-1] == self.config.reward_screen_passed:
            self.info_level["Level passed"][-1] += 1

        # Else maddeline did not finished
        else:
            self.info_level["Unfinished"][-1] += 1




        # Init the new max reward and new best to False
        new_max_reward = False
        new_best_reward = False

        restore = False


        nb_screen_passed = 0
        reward_unique, count = np.unique(reward, return_counts=True)
        if self.config.reward_screen_passed in reward_unique:
            nb_screen_passed = count[np.where(reward_unique == self.config.reward_screen_passed)][0]

        if nb_screen_passed > self.best_scren:
            self.best_scren = nb_screen_passed
            self.best_run = last_step
            new_best_reward = True

        elif nb_screen_passed == self.best_scren:
            if self.best_run > last_step:
                self.best_run = last_step
                new_best_reward = True


        # Only print graph is the episode is multiple of value to print
        if episode % self.config.nb_test_episode == 0:

            # Shape rewards to simplify calculs
            reshape_rewards = np.array(self.all_reward).reshape(-1, self.config.nb_test_episode)

            # If we get a new max mean reward
            if np.mean(reshape_rewards[-1]) >= self.max_mean_reward:

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
            if learning_step > 1:

                # Print the graphs
                self.print_result(reshape_rewards)

            # Add a new step to the info level
            for value in self.info_level.values():
                value.append(0)

        return new_max_reward, new_best_reward, restore

    def print_train_step(self, step, episode: int, reward: list):
        """Print the metrics infos at the current learning step

        Args:
            step (int): step of the learning
            episode (int): current episode
            reward (list): list of reward of episode
        """
        if episode == 0:
            self.train_reward.clear()

        if reward[-1] == self.config.reward_screen_passed:
            self.nb_terminated_train += 1

        # Actualise the total number of train steps
        self.nb_total_train_step += len(reward)

        # Get time
        time_spend = time.strftime("%H:%M:%S", time.gmtime(np.round(time.time() - self.init_time)))

        # Print the graph
        print("Time : {} --- Learning step : {}/{} --- episode {}/{} --- nb finished {}/{} --- Total train step {}  ".format(
            time_spend,
            step, self.config.nb_learning_step,
            episode, self.config.nb_train_episode,
            self.nb_terminated_train, episode,
            self.nb_total_train_step
        ), end="\r")



    def print_test_step(self, step, episode: int):
        """Print the metrics infos at the current learning step

        Args:
            step (int): step of the learning
            episode (int): current episode
        """

        # Get time
        time_spend = time.strftime("%H:%M:%S", time.gmtime(np.round(time.time() - self.init_time)))

        end = "\n" if episode % self.config.nb_test_episode == 0 else "\r"

        # Print the graph
        print("Time : {} --- Learning step : {}/{} --- episode {}/{}, mean reward {} --- reward ep {} --- max mean reward {} --- quickest {} --- Nb train win {}   ".format(
            time_spend,
            step, self.config.nb_learning_step,
            episode, self.config.nb_test_episode,
            np.round(np.mean(self.all_reward[-episode:]), 2),
            np.round(self.all_reward[-1], 2),
            np.round(self.max_mean_reward, 2),
            self.best_run,
            self.nb_terminated_train
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
            percentage_value = np.array(value) * 100 / self.config.nb_test_episode
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
