"""Main program file
"""

import absl.logging
import torch

from celeste_env import CelesteEnv
from config import Config

import rl_sac

from utils.metrics import Metrics

absl.logging.set_verbosity(absl.logging.ERROR)

def main():
    """Main program
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.empty_cache()

    # Create the instance of the general configuration and algorithm configuration
    config = Config()
    config_algo = rl_sac.ConfigAlgo()

    # Create the environnement
    env = CelesteEnv(config)

    # Create the RL algorithm
    algo = rl_sac.Algo(config_algo, config)

    # Create the metrics instance
    metrics = Metrics(config)

    env.set_action_mode(algo.action_mode)

    # For every episode
    for learning_step in range(1, config.nb_learning_step + 1):

        env.controls_before_start()

        # Reset nb terminated
        metrics.nb_terminated_train = 0

        if learning_step > 1 or not config.start_with_test:
            for episode_train in range(1, config.nb_train_episode + 1):

                # Reset the environnement
                state, image, terminated, truncated = env.reset()

                ep_reward = list()

                # For each step
                while not terminated and not truncated:

                    # Get the actions
                    actions = algo.choose_action(state, image)
                    #print(actions)
                    # Step the environnement
                    next_state, next_image, reward, terminated, truncated, info = env.step(actions)

                    if not info["fail_death"]:
                        # Insert the data in the algorithm memory
                        algo.insert_data(state, next_state, image, next_image, actions, reward, terminated, truncated)

                        # Actualise state
                        state = next_state
                        image = next_image

                        # Train the algorithm
                        algo.train()

                        ep_reward.append(reward)
                    else:
                        truncated=True

                metrics.print_train_step(learning_step, episode_train, ep_reward)

        for episode_test in range(1, config.nb_test_episode + 1):

            fail_death = False

            # Init the episode reward at 0
            reward_ep = list()

            # Reset the environnement
            state, image, terminated, truncated = env.reset(test=True)

            # For each step
            while not terminated and not truncated:

                # Get the actions
                actions = algo.choose_action(state, image)

                # Step the environnement
                next_state, next_image, reward, terminated, truncated, info = env.step(actions)

                if info["fail_death"]:
                    fail_death = True
                    truncated = True

                # Actualise state
                state = next_state
                image = next_image

                # Add the reward to the episode reward
                reward_ep.append(reward)

            if not fail_death:
                # Insert the metrics
                save_model, save_video, restore = metrics.insert_metrics(learning_step, reward_ep, episode_test, env.max_steps, env.game_step)

                # Print the information about the episode
                metrics.print_test_step(learning_step, episode_test)


                if save_video:
                    print("save")
                    env.save_video()
            else:
                episode_test -= 1

        # Save the model (will be True only if new max reward)
        if save_model:
            algo.save_model()

        if restore:
            algo.load_model()





if __name__ == "__main__":
    main()
