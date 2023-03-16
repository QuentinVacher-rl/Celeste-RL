"""Multiple DQN class file
"""

import random as r
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model
import tensorflow as tf



from config_multi_qnetworks import ConfigMultiQnetworks
from config import Config

class MultiQNetwork():
    """Class for Multiple DQN
    """
    def __init__(self, config_multi_qnetwork: ConfigMultiQnetworks, config: Config):

        # Config of Multiple Qnetwork
        self.config = config_multi_qnetwork

        # Get the right model
        self.type_model = "SIMPLE_MLP" if not config.use_image else "CNN_MLP"

        # Get the size of the image (will not be used if simple MLP)
        self.size_image = config.size_image

        # Memory of each step
        self.memory = []

        # Current learning rate
        self.cur_lr = self.config.init_lr

        # Current epsilone
        self.epsilon = self.config.init_epsilon

        # Observation and action size
        self.state_size = config.observation_size
        self.action_size = config.action_size

        # Create Qnetwork model
        self.q_network = self.build_model()

        # Create the target network by copying the Qnetwork model
        self.target_network = tf.keras.models.clone_model(self.q_network)

        # Restore the network with saved one
        if self.config.restore_networks:
            self.restore()


    def build_model(self):
        """Create the Qnetwork model

        Returns:
            Model: Qnetwork model
        """

        # If simple MLP
        if self.type_model == "SIMPLE_MLP":

            # Create layer input
            input_layer = Input(shape=(self.state_size,))

            # Create the main branch of the model
            dense1 = Dense(128, activation='relu')(input_layer)
            dense2 = Dense(64, activation='relu')(dense1)



        # If CNN model (use for image)
        elif self.type_model == "CNN_MLP":

            # Input for image
            input_img = Input(shape=(self.size_image))

            # CNN Layers
            conv1 = Conv2D(32, (3, 3), activation='relu')(input_img)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = Conv2D(16, (3, 3), activation='relu')(pool2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            conv3 = Conv2D(8, (3, 3), activation='relu')(pool2)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            flatten1 = Flatten()(pool3)

            # Input for classic infos
            input_gps = Input(shape=(self.state_size,))

            # Concat cnn layers and classic input
            merge = Concatenate()([flatten1, input_gps])

            # Create the main branch of the model
            dense1 = Dense(512, activation='relu')(merge)
            dense2 = Dense(64, activation='relu')(dense1)

            # Define the input layer with the two input layers
            input_layer = [input_img, input_gps]

        # Output layers
        outputs = list()

        # For each action
        for current_action_size in self.action_size:

            # Add MLP layer only for the proper action
            cur_dense = Dense(32, activation='relu')(dense2)

            # Output layer
            outputs.append(Dense(current_action_size, activation='sigmoid')(cur_dense))

        # Create the model
        model = Model(inputs=input_layer, outputs=outputs)

        # Compile the model
        model.compile(loss='mse', optimizer= tf.optimizers.Adam(learning_rate=self.cur_lr))


        return model

    def choose_action(self, state: np.ndarray, available_actions:list=None):
        """Choose the different actions to take

        Args:
            state (np.array): Current state of the environnement
            available_actions (list): for each actions, 1 if the action is possible, else 0, if None, all actions are available

        Returns:
            np.array: array of action taken
        """
        # Create the action array
        actions = np.zeros(self.action_size.shape, dtype=np.int32)

        # Get the action tensor
        list_action_tensor = self.q_network(state)

        # For each action
        for index, action_tensor in enumerate(list_action_tensor):

            # If random < epsilon (epsilon have a minimal value)
            if r.random() < max(self.epsilon, self.config.min_epsilon):

                # Get random action probability
                actions_probs = np.random.rand(self.action_size[index])
            else:

                # Get QDN action probability
                actions_probs = action_tensor.numpy()[0]

            # Multiply action probability with available actions to avoid unavailable actions
            if available_actions is not None:
                actions_probs = actions_probs * available_actions[index]

            # Get the action
            actions[index] = np.argmax(actions_probs)

            # If action[0] != 0, Madeline is dashing in a direction, so the directional action is desactivate
            if index == 1 and actions[0] != 0:
                actions[index] = 0

        return actions

    def insert_data(self, data: tuple):
        """Insert the data in the memory

        Args:
            data (tuple): different data put in the memory
        """
        # Insert the data
        self.memory.append(data)

        # If memory is full, delete the oldest one
        if len(self.memory) > self.config.memory_capacity:
            self.memory.pop(0)

    def train(self, episode: int):
        """Train the algorithm

        Args:
            episode (int): Current episode
        """

        # Do not train if not enough data, not the right episode or not enough episode to start training
        if len(self.memory) < self.config.batch_size or episode % self.config.nb_episode_learn != 0 or episode < self.config.start_learning:
            return

        # Get the data
        states, actions, rewards, next_states, dones = zip(*r.sample(self.memory, self.config.batch_size))

        # Those line switch the vector of shapes (because of the multiple outputs)
        states = [np.concatenate([states[i][j] for i in range(len(states))], axis=0) for j in range(len(states[0]))]
        next_states = [np.concatenate([next_states[i][j] for i in range(len(next_states))], axis=0) for j in range(len(next_states[0]))]
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Get the q values and next q values
        q_values = [tensor.numpy() for tensor in self.q_network(states)]
        next_q_values = [tensor.numpy() for tensor in self.target_network(next_states)]

        # For each q values
        for index, q_value in enumerate(q_values):

            # Apply the bellman equation of the next q value
            target_q_value = (next_q_values[index].max(axis=1) * self.config.discount_factor) + rewards

            # Only get the reward if the step has done = True
            target_q_value[dones] = rewards[dones]

            # Apply the target value on the q value
            q_value[np.arange(self.config.batch_size), actions[:, index]] = target_q_value

        # Train the model
        self.q_network.fit(states, q_values, epochs=3, verbose=0, batch_size=self.config.mini_batch_size)

        # Copy the target newtork if the episode is multiple of the coefficient
        if episode % self.config.nb_episode_copy_target == 0:
            self.copy_target_network()

        # Decay epsilon
        self.epsilon_decay()

        # Decay learning rate
        self.lr_decay(episode)


    def copy_target_network(self):
        """Set the weight on the target network based on the current q network
        """
        self.target_network.set_weights(self.q_network.get_weights())

    def epsilon_decay(self):
        """Decay espilon
        """
        self.epsilon = self.config.epsilon_decay*self.epsilon

    def lr_decay(self, cur_episode: int):
        """Decay the learning rate

        Args:
            cur_episode (int): current episode
        """
        if cur_episode % self.config.nb_episode_lr_decay == 0:
            self.cur_lr *= self.config.lr_decay
            tf.keras.backend.set_value(self.q_network.optimizer.learning_rate, self.cur_lr)



    def restore(self):
        """Restore the networks based on saved ones
        """
        self.q_network = tf.keras.models.load_model("network/network")
        self.q_network.compile(loss='mse', optimizer= tf.optimizers.Adam(learning_rate=self.cur_lr))
        self.copy_target_network()



    def save(self):
        """Save the networks
        """
        self.q_network.save("network/network")
