import numpy as np
import matplotlib.pyplot as plt
import time

import Reinforcement_Learning.State as State

class Monty:

    def __init__(self):

        # Parameters that can be tuned
        self.reward_for_winning = 1
        self.reward_for_idle = -0.1
        self.reward_for_fail = -1
        self.max_time_without_reward = 200
        self.training_array_steps = 200
        self.min_epsilon = 0.03

        # Starts Here
        self.pos_x = 0
        self.pos_y = 0

        self.reward = 0
        self.next_epsilon = None

        self.end_due_to_reward = False
        self.end_due_to_time = False
        self.time_without_reward = 0

        # Global States
        global_state_height = 210
        global_state_width = 160
        global_state_channels = 3

        self.global_state_img_size = np.array([global_state_height, global_state_width])
        self.global_state_shape = [global_state_height, global_state_width, global_state_channels] #Used to Define Neural Network

        # NN States
        self.nn_state_height = 66
        self.nn_state_width = 66
        self.nn_state_channels = 3

        self.nn_state_img_size = np.array([self.nn_state_height, self.nn_state_width])
        self.nn_state_shape = [self.nn_state_height, self.nn_state_width, self.nn_state_channels]

        # Paths
        self._update_global_path()
        self.remaining_path = self.global_path
        self.remaining_path_img = self.global_path_img

    def reset(self, img, num_episodes):

        self.__init__()
        self.GlobalState = State.GlobalState(img, self.global_state_img_size)
        self._update_monty_pos(img) #Essential for the Local State Calculation
        self._update_nn_state(img)
        self.num_episodes = num_episodes

    def update(self, img, episode_failed):
        self.end_due_to_episode = episode_failed
        self._update_monty_pos(img)
        self._update_reward(episode_failed)
        self._update_nn_state(img)
        self._update_reward_timer()
        self._update_training_array()
        self._update_epsilon()

    def view_states(self):
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(self.global_state[:, :, 0])

        plt.subplot(2, 3, 2)
        plt.imshow(self.global_state[:, :, 1])

        plt.subplot(2, 3, 3)
        plt.imshow(self.global_state[:, :, 2])

        plt.subplot(2, 3, 4)
        plt.imshow(self.nn_state[:, :, 0])

        plt.subplot(2, 3, 5)
        plt.imshow(self.nn_state[:, :, 1])

        plt.subplot(2, 3, 6)
        plt.imshow(self.nn_state[:, :, 2])
        plt.show()

    def get_nn_state(self):
        return self.nn_state

    def get_nn_state_shape(self):
        return self.nn_state_shape

    def get_epsilon(self):
        distance = self.global_path.shape[0] - self.remaining_path.shape[0]
        return self.epsilon_array[distance]

    def get_reward(self):
        return self.reward

    def create_training_array(self):

        # Training Path = [Timesteps, # of Points, (Counts, Epsilon)]
        self.training_array = np.zeros((self.training_array_steps, self.global_path.shape[0]))
        self.epsilon_array = np.ones(self.global_path.shape[0])

        return self.training_array, self.epsilon_array

    def get_training_array(self):
        return self.training_array, self.epsilon_array

    def send_training_array(self, training_array, epsilon_array):
        self.training_array = training_array
        self.epsilon_array = epsilon_array

    ## Internal Functions
    def _update_monty_pos(self, observation):
        ''' Updates the X and Y Position of Monty in the observation space

        :param observation:
        :return:
        '''

        # TODO: Check this works for all levels
        # Filter pixels to only red pixels
        red_channel = observation[:, :, 0]

        # Find Pixels Monty
        monty_idx = np.where(np.logical_or(np.equal(red_channel, 200),
                                            np.equal(red_channel, 210)))

        # Create numpy array containing only Monty pixels
        monty = np.zeros_like(red_channel)
        monty[monty_idx] = 1
        monty[:25] = 0 # Top 25 pixels are the score

        # Monty X, Y is the mean of the highest and lowest monty pixel values
        self.pos_x = int(np.mean([np.max(np.where(monty == 1)[1]), np.min(np.where(monty == 1)[1])]))
        self.pos_y = int(np.mean([np.max(np.where(monty == 1)[0]), np.min(np.where(monty == 1)[0])]))

    def _update_nn_state(self, img):
        self.global_state = self.GlobalState.get_state(img, self.remaining_path_img)

        self.global_state = np.array(self.global_state)

        # Pad Global State to make sure we don't run off the end when we get the local state
        height = int(self.nn_state_height / 2)
        width = int(self.nn_state_width/2)

        pad_width = [(height, height), (width, width), (0, 0)]
        constant_values = [(255, 255), (255, 255), (0, 0)]

        self.global_state_pad = np.pad(self.global_state, pad_width, mode='constant', constant_values=constant_values)

        y_start = self.pos_y - int(self.nn_state_height / 2) + width
        x_start = self.pos_x - int(self.nn_state_width / 2) + height
        y_end = y_start + self.nn_state_height
        x_end = x_start+self.nn_state_width

        self.nn_state = self.global_state_pad[y_start:y_end, x_start:x_end]

    def _update_epsilon(self):
        self.next_epsilon = None

    def _update_reward(self, episode_failed):

        if episode_failed:
            self.reward = self.reward_for_fail
        else:
            self._calc_reward()

    def _update_global_path(self, selection=None):

        """ Function to Control High level control of path The Next algorithm will be used to control this bit

            :param observation:
            :return:
        """

        points_1 = [[80, 85, 0],
                    [80, 125, 0],
                    [112, 125, 0],
                    [112, 104, 0],
                    [135, 125, 0],
                    [136, 125, 0],
                    [136, 170, 0],
                    [23, 170, 0],
                    [23, 125, 0],
                    [11, 110, 0]]

        all_points = [points_1]

        # List of points for the path [Reward, X, Y]
        self.global_path = create_path_from_points(all_points)

        # Create image containing the global path, this is what goes into the state
        self.global_path_img = create_path_img_from_list(self.global_state_img_size, self.global_path)

    def _calc_reward(self):

        overlap_idx = None
        found_point = None
        x = 0

        min_dist_to_end = 3
        # Check y +- 2 to account for errors in manual placement of path
        # Todo: Make path a 'blured line'
        for y in range(-2, 3):

            # Find any points within remaining path array that monty is currently overlapping
            overlap_idx = np.where(np.logical_and(self.pos_x == self.remaining_path[:, 1] + x,
                                                  self.pos_y == self.remaining_path[:, 2] + y))[0]

            if overlap_idx.size != 0:
                found_point = 1
                # Take first index to make sure we don't reward cutting ahead
                overlap_idx = overlap_idx[0]
                break

        # Check if we have won
        if len(self.remaining_path) - overlap_idx <= min_dist_to_end:
            self.end_due_to_reward = True
            self.reward = self.reward_for_winning

        # If we haven't won, but we have overlapped with the path
        elif  overlap_idx.size != 0:
            self.reward = self.remaining_path[overlap_idx][0]
            self.remaining_path = self.remaining_path[overlap_idx + 1:]
            # self.training_path = self.training_path[overlap_idx + 1:]

        else:
            self.reward = self.reward_for_idle

        self.remaining_path_img = create_path_img_from_list(self.global_state_img_size, self.remaining_path)

    def _update_reward_timer(self):
        if self.reward > 0:
            self.time_without_reward = 0
        else:
            self.time_without_reward += 1

        if self.time_without_reward > self.max_time_without_reward:
            self.end_due_to_time = True

    def _update_training_array(self):

        if self.end_due_to_time \
                or self.end_due_to_episode \
                or self.end_due_to_reward:

            distance = self.global_path.shape[0] - self.remaining_path.shape[0]

            # If we haven't filled up the prob array yet
            if self.num_episodes < self.training_array_steps:
                self.training_array[self.num_episodes][:distance] = 1

            else:

                self.training_array = np.roll(self.training_array, 1, axis=0)
                self.training_array[0] = 0
                self.training_array[0][:distance] += 1

                # print('Training Array Shape = {}'.format(self.training_array.shape))
                # print('Training Array First Val = {}'.format(self.training_array[0]))

                prob_array = np.sum(self.training_array, axis=0)

                epsilon_array = prob_array
                epsilon_array = np.divide(epsilon_array, self.training_array_steps)
                epsilon_array = np.add(epsilon_array, -1)
                epsilon_array = np.absolute(epsilon_array)
                # epsilon_array = np.add(epsilon_array, self.min_epsilon)
                self.epsilon_array = np.clip(epsilon_array, self.min_epsilon, 1)

############# Temporary Functions
def create_path_from_points(all_points):

    selection = None

    if selection is None:
        rand_point_selection = np.random.randint(0, len(all_points))
    else:
        rand_point_selection = selection

    list_of_points = all_points[rand_point_selection]

    current_x = list_of_points[0][0]
    current_y = list_of_points[0][1]

    target_path = np.zeros((1, 2))
    training_path = np.zeros((1, 1))

    target_path[0][0] = current_x
    target_path[0][1] = current_y

    for x_y_points in list_of_points[1:]:
        target_x = x_y_points[0]
        target_y = x_y_points[1]
        training = x_y_points[2]

        longest_dist = np.maximum(np.absolute(current_x - target_x),
                                  np.absolute(current_y - target_y))

        x_path = np.array(np.linspace(current_x, target_x, longest_dist).astype(np.int)).reshape(-1, 1)
        y_path = np.array(np.linspace(current_y, target_y, longest_dist).astype(np.int)).reshape(-1, 1)

        target_path_new = np.concatenate([x_path, y_path], axis=1)
        training_path_new = np.ones((target_path_new.shape[0], 1)) * training

        target_path = np.concatenate([target_path, target_path_new], axis=0)
        training_path = np.concatenate([training_path, training_path_new], axis=0)

        current_x = target_x
        current_y = target_y

    reward_function = np.ones((target_path.shape[0], 1))

    return np.concatenate([reward_function, target_path], axis=1)


def create_path_img_from_list(state_img_size, path):
    path_img = np.zeros((state_img_size[0], state_img_size[1]))

    for i in range(len(path)):
        x = int(path[i][1])
        y = int(path[i][2])

        path_img[y][x] = 255

    return path_img