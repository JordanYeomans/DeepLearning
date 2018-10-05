########################################################################
# This file is based on the TensorFlow Tutorials available at:
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
# Published under the MIT License. See the file LICENSE for details.
# Copyright 2017 by Magnus Erik Hvass Pedersen
########################################################################

import numpy as np

class ReplayMemory:
    """
    The replay-memory holds many previous states of the game-environment.
    This helps stabilize training of the Neural Network because the data
    is more diverse when sampled over thousands of different states.
    """

    def __init__(self, size, state_shape, num_actions, discount_factor=0.97):
        """

        :param size:
            Capacity of the replay-memory. This is the number of states.

        :param num_actions:
            Number of possible actions in the game-environment.

        :param discount_factor:
            Discount-factor used for updating Q-values.
        """

        # Array for the previous states of the game-environment.
        self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

        # Array for the Q-values corresponding to the states.
        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Array for the Q-values before being updated.
        # This is used to compare the Q-values before and after the update.
        self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Actions taken for each of the states in the memory.
        self.actions = np.zeros(shape=size, dtype=np.int)

        # Rewards observed for each of the states in the memory.
        self.rewards = np.zeros(shape=size, dtype=np.float)

        # Whether the life had ended in each state of the game-environment.
        self.end_life = np.zeros(shape=size, dtype=np.bool)

        # Whether the episode had ended (aka. game over) in each state.
        self.end_episode = np.zeros(shape=size, dtype=np.bool)

        # Estimation errors for the Q-values. This is used to balance
        # the sampling of batches for training the Neural Network,
        # so we get a balanced combination of states with high and low
        # estimation errors for their Q-values.
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

        # Capacity of the replay-memory as the number of states.
        self.size = size

        # Discount-factor for calculating Q-values.
        self.discount_factor = discount_factor

        # Reset the number of used states in the replay-memory.
        self.num_used = 0

        # Threshold for splitting between low and high estimation errors.
        self.error_threshold = 0.1

    def is_full(self):
        """Return boolean whether the replay-memory is full."""
        return self.num_used == self.size

    def used_fraction(self):
        """Return the fraction of the replay-memory that is used."""
        return self.num_used / self.size

    def reset(self):
        """Reset the replay-memory so it is empty."""
        self.num_used = 0

    def add(self, state, q_values, action, reward, end_life, end_episode):
        """
        Add an observed state from the game-environment, along with the
        estimated Q-values, action taken, observed reward, etc.

        :param state:
            Current state of the game-environment.
            This is the output of the MotionTracer-class.

        :param q_values:
            The estimated Q-values for the state.

        :param action:
            The action taken by the agent in this state of the game.

        :param reward:
            The reward that was observed from taking this action
            and moving to the next state.

        :param end_life:
            Boolean whether the agent has lost a life in this state.

        :param end_episode:
            Boolean whether the agent has lost all lives aka. game over
            aka. end of episode.
        """

        if not self.is_full():
            # Index into the arrays for convenience.
            k = self.num_used

            # Increase the number of used elements in the replay-memory.
            self.num_used += 1

            # Store all the values in the replay-memory.
            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action
            self.end_life[k] = end_life
            self.end_episode[k] = end_episode

            # Note that the reward is limited. This is done to stabilize
            # the training of the Neural Network.
            self.rewards[k] = np.clip(reward, -1, 1.0)

    def update_all_q_values(self):
        """
        Update all Q-values in the replay-memory.

        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we now know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """

        # Copy old Q-values so we can print their statistics later.
        # Note that the contents of the arrays are copied.
        self.q_values_old[:] = self.q_values[:]

        # Process the replay-memory backwards and update the Q-values.
        # This loop could be implemented entirely in NumPy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        for k in reversed(range(self.num_used - 1)):
            # Get the data for the k'th state in the replay-memory.
            action = self.actions[k]
            reward = self.rewards[k]
            end_life = self.end_life[k]
            end_episode = self.end_episode[k]

            # Calculate the Q-value for the action that was taken in this state.
            if end_life or end_episode:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                action_value = reward
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.

                ## Todo: Maybe we can take the maximum of the (max Q-Value, Next action reward)
                ## This might incentivise the agent to learn 1 off strategies that
                ## lead to high rewards
                action_value = reward + self.discount_factor * np.max(self.q_values[k + 1])

            # Error of the Q-value that was estimated using the Neural Network.
            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

            # Update the Q-value with the better estimate.
            self.q_values[k, action] = action_value

        self.print_statistics()

    def prepare_sampling_prob(self, batch_size=128):
        """
        Prepare the probability distribution for random sampling of states
        and Q-values for use in training of the Neural Network.

        The probability distribution is just a simple binary split of the
        replay-memory based on the estimation errors of the Q-values.
        The idea is to create a batch of samples that are balanced somewhat
        evenly between Q-values that the Neural Network already knows how to
        estimate quite well because they have low estimation errors, and
        Q-values that are poorly estimated by the Neural Network because
        they have high estimation errors.

        The reason for this balancing of Q-values with high and low estimation
        errors, is that if we train the Neural Network mostly on data with
        high estimation errors, then it will tend to forget what it already
        knows and hence become over-fit so the training becomes unstable.
        """

        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimation_errors[0:self.num_used]

        # Create an index of the estimation errors that are low.
        idx = err < self.error_threshold

        self.idx_err_lo = np.squeeze(np.where(idx))

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.idx_err_hi) / self.num_used
        prob_err_hi = max(prob_err_hi, 0.5)

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
        """
        Get a random batch of states and Q-values from the replay-memory.
        You must call prepare_sampling_prob() before calling this function,
        which also sets the batch-size.

        The batch has been balanced so it contains states and Q-values
        that have both high and low estimation errors for the Q-values.
        This is done to both speed up and stabilize training of the
        Neural Network.
        """

        # Random index of states and Q-values in the replay-memory.
        # These have LOW estimation errors for the Q-values.
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)

        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch

    def all_batches(self, batch_size=128):
        """
        Iterator for all the states and Q-values in the replay-memory.
        It returns the indices for the beginning and end, as well as
        a progress-counter between 0.0 and 1.0.

        This function is not currently being used except by the function
        estimate_all_q_values() below. These two functions are merely
        included to make it easier for you to experiment with the code
        by showing you an easy and efficient way to loop over all the
        data in the replay-memory.
        """

        # Start index for the current batch.
        begin = 0

        # Repeat until all batches have been processed.
        while begin < self.num_used:
            # End index for the current batch.
            end = begin + batch_size

            # Ensure the batch does not exceed the used replay-memory.
            if end > self.num_used:
                end = self.num_used

            # Progress counter.
            progress = end / self.num_used

            # Yield the batch indices and completion-counter.
            yield begin, end, progress

            # Set the start-index for the next batch to the end of this batch.
            begin = end

    def estimate_all_q_values(self, model):
        """
        Estimate all Q-values for the states in the replay-memory
        using the model / Neural Network.

        Note that this function is not currently being used. It is provided
        to make it easier for you to experiment with this code, by showing
        you an efficient way to iterate over all the states and Q-values.

        :param model:
            Instance of the NeuralNetwork-class.
        """

        print("Re-calculating all Q-values in replay memory ...")

        # Process the entire replay-memory in batches.
        for begin, end, progress in self.all_batches():
            # Print progress.
            msg = "\tProgress: {0:.0%}"
            msg = msg.format(progress)
            import Logging
            Logging.print_progress(msg)

            # Get the states for the current batch.
            states = self.states[begin:end]

            # Calculate the Q-values using the Neural Network
            # and update the replay-memory.
            self.q_values[begin:end] = model.get_q_values(states=states)

        # Newline.
        print()

    def save_numpy_arrays(self, data_path):
        np.save(data_path + 'states.npy', self.states)
        np.save(data_path + 'q_values.npy', self.q_values)
        np.save(data_path + 'actions.npy', self.actions)
        np.save(data_path + 'end_life.npy', self.end_life)
        np.save(data_path + 'end_episode.npy', self.end_episode)
        np.save(data_path + 'rewards.npy', self.rewards)
        np.save(data_path + 'estimation_errors.npy', self.estimation_errors)

    def load_numpy_arrays(self, data_path):
        self.states = np.load(data_path + 'states.npy')
        self.q_values = np.load(data_path + 'q_values.npy')
        self.actions = np.load(data_path + 'actions.npy')
        self.end_life = np.load(data_path + 'end_life.npy')
        self.end_episode = np.load(data_path + 'end_episode.npy')
        self.rewards = np.load(data_path + 'rewards.npy')
        self.estimation_errors = np.load(data_path + 'estimation_errors.npy')

    def print_statistics(self):
        """Print statistics for the contents of the replay-memory."""

        print("Replay-memory statistics:")

        # Print statistics for the Q-values before they were updated
        # in update_all_q_values().
        msg = "\tQ-values Before, Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values_old),
                         np.mean(self.q_values_old),
                         np.max(self.q_values_old)))

        # Print statistics for the Q-values after they were updated
        # in update_all_q_values().
        msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values),
                         np.mean(self.q_values),
                         np.max(self.q_values)))

        # Print statistics for the difference in Q-values before and
        # after the update in update_all_q_values().
        q_dif = self.q_values - self.q_values_old
        msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(q_dif),
                         np.mean(q_dif),
                         np.max(q_dif)))

        # Print statistics for the number of large estimation errors.
        # Don't use the estimation error for the last state in the memory,
        # because its Q-values have not been updated.
        err = self.estimation_errors[:-1]
        err_count = np.count_nonzero(err > self.error_threshold)
        msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
        print(msg.format(self.error_threshold, err_count,
                         self.num_used, err_count / self.num_used))

        # How much of the replay-memory is used by states with end_life.
        end_life_pct = np.count_nonzero(self.end_life) / self.num_used

        # How much of the replay-memory is used by states with end_episode.
        end_episode_pct = np.count_nonzero(self.end_episode) / self.num_used

        # How much of the replay-memory is used by states with non-zero reward.
        reward_nonzero_pct = np.count_nonzero(self.rewards) / self.num_used

        # Print those statistics.
        msg = "\tend_life: {0:.1%}, end_episode: {1:.1%}, reward non-zero: {2:.1%}"
        print(msg.format(end_life_pct, end_episode_pct, reward_nonzero_pct))


########################################################################
