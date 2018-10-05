########################################################################
# This file is based on the TensorFlow Tutorials available at:
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
# Published under the MIT License. See the file LICENSE for details.
# Copyright 2017 by Magnus Erik Hvass Pedersen
########################################################################

import numpy as np
import sys
import os
import csv

# Default base-directory for the checkpoints and log-files.
# The environment-name will be appended to this.

# Combination of base-dir and environment-name.
checkpoint_dir = None

# Full path for the log-file for rewards.
log_reward_path = None

# Full path for the log-file for Q-values.
log_q_values_path = None


def update_paths(env_name, checkpoint_base_dir):
    """
    Update the path-names for the checkpoint-dir and log-files.

    Call this after you have changed checkpoint_base_dir and
    before you create the Neural Network.

    :param env_name:
        Name of the game-environment you will use in OpenAI Gym.
    """

    global checkpoint_dir
    global log_reward_path
    global log_q_values_path

    # Add the environment-name to the checkpoint-dir.
    checkpoint_dir = checkpoint_base_dir

    # Create the checkpoint-dir if it does not already exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # File-path for the log-file for episode rewards.
    log_reward_path = os.path.join(checkpoint_dir, "log_reward.txt")

    # File-path for the log-file for Q-values.
    log_q_values_path = os.path.join(checkpoint_dir, "log_q_values.txt")


########################################################################
# Classes used for logging data during training.


class Log:
    """
    Base-class for logging data to a text-file during training.

    It is possible to use TensorFlow / TensorBoard for this,
    but it is quite awkward to implement, as it was intended
    for logging variables and other aspects of the TensorFlow graph.
    We want to log the reward and Q-values which are not in that graph.
    """

    def __init__(self, file_path):
        """Set the path for the log-file. Nothing is saved or loaded yet."""

        # Path for the log-file.
        self.file_path = file_path

        # Data to be read from the log-file by the _read() function.
        self.count_episodes = None
        self.count_states = None
        self.data = None

    def _write(self, count_episodes, count_states, msg):
        """
        Write a line to the log-file. This is only called by sub-classes.

        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states:
            Counter for the number of states processed during training.

        :param msg:
            Message to write in the log.
        """

        with open(file=self.file_path, mode='a', buffering=1) as file:
            msg_annotated = "{0}\t{1}\t{2}\n".format(count_episodes, count_states, msg)
            file.write(msg_annotated)

    def _read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states and self.data
        """

        # Open and read the log-file.
        with open(self.file_path) as f:
            reader = csv.reader(f, delimiter="\t")
            self.count_episodes, self.count_states, *data = zip(*reader)

        # Convert the remaining log-data to a NumPy float-array.
        self.data = np.array(data, dtype='float')


class LogReward(Log):
    """Log the rewards obtained for episodes during training."""

    def __init__(self):
        # These will be set in read() below.
        self.episode = None
        self.mean = None

        # Super-class init.
        Log.__init__(self, file_path=log_reward_path)

    def write(self, count_episodes, count_states, reward_episode, reward_mean):
        """
        Write the episode and mean reward to file.

        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states:
            Counter for the number of states processed during training.

        :param reward_episode:
            Reward for one episode.

        :param reward_mean:
            Mean reward for the last e.g. 30 episodes.
        """

        msg = "{0:.1f}\t{1:.1f}".format(reward_episode, reward_mean)
        self._write(count_episodes=count_episodes, count_states=count_states, msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states, self.episode and self.mean
        """

        # Read the log-file using the super-class.
        self._read()

        # Get the episode reward.
        self.episode = self.data[0]

        # Get the mean reward.
        self.mean = self.data[1]


class LogQValues(Log):
    """Log the Q-Values during training."""

    def __init__(self):
        # These will be set in read() below.
        self.min = None
        self.mean = None
        self.max = None
        self.std = None

        # Super-class init.
        Log.__init__(self, file_path=log_q_values_path)

    def write(self, count_episodes, count_states, q_values):
        """
        Write basic statistics for the Q-values to file.

        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states:
            Counter for the number of states processed during training.

        :param q_values:
            Numpy array with Q-values from the replay-memory.
        """

        msg = "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(np.min(q_values),
                                                          np.mean(q_values),
                                                          np.max(q_values),
                                                          np.std(q_values))

        self._write(count_episodes=count_episodes,
                    count_states=count_states,
                    msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states, self.min / mean / max / std.
        """

        # Read the log-file using the super-class.
        self._read()

        # Get the logged statistics for the Q-values.
        self.min = self.data[0]
        self.mean = self.data[1]
        self.max = self.data[2]
        self.std = self.data[3]


########################################################################


def print_progress(msg):
    """
    Print progress on a single line and overwrite the line.
    Used during optimization.
    """

    sys.stdout.write("\r" + msg)
    sys.stdout.flush()
