########################################################################
# This file is based on the TensorFlow Tutorials available at:
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
# Published under the MIT License. See the file LICENSE for details.
# Copyright 2017 by Magnus Erik Hvass Pedersen
########################################################################

import numpy as np

class LinearControlSignal:
    """
    A control signal that changes linearly over time.

    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.

    TensorFlow has functionality for doing this, but it uses the
    global_step counter inside the TensorFlow graph, while we
    want the control signals to use a state-counter for the
    game-environment. So it is easier to make this in Python.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):
        """
        Create a new object.

        :param start_value:
            Start-value for the control signal.

        :param end_value:
            End-value for the control signal.

        :param num_iterations:
            Number of iterations it takes to reach the end_value
            from the start_value.

        :param repeat:
            Boolean whether to reset the control signal back to the start_value
            after the end_value has been reached.
        """

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value


########################################################################


class EpsilonGreedy:
    """
    The epsilon-greedy policy either takes a random action with
    probability epsilon, or it takes the action for the highest
    Q-value.

    If epsilon is 1.0 then the actions are always random.
    If epsilon is 0.0 then the actions are always argmax for the Q-values.

    Epsilon is typically decreased linearly from 1.0 to 0.1
    and this is also implemented in this class.

    During testing, epsilon is usually chosen lower, e.g. 0.05 or 0.01
    """

    def __init__(self, num_actions,
                 epsilon_testing=0.05,
                 num_iterations=1e6,
                 start_value=1.0,
                 end_value=0.1,
                 repeat=False):
        """

        :param num_actions:
            Number of possible actions in the game-environment.

        :param epsilon_testing:
            Epsilon-value when testing.

        :param num_iterations:
            Number of training iterations required to linearly
            decrease epsilon from start_value to end_value.

        :param start_value:
            Starting value for linearly decreasing epsilon.

        :param end_value:
            Ending value for linearly decreasing epsilon.

        :param repeat:
            Boolean whether to repeat and restart the linear decrease
            when the end_value is reached, or only do it once and then
            output the end_value forever after.
        """

        # Store parameters.
        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing

        # Create a control signal for linearly decreasing epsilon.
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
                                                  start_value=start_value,
                                                  end_value=end_value,
                                                  repeat=repeat)

    def get_epsilon(self, iteration, training):
        """
        Return the epsilon for the given iteration.
        If training==True then epsilon is linearly decreased,
        otherwise epsilon is a fixed number.
        """

        if training:
            epsilon = self.epsilon_linear.get_value(iteration=iteration)
        else:
            epsilon = self.epsilon_testing

        return epsilon

    def get_action(self, q_values, iteration, training, epsilon_override=None):
        """
        Use the epsilon-greedy policy to select an action.

        :param q_values:
            These are the Q-values that are estimated by the Neural Network
            for the current state of the game-environment.

        :param iteration:
            This is an iteration counter. Here we use the number of states
            that has been processed in the game-environment.

        :param training:
            Boolean whether we are training or testing the
            Reinforcement Learning agent.

        :param epsilon_override:
            Value to override epsilon for this step only

        :return:
            action (integer), epsilon (float)
        """

        if epsilon_override is None:
            epsilon = self.get_epsilon(iteration=iteration, training=training)
        else:
            epsilon = epsilon_override

        # With probability epsilon.
        if np.random.random() < epsilon:
            # Select a random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Otherwise select the action that has the highest Q-value.
            action = np.argmax(q_values)

        return action, epsilon