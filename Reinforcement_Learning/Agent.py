########################################################################
# This file is based on the TensorFlow Tutorials available at:
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
# Published under the MIT License. See the file LICENSE for details.
# Copyright 2017 by Magnus Erik Hvass Pedersen
########################################################################

import numpy as np
import time
import gym

import os
import shutil

import Logging as Logging
import NeuralNetwork as NeuralNetwork
import ReplayMemory
import RLFunctions

class Agent:
    """
    This implements the function for running the game-environment with
    an agent that uses Reinforcement Learning. This class also creates
    instances of the Replay Memory and Neural Network.
    """

    def __init__(self, env_name, Monty, CommandCenter, render=False, use_logging=True, verbose=False):
        """
        Create an object-instance. This also creates a new object for the
        Replay Memory and the Neural Network.

        Replay Memory will only be allocated if training==True.

        :param env_name:
            Name of the game-environment in OpenAI Gym.
            Examples: 'Breakout-v0' and 'SpaceInvaders-v0'

        :param training:
            Boolean whether to train the agent and Neural Network (True),
            or test the agent by playing a number of episodes of the game (False).

        :param render:
            Boolean whether to render the game-images to screen during testing.

        :param use_logging:
            Boolean whether to use logging to text-files during training.
        """
        self.replay_size = 5000

        # Whether this bot is acting as a worker -> saving data rather than processing
        self.worker = CommandCenter.worker
        self.trainer = CommandCenter.trainer

        self.env_name = env_name
        # Create the game-environment using OpenAI Gym.
        self.env = gym.make(self.env_name)

        # Get Checkpoint Directory
        # self.checkpoint_dir = CommandCenter.model_path
        if self.worker:
            checkpoint_base_dir = CommandCenter.data_path
        elif self.trainer:
            checkpoint_base_dir = CommandCenter.model_path
        else:
            checkpoint_base_dir = None

        Logging.update_paths(env_name=env_name, checkpoint_base_dir=checkpoint_base_dir)

        # The number of possible actions that the agent may take in every step.
        self.num_actions = self.env.action_space.n

        # Whether to render each image-frame of the game-environment to screen.
        self.render = render

        # Whether to use logging during training.
        self.use_logging = use_logging
        self.verbose = verbose

        if self.use_logging and self.worker:
            # Used for logging Q-values and rewards during training.
            self.log_q_values = Logging.LogQValues()
            self.log_reward = Logging.LogReward()
        else:
            self.log_q_values = None
            self.log_reward = None

        # List of string-names for the actions in the game-environment.
        self.action_names = self.env.unwrapped.get_action_meanings()

        self.epsilon_greedy = RLFunctions.EpsilonGreedy(start_value=1.0,
                                                        end_value=0.1,
                                                        num_iterations=1e6,
                                                        num_actions=self.num_actions,
                                                        epsilon_testing=0.01)

        self.replay_fraction = 1.0
        self.learning_rate = 1e-5
        self.loss_limit = 0.01
        self.max_epochs = 5.0

        # We only create the replay-memory when we are training the agent,
        # because it requires a lot of RAM. The image-frames from the
        # game-environment are resized to 105 x 80 pixels gray-scale,
        # and each state has 2 channels (one for the recent image-frame
        # of the game-environment, and one for the motion-trace).
        # Each pixel is 1 byte, so this replay-memory needs more than
        # 3 GB RAM (105 x 80 x 2 x 200000 bytes).
        self.replay_memory = ReplayMemory.ReplayMemory(size=self.replay_size, state_shape=Monty.get_nn_state_shape(), num_actions=self.num_actions)

        # Create the Neural Network used for estimating Q-values.
        self.model = NeuralNetwork.NeuralNetwork(num_actions=self.num_actions,
                                                 state_shape=Monty.get_nn_state_shape(),
                                                 checkpoint_dir=CommandCenter.model_path,
                                                 worker=self.worker)

        # Log of the rewards obtained in each episode during calls to run()
        self.episode_rewards = []

        self.agent_model_version = CommandCenter.model_version

    def reset_episode_rewards(self):
        """Reset the log of episode-rewards."""
        self.episode_rewards = []

    def get_action_name(self, action):
        """Return the name of an action."""
        return self.action_names[action]

    def get_lives(self):
        """Get the number of lives the agent has in the game-environment."""
        return self.env.unwrapped.ale.lives()

    def check_end_of_life(self, num_lives, end_episode):

        # Determine if a life was lost in this step.
        num_lives_new = self.get_lives()
        end_life = (num_lives_new < num_lives)

        if end_life:
            end_episode = True

        return end_life, end_episode

    def run_worker(self, Monty, CommandCenter):
        """
        Run the game-environment and use the Neural Network to decide
        which actions to take in each step through Q-value estimates.

        :param num_episodes:
            Number of episodes to process in the game-environment.
            If None then continue forever. This is useful during training
            where you might want to stop the training using Ctrl-C instead.
        """

        # This will cause a reset in the first iteration of the following loop.
        end_episode = True
        reward_episode = 0.0
        num_lives = 0

        # Counter for the number of states we have processed.
        # This is stored in the TensorFlow graph so it can be
        # saved and reloaded along with the checkpoint.
        count_states = self.model.get_count_states()

        # Counter for the number of episodes we have processed.
        count_episodes = self.model.get_count_episodes()
        sess_episodes = 0

        try:
            training_array = np.load(CommandCenter.data_path + 'TrainingArray.npy')
            epsilon_array = np.load(CommandCenter.data_path + 'EpsilonArray.npy')
        except:
            training_array, epsilon_array = Monty.create_training_array()

        Monty.send_training_array(training_array, epsilon_array)

        while True:
            if end_episode:

                # Reset the game-environment and get the first image-frame.
                img = self.env.reset()

                # Reset the reward for the entire episode to zero.
                # This is only used for printing statistics.
                reward_episode = 0.0

                # Increase the counter for the number of episodes.
                # This counter is stored inside the TensorFlow graph
                # so it can be saved and restored with the checkpoint.
                count_episodes = self.model.increase_count_episodes()

                # Get the number of lives that the agent has left in this episode.
                num_lives = self.get_lives()

                # Keep a record of current training array
                training_array, epsilon_array = Monty.get_training_array()

                # Reset Monty Agent (This deletes the training array)
                Monty.reset(img, sess_episodes)

                # Send training array to monty
                Monty.send_training_array(training_array, epsilon_array)
                sess_episodes += 1

            # Get the state of the game-environment from the motion-tracer.
            # The state has two images: (1) The last image-frame from the game
            # and (2) a motion-trace that shows movement trajectories.
            state = Monty.get_nn_state()

            if self.verbose:
                Monty.view_states()

            # Use the Neural Network to estimate the Q-values for the state.
            # Note that the function assumes an array of states and returns
            # a 2-dim array of Q-values, but we just have a single state here.
            q_values = self.model.get_q_values(states=[state])[0]

            if self.render:
                epsilon = 0.03
            else:
                epsilon = Monty.get_epsilon()

            # Determine the action that the agent must take in the game-environment.
            # The epsilon is just used for printing further below.
            action, epsilon = self.epsilon_greedy.get_action(q_values=q_values,
                                                             iteration=count_states,
                                                             training=self.worker,
                                                             epsilon_override=epsilon)

            # Take a step in the game-environment using the given action.
            img, _, end_episode, info = self.env.step(action=action)

            # Check if end of game
            episode_failed, end_episode = self.check_end_of_life(num_lives=num_lives, end_episode=end_episode)

            # Update Monty
            Monty.update(img, episode_failed)

            # Update Reward
            reward_episode += Monty.get_reward()

            if Monty.end_due_to_reward:
                end_episode = True

            if Monty.end_due_to_time:
                print('Ended Due To Time')
                end_episode = True

            # Increase the counter for the number of states that have been processed.
            count_states = self.model.increase_count_states()

            # If we want to render the game
            if self.render:
                self.env.render()
                time.sleep(0.005)

            # Add the state of the game-environment to the replay-memory.
            self.replay_memory.add(state=state,
                                   q_values=q_values,
                                   action=action,
                                   reward=Monty.reward,
                                   end_life=episode_failed,
                                   end_episode=end_episode)

            # When the replay-memory is sufficiently full.
            if self.replay_memory.is_full():


                # Update all Q-values in the replay-memory through a backwards-sweep.
                self.replay_memory.update_all_q_values()

                # Log statistics for the Q-values to file.
                if self.use_logging:
                    self.log_q_values.write(count_episodes=count_episodes,
                                            count_states=count_states,
                                            q_values=self.replay_memory.q_values)

                # Get the control parameters for optimization of the Neural Network.
                # These are changed linearly depending on the state-counter.
                # learning_rate = self.learning_rate_control.get_value(iteration=count_states)
                # loss_limit = self.loss_limit_control.get_value(iteration=count_states)
                # max_epochs = self.max_epochs_control.get_value(iteration=count_states)

                # Save Training and Epsilon Arrays
                np.save(CommandCenter.data_path + 'TrainingArray.npy', training_array)
                np.save(CommandCenter.data_path + 'EpsilonArray.npy', epsilon_array)

                # Save Replay Memory
                data_save_folder = CommandCenter.data_path + 'data_' +str(time.time())[:10] + '/'

                if not os.path.exists(data_save_folder):
                    os.makedirs(data_save_folder)

                try:
                    self.replay_memory.save_numpy_arrays(data_save_folder)
                except FileNotFoundError:
                    pass

                self.replay_memory.reset()


                CommandCenter.update_worker_paths()

                if CommandCenter.model_version != self.agent_model_version:
                    loaded = False

                    while loaded is False:
                        Logging.update_paths(env_name=self.env_name, checkpoint_base_dir=CommandCenter.data_path)
                        self.model.update_checkpoint_dir(CommandCenter.model_path)
                        loaded = self.model.load_checkpoint(reset_if_error=False)

                        if loaded is True:
                            self.agent_model_version = CommandCenter.model_version
                        else:
                            print('Tried to load model. Trying again.')
                            CommandCenter.update_worker_paths()
                            time.sleep(5)

            if end_episode:
                # Add the episode's reward to a list for calculating statistics.
                self.episode_rewards.append(reward_episode)

            # Mean reward of the last 30 episodes.
            if len(self.episode_rewards) == 0:
                # The list of rewards is empty.
                reward_mean = 0.0
            else:
                reward_mean = np.mean(self.episode_rewards[-30:])

            if end_episode:
                # Log reward to file.
                if self.use_logging:
                    self.log_reward.write(count_episodes=count_episodes,
                                          count_states=count_states,
                                          reward_episode=reward_episode,
                                          reward_mean=reward_mean)

                # Print reward to screen.
                msg = "{0:4}:{1}\t Epsilon: {2:4.2f}\t Reward: {3:.1f}\t Episode Mean: {4:.1f}"
                print(msg.format(count_episodes, count_states, epsilon,
                                 reward_episode, reward_mean))

            # elif (Monty.reward != 0.0 or episode_failed or end_episode):
            #     # Print Q-values and reward to screen.
            #     msg = "{0:4}:{1}\tQ-min: {2:5.3f}\tQ-max: {3:5.3f}\tSess_Eps: {4}\tReward: {5:.1f}\tEpisode Mean: {6:.1f}"
            #     print(msg.format(count_episodes, count_states, np.min(q_values),
            #                      np.max(q_values), sess_episodes, reward_episode, reward_mean))

    def run_trainer(self, CommandCenter):

        self.replay_memory.num_used = self.replay_size

        while True:
            CommandCenter.trainer_find_data_filepaths()

            # Check that all_valid_data_paths is not empty
            if CommandCenter.all_valid_data_paths:
                increment = CommandCenter.calc_increment_trainer()

                if increment:
                    CommandCenter.delete_old_data_path()
                    print('Total Folders Deleted = {}'.format(CommandCenter.deleted_folders))

                    # Save a checkpoint of the Neural Network so we can reload it.
                    self.model.update_next_version_checkpoint_dir(CommandCenter.model_path)
                    self.model.save_checkpoint()

                for data_path in CommandCenter.all_valid_data_paths:
                    print('Using Data from '.format(data_path))

                    try:
                        self.replay_memory.load_numpy_arrays(data_path=data_path)
                        # Perform an optimization run on the Neural Network so as to
                        # improve the estimates for the Q-values.
                        # This will sample random batches from the replay-memory.
                        self.model.optimize(replay_memory=self.replay_memory,
                                            learning_rate=self.learning_rate,
                                            loss_limit=self.loss_limit,
                                            max_epochs=self.max_epochs)

                        shutil.rmtree(data_path)

                    except:
                        try:
                            shutil.rmtree(data_path)
                        except:
                            pass

            else:
                print('Waiting for Data for Version {}'.format(CommandCenter.model_version))
                time.sleep(5)