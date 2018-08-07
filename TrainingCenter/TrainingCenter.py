import tensorflow as tf
import numpy as np

from DeepLearning.Misc.ProgBar import new_prog_bar, update_prog_bar

import DeepLearning.Tensorflow_Base_Functions.evaluation as tfEval
import DeepLearning.Tensorflow_Base_Functions.loss as tfLoss
import DeepLearning.Tensorflow_Base_Functions.optimizers as tfOptimizers

import time

class TrainingCenter():

    def __init__(self):

        self.loss_type = 'categorical_cross_entropy'

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.model_save_folder = './saved_model/'
        self.model_save_name = 'model'

        self.model_load_folder = './saved_model'
        self.model_load_name = 'model'

        self.prog_bar = new_prog_bar()

        self.switch_load_model = False

        self.val_metrics = ['predict_val_acc']
        self.update_val_metrics_on_batch_update = False
        self.update_val_metrics_on_n_batches = False

        self.loss = 'categorical_cross_entropy'
        self.optimizer = 'adam'

        # Internal Parameters
        self._batch_num = 0
        self.val_acc = 0
        self.train_acc = 0
        self.best_val_acc = 0
        self.epoch = 0

        # Tensorboard Parameters
        self.tb_epoch_train_loss_var = tf.Variable(0, dtype=tf.float32)
        self.tb_epoch_train_acc_var = tf.Variable(0,dtype=tf.float32)
        self.tb_epoch_val_acc_var = tf.Variable(0, dtype=tf.float32)

        self.tb_time_train_loss_var = tf.Variable(0, dtype=tf.float32)
        self.tb_time_val_acc_var = tf.Variable(0, dtype=tf.float32)

        self.start_time = time.time()
        self.last_time = time.time()
        self.time_count = 0
        self.tb_time_step = 1 # Minutes

    def initialize_sess(self):
        self.sess.run(tf.global_variables_initializer())

    def update_val_metrics(self, DataCenter, model):
        # Epoch Train Loss
        self.epoch_loss_latest = np.mean(self.epoch_loss)

        # Prediction Accuracy
        if 'predict_val_acc' in self.val_metrics:
            self.val_acc = tfEval.prediction_accuracy(DataCenter, model,
                                                      DataCenter.val_input_batches,
                                                      DataCenter.val_output_batches)
            print('Validation Acc = {}'.format(self.val_acc))
            # ToDo: Add to Tensorboard

        if 'predict_train_acc' in self.val_metrics:
            self.train_acc = tfEval.prediction_accuracy(DataCenter, model,
                                                      DataCenter.train_input_batches,
                                                      DataCenter.train_output_batches)
            print('Training Acc = {}'.format(self.train_acc))

        if 'export_val_predictions' in self.val_metrics:
            val_predictions = tfEval.export_val_one_hot_predictions(DataCenter, model)
            np.savetxt(self.model_save_folder + self.model_save_name + 'val_predictions.csv', val_predictions, delimiter=',')


    def create_epoch_tensorboard(self):
        self.tb_epoch_train_loss = tf.summary.scalar('Training Loss', self.tb_epoch_train_loss_var)
        self.tb_epoch_train_acc = tf.summary.scalar('Training Acc', self.tb_epoch_train_acc_var)
        self.tb_epoch_val_acc = tf.summary.scalar('Validation Accuracy', self.tb_epoch_val_acc_var)

        self.tb_epoch_merged = tf.summary.merge([self.tb_epoch_train_loss,
                                                 self.tb_epoch_train_acc,
                                                 self.tb_epoch_val_acc])

        self.tb_epoch_train_writer = tf.summary.FileWriter(self.model_save_folder + 'epoch_tb')

    def create_time_tensorboard(self):
        self.tb_time_train_loss = tf.summary.scalar('Training Loss', self.tb_time_train_loss_var)
        self.tb_time_val_acc = tf.summary.scalar('Validation Accuracy', self.tb_time_val_acc_var)

        self.tb_time_merged = tf.summary.merge([self.tb_time_train_loss,
                                                 self.tb_time_val_acc])

        self.tb_time_train_writer = tf.summary.FileWriter(self.model_save_folder + 'time_tb')


    def update_epoch_tensorboard(self):
        # Assign Variables
        self.sess.run([self.tb_epoch_train_acc_var.assign(self.train_acc),
                       self.tb_epoch_val_acc_var.assign(self.val_acc),
                       self.tb_epoch_train_loss_var.assign(self.epoch_loss_latest)])

        summary = self.sess.run(self.tb_epoch_merged)

        self.tb_epoch_train_writer.add_summary(summary, self.epoch)

    def update_time_tensorboard(self, DataCenter, model):
        self.time_step_sec = self.tb_time_step * 60
        next_time = self.last_time + self.time_step_sec

        if time.time() > next_time:
            self.last_time = time.time()
            self.update_time_tensorboard_vars(DataCenter, model)

            # Assign Variables
            self.sess.run([self.tb_time_val_acc_var.assign(self.time_val_acc),
                           self.tb_time_train_loss_var.assign(self.time_train_loss)])

            summary = self.sess.run(self.tb_time_merged)

            self.tb_time_train_writer.add_summary(summary, self.time_count)
            self.time_count += 1

    def update_time_tensorboard_vars(self, DataCenter, model):

        rand_choice = np.random.randint(0,DataCenter.val_input_batches.shape[0],size=5)

        x_data = DataCenter.val_input_batches[rand_choice]
        y_data = DataCenter.val_output_batches[rand_choice]

        self.time_val_acc = tfEval.prediction_accuracy(DataCenter, model, x_data, y_data)
        self.time_train_loss = np.mean(self.epoch_loss[:self._batch_num])

    def prog_bar_update(self):
        update = self._batch_num/self.num_train_batches
        update_prog_bar(self.prog_bar, update)

    def load_model(self, load):
        if load is True:
            new_saver = tf.train.import_meta_graph(self.model_load_folder + self.model_load_name + '.meta')
            new_saver.restore(self.sess, tf.train.latest_checkpoint(self.model_load_folder))

    def save_sess(self):
        print('Saving Model')
        self.saver.save(self.sess, self.model_save_folder + self.model_save_name)

    def save_model(self, save=True, save_on_best=True, save_on_best_metric='acc'):
        self.calc_best_model()

        if save is True:
            if save_on_best is True:
                if save_on_best_metric is True:
                    if self.best_model is True:
                        self.save_sess()
                elif save_on_best_metric is False:
                    self.save_sess()
            elif save_on_best is False:
                self.save_sess()

    def calc_best_model(self):
        self.best_model = False
        if self.val_acc > self.best_val_acc:
            self.best_model = True
            self.best_val_acc = self.val_acc

    def initialize_loss(self, DataCenter, model):
        if self.loss == 'categorical_cross_entropy':
            print('Setting the Loss to: Categorical Cross Entropy')
            self._loss = tfLoss.categorical_cross_entropy(DataCenter, model)

    def initialize_optimizer(self, DataCenter):
        if self.optimizer == 'adam':
            print('Setting the Optimizer to: Adam')
            self._optimizer = tfOptimizers.adam_optimizer(DataCenter.learning_rate)

    def train_step(self):
        _, self.step_c = self.sess.run([self.learning_step, self._loss],
                                       feed_dict={self.x: self.train_input_batch,
                                                  self.y: self.train_output_batch})
        self.update_epoch_loss()

    def set_train_batch(self, DataCenter):
        self.train_input_batch = DataCenter.train_input_batches[self._batch_num]  # Get Input Data for batch
        self.train_output_batch = DataCenter.train_output_batches[self._batch_num]  # Get Output Data for batch

    def set_placeholders(self, DataCenter):
        self.x = DataCenter.x_placeholder
        self.y = DataCenter.y_placeholder

    def set_learning_step(self, DataCenter, model):
        self.initialize_loss(DataCenter, model)
        self.initialize_optimizer(DataCenter)
        self.learning_step = self._optimizer.minimize(self._loss)

    def reset_epoch_loss(self):
        self.epoch_loss = np.zeros(self.num_train_batches)

    def update_epoch_loss(self):
        self.epoch_loss[self._batch_num] = self.step_c

    def update_consol(self):
        self.prog_bar_update()
        print('\n' + str(self.step_c))

    def train_model(self, DataCenter, model, save=True, load=False, save_on_best=True, save_on_best_metric='acc'):

        self.set_placeholders(DataCenter)                           # Create Placeholders
        self.set_learning_step(DataCenter, model)                   # Create Cost, Optimiser and learning step
        self.num_train_batches = DataCenter.num_train_batches       # Set training batches per epoch

        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            #self.initialize_sess()                                     # Initialise Session
            self.load_model(load)                                       # Check if we need to load model. If so, load
            self.create_epoch_tensorboard()                             # Create Tensorboard Parameters
            self.create_time_tensorboard()                              # Create Time Based Tensorboard
            for self.epoch in range(DataCenter.epochs):                 # Iterate over all epochs
                DataCenter.reset_train_batches()                        # Reset DataCenter Training Batches
                self.reset_epoch_loss()                                 # Reset Epoch Loss

                for self._batch_num in range(self.num_train_batches):   # Iterate over all batches
                    self.set_train_batch(DataCenter)                    # Set training batch data
                    self.train_step()                                   # Run 1 step
                    self.update_consol()                                # Update Progress Bar
                    self.update_time_tensorboard(DataCenter, model)     # Update time based tensorboard if needed

                    if self.update_val_metrics_on_n_batches is not False:
                        if self._batch_num != 0 and self._batch_num%self.update_val_metrics_on_n_batches == 0:
                            self.update_val_metrics(DataCenter, model)
                            self.update_epoch_tensorboard()

                # Validation Metrics
                self.update_val_metrics(DataCenter, model)
                self.update_epoch_tensorboard()
                self.save_model(save, save_on_best, save_on_best_metric)