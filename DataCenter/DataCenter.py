import numpy as np
import pandas as pd

import DeepLearning.DataCenter.DataProcessing as data

class DataCenter():

    def __init__(self, plot_type='Research', batch_size = 100):
        self.graph_type = plot_type
        self.batch_size = batch_size

    ## Data Management Functions
    def load_all_data_single(self,data_folder,data_file):
        print('Loading Data from CSV file')
        self.data_location = data_folder
        self.all_data = data.load_data(data_folder+data_file)
        self.all_data = np.nan_to_num(self.all_data)

    # Todo
        # def load_all_data_multiple

    def split_input_output_data(self, num_outputs, output_first = True):
        # Split into input and output data
        self.all_input_data, self.all_output_data = data.split_input_output_data(self.all_data, num_outputs, output_first)

        assert self.all_input_data.shape[0] == self.all_output_data.shape[0]

    def cut_input_data_seq_length(self, out_length = None):
        self.all_input_data = data.cut_input_data_seq_length(self.all_input_data, out_length)

    def split_train_val_eval(self, val_split=0.3, eval_split=0, shuffle = False):

        if shuffle == True:
            self.all_input_data, self.all_output_data = data.shuffle_input_output(self.all_input_data, self.all_output_data)

        total_samples = int(self.all_input_data.shape[0])

        # Split into training/validation/evaluation data
        self.eval_samples = int(self.all_input_data.shape[0] * eval_split)
        self.val_samples = int(self.all_input_data.shape[0] * val_split)
        self.train_samples = int(self.all_input_data.shape[0] - self.eval_samples - self.val_samples)
        print('Train Samples = {}({}%), Val Samples = {}({}%), Eval Samples = {}({}%)'.format(self.train_samples,
                                                                                              np.round(
                                                                                                  self.train_samples / total_samples * 100,
                                                                                                  2),
                                                                                              self.val_samples,
                                                                                              np.round(
                                                                                                  self.val_samples / total_samples * 100,
                                                                                                  2),
                                                                                              self.eval_samples,
                                                                                              np.round(
                                                                                                  self.eval_samples / total_samples * 100,
                                                                                                  2)))

        self.eval_input_data = self.all_input_data[total_samples - self.eval_samples:]
        self.eval_output_data = self.all_output_data[total_samples - self.eval_samples:]

        self.val_input_data = self.all_input_data[
                              total_samples - self.eval_samples - self.val_samples:total_samples - self.eval_samples]
        self.val_output_data = self.all_output_data[
                               total_samples - self.eval_samples - self.val_samples:total_samples - self.eval_samples]

        self.train_input_data = self.all_input_data[:total_samples - self.eval_samples - self.val_samples]
        self.train_output_data = self.all_output_data[:total_samples - self.eval_samples - self.val_samples]

    def save_data(self, folder_path='./data/', file_prefix=''):
        print('Savings data to {}'.format(folder_path))
        self.folder_path = folder_path
        np.save(folder_path + file_prefix + 'training_input_data.npy', self.train_input_data)
        np.save(folder_path + file_prefix + 'validation_input_data.npy', self.val_input_data)
        np.save(folder_path + file_prefix + 'evaluation_input_data.npy', self.eval_input_data)

        np.save(folder_path + file_prefix + 'training_output_data.npy', self.train_output_data)
        np.save(folder_path + file_prefix + 'validation_output_data.npy', self.val_output_data)
        np.save(folder_path + file_prefix + 'evaluation_output_data.npy', self.eval_output_data)

    def load_data(self, folder_path='./data/', file_prefix=''):
        print('Loading data from {}'.format(folder_path))
        self.train_input_data = np.load(folder_path + file_prefix + 'training_input_data.npy')
        self.val_input_data = np.load(folder_path + file_prefix + 'validation_input_data.npy')
        self.eval_input_data = np.load(folder_path + file_prefix + 'evaluation_input_data.npy')

        self.train_output_data = np.load(folder_path + file_prefix + 'training_output_data.npy')
        self.val_output_data = np.load(folder_path + file_prefix + 'validation_output_data.npy')
        self.eval_output_data = np.load(folder_path + file_prefix + 'evaluation_output_data.npy')

        self.train_samples = self.train_input_data.shape[0]
        self.val_samples = self.val_input_data.shape[0]
        self.eval_samples = self.eval_input_data.shape[0]
        total_samples = self.train_samples + self.val_samples + self.eval_samples

        print('Train Samples = {}({}%), Val Samples = {}({}%), Eval Samples = {}({}%)'.format(self.train_samples,
                                                                                              np.round(
                                                                                                  self.train_samples / total_samples * 100,
                                                                                                  2),
                                                                                              self.val_samples,
                                                                                              np.round(
                                                                                                  self.val_samples / total_samples * 100,
                                                                                                  2),
                                                                                              self.eval_samples,
                                                                                              np.round(
                                                                                                  self.eval_samples / total_samples * 100,
                                                                                                  2)))

    # Input/Output Scaling
    def scale_input(self, scale = None):
        self.all_input_data, self.input_scale = data.scale_input(self.all_input_data, scale=scale)

    def reshape_1D_input(self):
        self.all_input_data = data.reshape_1D_input(self.all_input_data)

    def one_hot_output(self, column, concat = True):
        self.all_output_data, self.one_hot_labels = data.one_hot_output(self.all_output_data, column, concat=concat)

        # Save one_hot_labels
        np.savetxt(self.data_location + 'one_hot_labels.csv', self.one_hot_labels, delimiter=',')

    def scale_outputs(self):
        self.all_output_data, self.output_scale = data.scale_outputs(self.all_output_data)

    ## Neural Network Functions
    def reset_train_batches(self, batch_size, num_batches = None):
        self.batch_size = batch_size

        if num_batches != None:
            self.num_train_batches = int(num_batches)
        else:
            self.num_train_batches = int(np.floor(self.train_input_data.shape[0]/batch_size))

        self.train_input_batches = self.train_input_data
        self.train_output_batches = self.train_output_data

        self.train_input_batches, self.train_output_batches = data.shuffle_input_output(self.train_input_batches,self.train_output_batches)

        self.train_input_batches = self.train_input_batches[:self.batch_size * self.num_train_batches]
        self.train_output_batches = self.train_output_batches[:self.batch_size * self.num_train_batches]

    def reset_val_batches(self):

        self.val_batches = int(np.floor(self.val_input_data.shape[0]/self.batch_size))

        if self.val_batches > 0:
            self.val_input_batches = self.val_input_data
            self.val_output_batches = self.val_output_data

            self.val_input_batches, self.val_output_batches = data.shuffle_input_output(self.val_input_batches,self.val_output_batches)

            self.val_input_batches = self.val_input_batches[:self.batch_size * self.val_batches]
            self.val_output_batches = self.val_output_batches[:self.batch_size * self.val_batches]

        else:
            self.val_input_batches = np.zeros((self.batch_size, self.val_input_data.shape[1]))
            self.val_output_batches = np.zeros((self.batch_size, self.val_output_data.shape[1]))

            self.val_input_batches = data.reshape_1D_input(self.val_input_batches)

            self.val_input_batches[:self.val_input_data.shape[0]] = self.val_input_data
            self.val_output_batches[:self.val_output_data.shape[0]] = self.val_output_data

    def reset_eval_batches(self):

        self.eval_batches = int(np.floor(self.eval_input_data.shape[0]/self.batch_size))

        if self.eval_batches > 0:
            self.eval_input_batches = self.eval_input_data
            self.eval_output_batches = self.eval_output_data

            self.eval_input_batches, self.eval_output_batches = data.shuffle_input_output(self.eval_input_batches,self.eval_output_batches)

            self.eval_input_batches = self.eval_input_batches[:self.batch_size * self.eval_batches]
            self.eval_output_batches = self.eval_output_batches[:self.batch_size * self.eval_batches]

        else:
            self.eval_input_batches = np.zeros((self.batch_size, self.eval_input_data.shape[1]))
            self.eval_output_batches = np.zeros((self.batch_size, self.eval_output_data.shape[1]))

            self.eval_input_batches = data.reshape_1D_input(self.eval_input_batches)

            self.eval_input_batches[:self.eval_input_data.shape[0]] = self.eval_input_data
            self.eval_output_batches[:self.eval_output_data.shape[0]] = self.eval_output_data

    def reset_all_batches(self):
        self.reset_train_batches(self.batch_size)
        self.reset_val_batches()
        self.reset_eval_batches()