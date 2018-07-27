import numpy as np
import DeepLearning.DataCenter.DataProcessing as data


class DataCenter():
    ''' A DataCenter object can be considered the hard-drive of the model. We use it to store any information or
    data we need for deep learning Tasks:

    Categories:

        Data:
            - Training Data
            - Validation Data
            - Evaluation Data

        Neural Network:
            - Framework being used
                - Keras
                - Tensorflow

            - NN parameters:
                batch_size
                learning_rate
                loss
                cost

        Graphing:
            - Academic
            - Research

    '''

    def __init__(self):

        ## Neural Network Default Parameters
        self.frame_work = 'Tensorflow'
        self.learning_rate = 0.005
        self.batch_size = 100
        self.epochs = 10

        self.num_train_batches = 2
        self.num_val_batches = 2
        self.num_eval_batches = 2

        ## Folder Paths
        self.model_save_folder = 'saved_models/'
        self.model_save_name = 'latest_model'
        self.model_load_folder = 'saved_models/'
        self.model_load_name = 'latest_model'

        self.new_model_scope = 'new_model'
        self.transfer_model_scope = 'transfer'

        self.x_placeholder = None
        self.y_placeholder = None

        # Network Save Controls
        self.save_network_increment = 0.01
        self.recalc_eval = 1
        self.eval_metric = 'best_val_accuracy'
        self.max_save_mse = 0
        self.early_stopping = self.epochs

        # Learning Rate Decay
        self.lr_start = 0.00001
        self.lr_decay_rate = 0.95
        self.lr_step_rate = 25

        # Transfer Learning
        self.transfer_var_names = None
        self.transfer_var_values = None

        # Training History
        self.loss_train_log = None
        self.loss_val_log = None
        self.acc_train_log = None
        self.acc_val_log = None

        # Data
        self.all_data = None
        self.folder_path = './data/'
        self.data_location = './'
        self.file_prefix = ''

        self.train_input_data = None
        self.val_input_data = None
        self.eval_input_data = None

        self.one_hot_labels = None
        self.one_hot_balance_rate = None
        self.one_hot_balance_samples = 0

        # Predictions
        self.predict_input_data = []
        self.predict_output_data = []
        self.truth_output_data = []

        # Cost Functions
        self.set_cost_function = 'mse'

        # Graphing Parameters
        self.graph_type = 'Research'

        ## Manipulating Data
        # Continuous Value One Hot Array
        self.one_hot_length = None
        self.one_hot_val_min = None
        self.one_hot_val_max = None

        # Dynamic Updating MSE Loss
        self.dyn_mse_base_width = 100
        self.dyn_mse_top_width = 1
        self.dyn_mse_power = 1
        self.dyn_mse_offset = 0


    ## Data Management Functions
    def load_all_data_single(self,data_folder,data_file):
        print('Loading Data from CSV file')
        self.data_location = data_folder
        self.all_data = data.load_data(data_folder+data_file)
        self.all_data = np.nan_to_num(self.all_data)

    def load_all_data_multiple(self, data_folder, data_files):
        print('Loading Data from multiple CSV files')
        print('Loading from {}. {} Left'.format(data_folder + data_files[0], len(data_files)))
        self.data_location = data_folder
        self.all_data = data.load_data(data_folder + data_files[0])
        print('Current Samples = {}'.format(self.all_data.shape[0]))

        for i in range(1,len(data_files)):
            print('Loading from {}. {} Left'.format(data_folder + data_files[i], len(data_files)-i))
            new_data = data.load_data(data_folder + data_files[i])
            self.all_data = np.concatenate([self.all_data, new_data], axis=0)
            print('Current Samples = {}'.format(self.all_data.shape[0]))

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

    def save_data(self):
        path = self.folder_path + self.file_prefix
        print('Savings data to {}'.format(self.folder_path))

        np.save(path + 'training_input_data.npy', self.train_input_data)
        np.save(path + 'validation_input_data.npy', self.val_input_data)
        np.save(path + 'evaluation_input_data.npy', self.eval_input_data)

        np.save(path + 'training_output_data.npy', self.train_output_data)
        np.save(path + 'validation_output_data.npy', self.val_output_data)
        np.save(path + 'evaluation_output_data.npy', self.eval_output_data)

    def save_train_data(self):
        path = self.folder_path + self.file_prefix
        print('Savings Training data to {}'.format(self.folder_path))
        np.save(path + 'training_input_data.npy', self.train_input_data)
        np.save(path + 'training_output_data.npy', self.train_output_data)

    def save_val_data(self):
        path = self.folder_path + self.file_prefix
        print('Savings Validation data to {}'.format(self.folder_path))
        np.save(path + 'validation_input_data.npy', self.val_input_data)
        np.save(path + 'validation_output_data.npy', self.val_output_data)

    def save_eval_data(self):
        path = self.folder_path + self.file_prefix
        print('Savings Validation data to {}'.format(self.folder_path))
        np.save(path + 'evaluation_input_data.npy', self.eval_input_data)
        np.save(path + 'evaluation_output_data.npy', self.eval_output_data)

    def load_data(self):
        path = self.folder_path + self.file_prefix

        print('Loading data from {}'.format(path))
        self.train_input_data = np.load(path + 'training_input_data.npy')
        self.val_input_data = np.load(path + 'validation_input_data.npy')
        self.eval_input_data = np.load(path + 'evaluation_input_data.npy')

        self.train_output_data = np.load(path + 'training_output_data.npy')
        self.val_output_data = np.load(path + 'validation_output_data.npy')
        self.eval_output_data = np.load(path + 'evaluation_output_data.npy')

        try:
            self.one_hot_labels = data.load_data(self.folder_path + self.file_prefix + 'one_hot_labels.csv')
        except:
            'No one-hot labels'

        self.print_num_samples()

    # Input/Output Scaling
    def scale_input(self, scale = None):
        self.all_input_data, self.input_scale = data.scale_input(self.all_input_data, scale=scale)

    def scale_multi_chan_input(self, scale = None):
        self.all_input_data, self.input_scale = data.scale_multi_chan_input(self.all_input_data, scale=scale)

    def reshape_1D_input(self):
        self.all_input_data = data.reshape_1D_input(self.all_input_data)

    def one_hot_output(self, column, concat=False):
        self.all_output_data, self.one_hot_labels = data.one_hot_output(self.all_output_data, column, concat=concat)

        # Save one_hot_labels
        np.savetxt(self.data_location + self.file_prefix + 'one_hot_labels.csv', self.one_hot_labels, delimiter=',')
        np.savetxt(self.folder_path + self.file_prefix + 'one_hot_labels.csv', self.one_hot_labels, delimiter=',')

    def contin_one_hot_output(self):
        self.all_output_data, self.one_hot_range = data.create_continuous_one_hot_array(self.all_output_data,
                                                                                        self.one_hot_val_min,
                                                                                        self.one_hot_val_max,
                                                                                        self.one_hot_length)
        # Save one_hot range
        np.savetxt(self.data_location + self.file_prefix + 'one_hot_labels.csv', self.one_hot_range, delimiter=',')
        np.savetxt(self.folder_path + self.file_prefix + 'one_hot_labels.csv', self.one_hot_range, delimiter=',')

    def balance_batch_for_dual_sided_one_hot(self):
        print('Balancing Batches for Dual Sided One Hot Array')
        if self.one_hot_balance_rate is None:
            self.one_hot_balance_rate = 1

        self.load_data()
        self.train_input_data, self.train_output_data = data.balance_batch_for_dual_sided_one_hot(self.train_input_data, self.train_output_data)

    def padd_one_hot_output(self, pad_reduce):
        self.all_output_data = data.padd_one_hot_array(self.all_output_data, pad_reduce)

    def scale_outputs(self):
        self.all_output_data, self.output_scale = data.scale_outputs(self.all_output_data)

    ## Neural Network Functions
    def reset_train_batches(self, batch_size = None, num_batches = None):
        ''' This function resets the training batches

        Example:    Training data set = 1000 samples,
                    Batch size = 300

        Framework:  Keras
            - The output is an array of as many training samples that fit within the batch size.
                    Train_input_batches.shape = [900, data_size]

        Framework: Tensorflow
            - The Tensorflow implementation requires each mini-batch to be explicitly set.
                    Train_input_batches.shape = (# batches, ) - In each batch is a numpy array of size (batch_size, data_size)

        :param batch_size:
        :param num_batches:
        :return:
        '''

        # Update batch size if passed
        if batch_size is not None:
            self.batch_size = batch_size

        # Calc number of batches
        if num_batches is not None:
            self.num_train_batches = int(num_batches)
        else:
            self.num_train_batches = int(np.floor(self.train_input_data.shape[0]/self.batch_size))

        # Copy all training data
        self.train_input_batches = self.train_input_data
        self.train_output_batches = self.train_output_data

        # Shuffle Training data
        self.train_input_batches, self.train_output_batches = data.shuffle_input_output(self.train_input_batches,self.train_output_batches)

        ## Restrict the amount of training Data a number that fits in the number of batches
        self.train_input_batches = self.train_input_batches[:self.batch_size * self.num_train_batches]
        self.train_output_batches = self.train_output_batches[:self.batch_size * self.num_train_batches]

        if self.frame_work == 'Keras':
            return

        if self.frame_work == 'Tensorflow':
            self.train_input_batches, self.train_output_batches = data.convert_to_tensorflow_minbatch(self.train_input_batches, self.train_output_batches, self.batch_size)

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

        if self.frame_work == 'Tensorflow':
            self.val_input_batches, self.val_output_batches = data.convert_to_tensorflow_minbatch(self.val_input_batches, self.val_output_batches, self.batch_size)

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

        if self.frame_work == 'Tensorflow':
            self.eval_input_batches, self.eval_output_batches = data.convert_to_tensorflow_minbatch(self.eval_input_batches, self.eval_output_batches, self.batch_size)

    def reset_all_batches(self):
        self.reset_train_batches()
        self.reset_val_batches()
        self.reset_eval_batches()

    def balance_batch_for_one_hot(self):
        print('Balancing Batches for One Hot Array')
        if self.one_hot_balance_rate is None:
            self.one_hot_balance_rate = 1

        self.load_data()
        self.train_input_data, self.train_output_data = data.balance_batch_for_one_hot(self.train_input_data, self.train_output_data, self.one_hot_balance_samples)

    def clear_memory(self):
        self.train_input_data = None
        self.train_output_data = None
        self.val_input_data = None
        self.val_output_data = None
        self.eval_input_data = None
        self.eval_output_data = None

    def print_num_samples(self):
        self.train_samples = self.train_input_data.shape[0]
        self.val_samples = self.val_input_data.shape[0]
        self.eval_samples = self.eval_input_data.shape[0]

        total_samples = self.train_samples + self.val_samples + self.eval_samples

        print_train_1 = self.train_samples
        print_train_2 = np.round(self.train_samples / total_samples * 100,2)
        print_val_1 = self.val_samples
        print_val_2 = np.round(self.val_samples / total_samples * 100,2)
        print_eval_1 = self.eval_samples
        print_eval_2 = np.round(self.eval_samples / total_samples * 100,2)

        print('Train Samples = {}({}%), Val Samples = {}({}%), Eval Samples = {}({}%)'.format(print_train_1, print_train_2,print_val_1, print_val_2,print_eval_1, print_eval_2))

    def restrict_to_ids(self,ids, column=0):
        print(self.all_output_data.shape)
        self.all_input_data, self.all_output_data = data.restrict_to_ids(self.all_input_data, self.all_output_data, ids, column)
        print(self.all_output_data.shape)

    ## Neural Network Training History
    # Loss History
    def initialize_loss_log(self):
        self.loss_train_log = np.zeros((self.epochs,self.num_train_batches))
        self.loss_val_log = np.zeros(self.epochs)

    def update_loss_train_log(self, loss, epoch_num, batch_num):
        self.loss_train_log[epoch_num][batch_num] = loss

    def update_loss_val_log(self, loss, epoch_num):
        self.loss_val_log[epoch_num] = loss

    # Prediction Accuracy History
    def initialize_acc_log(self):
        self.acc_train_log = np.zeros((self.epochs,self.num_train_batches))
        self.acc_val_log = np.zeros(self.epochs)

    def update_acc_train_log(self, acc, epoch_num, batch_num):
        self.acc_train_log[epoch_num][batch_num] = acc

    def update_acc_val_log(self, acc, epoch_num):
        self.acc_val_log[epoch_num] = acc

    def initialize_all_logs(self):
        self.initialize_loss_log()
        self.initialize_acc_log()

    def save_history_logs(self, log_prefix = ''):
        if self.loss_train_log is not None:
            np.savetxt(self.model_save_folder + log_prefix + 'Train_Loss_Log.csv', self.loss_train_log, delimiter=',', fmt='%1.4f')
            np.savetxt(self.model_save_folder + log_prefix + 'Val_Loss_Log.csv', self.loss_val_log, delimiter=',', fmt='%1.4f')

        if self.acc_train_log is not None:
            np.savetxt(self.model_save_folder + log_prefix + 'Train_Acc_Log.csv', self.acc_train_log, delimiter=',', fmt='%1.4f')
            np.savetxt(self.model_save_folder + log_prefix + 'Val_Acc_Log.csv', self.acc_val_log, delimiter=',', fmt='%1.4f')

    ## Data Augmentation:
    def augment_1D_left_right(self, left=6, right=6, step=1):
        print('Augmenting Data Left and Right. New Samples =')
        self.train_input_data, self.train_output_data  = data.augment_1D_left_right(self.train_input_data, self.train_output_data, left, right, step)
        self.print_num_samples()

    def augment_1D_squeeze_stretch(self, squeeze=0.98, stretch=1.02, steps=3):
        print('Augmenting - Squeeze & Stretch \n Inefficient Implementation, perform before other Augmentation')
        self.train_input_data, self.train_output_data = data.augment_1D_squeeze_stretch(self.train_input_data, self.train_output_data, squeeze, stretch, steps)
        self.print_num_samples()

    def augment_1D_squash_pull(self, squash=0.98, pull=1.02, steps=10, type='multiply'):
        print('Augmenting - Squash & Pull')
        self.train_input_data, self.train_output_data = data.augment_1D_squash_pull(self.train_input_data, self.train_output_data, squash, pull, steps, type)
        self.print_num_samples()

    def augment_add_noise(self, std_dev=0.01):
        print('Augmenting - Adding Gausian Noise')
        self.train_input_data = data.augment_add_noise(self.train_input_data, std_dev)

    def shuffle_training_only(self):
        print('Shuffling Training Data')
        self.train_input_data, self.train_output_data = data.shuffle_input_output(self.train_input_data,self.train_output_data)

    ## Manipulating Data
    def integrate_input_curve(self, col_start=None, col_end=None):
        self.all_output_data = data.integrate_input_curve(self.all_input_data, col_start=col_start, col_end=col_end)

    def dynamic_updating_continuous_mse_loss(self):
        self.all_output_data, self.dyn_mse_shift = data.continuous_mse_loss(self.all_output_data, self.dyn_mse_base_width, self.dyn_mse_power, self.dyn_mse_top_width, self.dyn_mse_offset)

    ## Siamese Network
    def calc_train_siamese_batches(self):
        self.unique_ids = data.calc_unique_ids(self.all_output_data)
        self.siamese_train_input_batches_left, self.siamese_train_input_batches_right, self.siamese_train_output_batches, self.siamese_train_left_idx, self.siamese_train_right_idx = data.calc_siamese_batches(self.train_input_data, self.train_output_data, self.unique_ids, self.num_train_batches, self.batch_size)

    def calc_val_siamese_batches(self):
        self.unique_ids = data.calc_unique_ids(self.all_output_data)
        self.siamese_val_input_left, self.siamese_val_input_right, self.siamese_val_output, self.siamese_val_left_idx, self.siamese_val_right_idx = data.calc_siamese_batches(self.val_input_data, self.val_output_data, self.unique_ids, self.num_val_batches, self.batch_size, reshape=False)

    def calc_eval_siamese_batches(self):
        self.unique_ids = data.calc_unique_ids(self.all_output_data)
        self.siamese_eval_input_left, self.siamese_eval_input_right, self.siamese_eval_output_batches, self.siamese_eval_left_idx, self.siamese_eval_right_idx = data.calc_siamese_batches(self.eval_input_data, self.eval_output_data, self.unique_ids, self.num_eval_batches, self.batch_size, reshape=False)