import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from pylab import *
from DeepLearning.Misc.ProgBar import new_prog_bar, update_prog_bar

import DeepLearning.Tensorflow_Base_Functions.cost as tfCost
import DeepLearning.Tensorflow_Base_Functions.optimizers as tfOptimizers
import DeepLearning.Tensorflow_Base_Functions.evaluation as tfEval


def TransferLearning_train_categorical_network(DataCenter, model, transfer_layers, transfer_layer_types, save = True, min_save_acc = 0, noise = False, log_train_acc = False):

    ''' This function is replicated from Tensorflow_Base_Functions.train to adapt transfer learning

    '''

    cost = tfCost.categorical_cross_entropy(DataCenter, model)
    learning_step, optimizer = tfOptimizers.adam_optimizer_w_lr_decay(DataCenter, cost)
    saver = tf.train.Saver()

    x = DataCenter.x_placeholder
    y = DataCenter.y_placeholder

    epochs = DataCenter.epochs

    prog_bar = new_prog_bar()

    val_x_all = DataCenter.val_input_batches
    val_y_all = DataCenter.val_output_batches

    acc_best = 0

    DataCenter.initialize_all_logs()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Transfer Learning - Loading Weights - NO LSTM's yet")

        ##########_________________###############
        ## Assign Weights For transfer Layers
        for i in range(len(transfer_layers)):
            # Get Layer Type
            type = transfer_layer_types[i]

            with tf.variable_scope(DataCenter.transfer_model_scope + '/' + transfer_layers[i], reuse=True):
                # Convolution or Dense Layers
                if type == 'conv' or type == 'dense':

                    weight = tf.get_variable('kernel')
                    bias = tf.get_variable('bias')

                    weight_name = str(weight.name)
                    bias_name = str(bias.name)

                    weight_name = weight_name[weight_name.find('/'):]
                    bias_name = bias_name[bias_name.find('/'):]

                    weight_pos = np.where(DataCenter.transfer_var_names == weight_name)[0][0]
                    bias_pos = np.where(DataCenter.transfer_var_names == bias_name)[0][0]

                    sess.run(tf.assign(weight, DataCenter.transfer_var_values[weight_pos]))
                    sess.run(tf.assign(bias, DataCenter.transfer_var_values[bias_pos]))

                elif type == 'batchnorm':
                    gamma = tf.get_variable('gamma')
                    beta = tf.get_variable('beta')

                    gamma_name = str(gamma.name)
                    beta_name = str(beta.name)

                    gamma_name = gamma_name[gamma_name.find('/'):]
                    beta_name = beta_name[beta_name.find('/'):]

                    gamma_pos = np.where(DataCenter.transfer_var_names == gamma_name)[0][0]
                    beta_pos = np.where(DataCenter.transfer_var_names == beta_name)[0][0]

                    sess.run(tf.assign(gamma, DataCenter.transfer_var_values[gamma_pos]))
                    sess.run(tf.assign(beta, DataCenter.transfer_var_values[beta_pos]))

                print('Loaded {}% of Weights'.format(np.round(i/len(transfer_layers)*100)))


        # Train Network
        for epoch in range(epochs):

            for batch in range(DataCenter.num_train_batches):

                epoch_x = DataCenter.train_input_batches[batch]
                epoch_y = DataCenter.train_output_batches[batch]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                # Log Training Loss
                DataCenter.update_loss_train_log(c, epoch, batch)

                if log_train_acc == True:
                    train_acc = tfEval.prediction_accuracy(DataCenter, model, DataCenter.train_input_batches, DataCenter.train_output_batches)
                    DataCenter.update_acc_train_log(train_acc, epoch, batch)

                update = batch/DataCenter.num_train_batches
                update_prog_bar(prog_bar, update)

            print('\n Learning rate: %f' % (sess.run(learning_step._lr_t)))

            print('Epoch', epoch, 'completed out of', epochs, 'loss:', c)

            # Calculate Validation Accuracy
            val_acc = tfEval.prediction_accuracy(DataCenter, model, val_x_all, val_y_all)
            print('Validation Accuracy: {}%'.format(np.round(np.mean(val_acc)*100,2)))

            acc_best = val_acc if val_acc > acc_best else acc_best
            print('Best Accuracy: {}%'.format(np.round(np.mean(acc_best) * 100, 2)))

            # Update History Logs:
            DataCenter.update_acc_val_log(val_acc, epoch)
            DataCenter.save_history_logs()

            # Add Noise and Shuffle
            if noise == True:
                DataCenter.augment_add_noise(std_dev=0.0005)
                DataCenter.shuffle_training_only()

            if save is True and val_acc == acc_best and val_acc >= min_save_acc:
                print('Saving Model')
                min_save_acc += 0.01
                saver.save(sess, DataCenter.model_save_folder + DataCenter.model_save_name)

        tfEval.predict_train_val_eval(DataCenter, model)


def extract_variables(DataCenter):

    with tf.Session() as sess:
        # Load Model
        new_saver = tf.train.import_meta_graph(DataCenter.model_load_folder + DataCenter.model_load_name + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(DataCenter.model_load_folder))

        # Get All Variables
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        var_names = []
        var_values = []

        for i in range(np.array(variables).shape[0]):

            variable = variables[i]

            name = variable.name
            value = sess.run(variable)

            var_names.append(name[name.find('/'):])
            var_values.append(value)

    sess.close()
    tf.reset_default_graph()

    DataCenter.transfer_var_names = np.array(var_names)
    DataCenter.transfer_var_values = var_values

    return
