import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pylab import *
from DeepLearning.Misc.ProgBar import new_prog_bar, update_prog_bar

import DeepLearning.Tensorflow_Base_Functions.cost as tfCost
import DeepLearning.Tensorflow_Base_Functions.optimizers as tfOptimizers
import DeepLearning.Tensorflow_Base_Functions.evaluation as tfEval


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


def TransferLearning_train_categorical_network(DataCenter, model, transfer_layers, transfer_layer_types, save=True, min_save_acc=0, noise=False, log_train_acc=False, balance_one_hot=True):
    ''' This function is replicated from Tensorflow_Base_Functions.train to adapt transfer learning
    '''

    if DataCenter.set_cost_function == 'categorical_cross_entropy':
        cost = tfCost.categorical_cross_entropy(DataCenter, model)

    elif DataCenter.set_cost_function == 'mse':
        cost = tfCost.mean_squared_error(DataCenter, model)

    learning_step, optimizer = tfOptimizers.adam_optimizer_w_lr_decay(DataCenter, cost)
    saver = tf.train.Saver()

    x = DataCenter.x_placeholder
    y = DataCenter.y_placeholder

    epochs = DataCenter.epochs
    prog_bar = new_prog_bar()
    acc_best = 0
    val_mse_best = 10000
    save_state = False

    early_stopping = DataCenter.early_stopping

    DataCenter.initialize_all_logs()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Transfer Learning - Loading Weights - NO LSTM's yet")
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

            # Epochs Between Validation Checks
            if (epoch+1) % DataCenter.recalc_eval == 0:

                # Evaluation Validation Accuracy
                if DataCenter.eval_metric == 'best_val_accuracy':
                    val_acc = tfEval.prediction_accuracy(DataCenter, model, DataCenter.val_input_batches, DataCenter.val_output_batches)
                    acc_best = val_acc if val_acc > acc_best else acc_best
                    # Update History Logs:
                    DataCenter.update_acc_val_log(val_acc, epoch)
                    DataCenter.save_history_logs()
                    print('Validation Accuracy: {}%'.format(np.round(np.mean(val_acc) * 100, 2)))
                    print('Best Accuracy: {}%'.format(np.round(np.mean(acc_best) * 100, 2)))

                    if val_acc == acc_best and val_acc >= min_save_acc:
                        save_state = True
                        min_save_acc = val_acc + DataCenter.save_network_increment  # Increment Save accuracy (Default 0.01)
                    # Update Save State

                elif DataCenter.eval_metric == 'mse':
                    val_mse = tfEval.mse(DataCenter, tfCost.mean_squared_error(DataCenter, model))
                    val_mse_best = val_mse if val_mse < val_mse_best else val_mse_best
                    if val_mse == val_mse_best and val_mse <= DataCenter.max_save_mse:
                        save_state = True
                        DataCenter.max_save_mse = val_mse - DataCenter.save_network_increment  # Increment Save accuracy (Default 0.01)
                        print(DataCenter.max_save_mse)
                        early_stopping = DataCenter.early_stopping
                    else:
                        early_stopping -= 1

                    print('Validation MSE: {}'.format(np.round(val_mse, 8)))
                    print('Best MSE: {}'.format(np.round(val_mse_best, 8)))

            # Save Model
            if save is True and save_state is True:
                print('Saving Model Based on {}'.format(DataCenter.eval_metric))
                saver.save(sess, DataCenter.model_save_folder + DataCenter.model_save_name)
                save_state = False

            # Add Noise and Shuffle
            if noise == True:
                DataCenter.augment_add_noise(std_dev=0.0005)
                DataCenter.shuffle_training_only()

            # Balance data for 1 hot array
            if balance_one_hot is True and epoch % DataCenter.one_hot_balance_rate == 0:
                DataCenter.balance_batch_for_one_hot()
                DataCenter.reset_train_batches()

            # Check for Early Stopping
            if early_stopping == 0:
                print('Stopping Early')
                break
    return