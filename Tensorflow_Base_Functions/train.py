import tensorflow as tf
import numpy as np
import time

from DeepLearning.Misc.ProgBar import new_prog_bar, update_prog_bar

import DeepLearning.Tensorflow_Base_Functions.cost as tfCost
import DeepLearning.Tensorflow_Base_Functions.optimizers as tfOptimizers
import DeepLearning.Tensorflow_Base_Functions.evaluation as tfEval

import matplotlib.pyplot as plt

def train_categorical_network(DataCenter, model, save = True, load_model = False, min_save_acc = 0):

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

        if load_model == True:
            new_saver = tf.train.import_meta_graph(DataCenter.model_load_folder + DataCenter.model_load_name + '.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(DataCenter.model_load_folder))

        # Print Data Information
        print('Data being used: {}'.format(DataCenter.folder_path + DataCenter.file_prefix))
        print('Total Training Samples {}'.format(DataCenter.train_samples))
        print('Total Validation Samples {}'.format(DataCenter.val_samples))

        for epoch in range(epochs):
            for batch in range(DataCenter.num_train_batches):

                epoch_x = DataCenter.train_input_batches[batch]
                epoch_y = DataCenter.train_output_batches[batch]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                # Log Training Loss
                DataCenter.update_loss_train_log(c, epoch, batch)

                update = batch/DataCenter.num_train_batches
                update_prog_bar(prog_bar, update)
            print('\n Learning rate: %f' % (sess.run(learning_step._lr_t)))
            print('\n Epoch', epoch, 'completed out of', epochs, 'loss:', c)

            # Calculate Validation Accuracy
            val_acc = tfEval.prediction_accuracy(DataCenter, model, val_x_all, val_y_all)
            print('Validation Accuracy: {}%'.format(np.round(np.mean(val_acc)*100, 2)))

            acc_best = val_acc if val_acc > acc_best else acc_best
            print('Best Accuracy: {}%'.format(np.round(np.mean(acc_best) * 100, 2)))

            # Update History Logs:
            DataCenter.update_acc_val_log(val_acc, epoch)
            DataCenter.save_history_logs()

            # Save Model
            if save is True and val_acc == acc_best and val_acc > min_save_acc:
                print('Saving Model')
                min_save_acc += 0.01
                saver.save(sess, DataCenter.model_save_folder + DataCenter.model_save_name)

    return