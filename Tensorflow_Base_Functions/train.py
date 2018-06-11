import tensorflow as tf
import numpy as np
import time

from DeepLearning.Misc.ProgBar import new_prog_bar, update_prog_bar

import DeepLearning.Tensorflow_Base_Functions.cost as tfCost
import DeepLearning.Tensorflow_Base_Functions.optimizers as tfOptimizers
import DeepLearning.Tensorflow_Base_Functions.evaluation as tfEval

import matplotlib.pyplot as plt

def train_categorical_network(DataCenter, model, save = True, min_save_acc = 0):

    cost = tfCost.categorical_cross_entropy(DataCenter, model)
    optimizer = tfOptimizers.adam_optimizer(DataCenter, cost)
    saver = tf.train.Saver()

    x = DataCenter.x_placeholder
    y = DataCenter.y_placeholder

    epochs = DataCenter.epochs

    prog_bar = new_prog_bar()

    val_x_all = DataCenter.val_input_batches
    val_y_all = DataCenter.val_output_batches

    acc_best = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for batch in range(DataCenter.num_train_batches):

                epoch_x = DataCenter.train_input_batches[batch]
                epoch_y = DataCenter.train_output_batches[batch]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                update = batch/DataCenter.num_train_batches
                update_prog_bar(prog_bar, update)

            print('\n Epoch', epoch, 'completed out of', epochs, 'loss:', c)

            # Calculate Validation Accuracy
            acc = tfEval.prediction_accuracy(DataCenter, model, val_x_all, val_y_all)
            print('Validation Accuracy: {}%'.format(np.round(np.mean(acc)*100,2)))

            acc_best = acc if acc > acc_best else acc_best

            if save is True and acc == acc_best and acc > min_save_acc:
                print('Saving Model')
                saver.save(sess, DataCenter.model_save_folder + DataCenter.model_save_name)

        tfEval.predict_train_val_eval(DataCenter, model)
