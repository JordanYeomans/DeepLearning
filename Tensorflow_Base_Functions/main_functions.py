import tensorflow as tf
import numpy as np
import time

from DeepLearning.Misc.ProgBar import new_prog_bar, update_prog_bar


def initialise_placeholders(DataCenter):

    dims = len(DataCenter.train_input_data.shape)-1

    dim_1 = DataCenter.train_input_data.shape[1]

    if dims == 1:
        x = tf.placeholder('float', [None, dim_1], name='Input')

    if dims == 2:
        dim_2 = DataCenter.train_input_data.shape[2]
        x = tf.placeholder('float', [None, dim_1, dim_2], name='Input')

    if dims == 3:
        dim_2 = DataCenter.train_input_data.shape[2]
        dim_3 = DataCenter.train_input_data.shape[3]
        x = tf.placeholder('float', [None, dim_1, dim_2, dim_3], name='Input')

    y = tf.placeholder('float', name='Output')

    DataCenter.x_placeholder = x
    DataCenter.y_placeholder = y

    return

def train_categorical_network(DataCenter, model, cost, optimizer):

    x = DataCenter.x_placeholder
    y = DataCenter.y_placeholder
    epochs = DataCenter.epochs

    prog_bar = new_prog_bar()

    val_x_all = DataCenter.val_input_batches
    val_y_all = DataCenter.val_output_batches

    acc_rec = np.zeros(val_x_all.shape[0])

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):

            for batch in range(DataCenter.num_train_batches):

                epoch_x = np.array(DataCenter.train_input_batches[batch])
                epoch_y = DataCenter.train_output_batches[batch]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss = c

                update = batch/DataCenter.num_train_batches

                update_prog_bar(prog_bar, update)

            print('\n Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            for i in range(val_x_all.shape[0]):

                val_x = val_x_all[i]
                val_y = val_y_all[i]

                acc_rec[i] = accuracy.eval({x: val_x, y: val_y})

            print('Validation Accuracy: {}%'.format(np.round(np.mean(acc_rec)*100,2)))