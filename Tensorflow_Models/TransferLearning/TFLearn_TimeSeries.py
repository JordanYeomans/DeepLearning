import tensorflow as tf
import time
import numpy as np

import DeepLearning.Tensorflow_Base_Functions.initialization as tfInitial

def Conv_Dense_0001_a(DataCenter, trainable = False):

    tfInitial.initialize_placeholders(DataCenter)

    if trainable == False:
        print('Trainable is set to False for Transfer Layers')

    scope = DataCenter.transfer_model_scope

    with tf.variable_scope(scope):
        x = DataCenter.x_placeholder

        nn = tf.layers.conv1d(x, filters=64, kernel_size=5, name='conv1_1', trainable=trainable)
        nn = tf.layers.conv1d(nn, filters=64, kernel_size=5, name='conv1_2', trainable=trainable)
        nn = tf.layers.conv1d(nn, filters=64, kernel_size=5, name='conv1_3', trainable=trainable)
        nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2, name = 'max_pool1')
        nn = tf.layers.batch_normalization(nn, name='batchnorm_1', trainable=trainable)

        nn = tf.layers.conv1d(nn, filters=128, kernel_size=5, activation=tf.nn.relu, name='conv2_1', trainable=trainable)
        nn = tf.layers.conv1d(nn, filters=128, kernel_size=5, activation=tf.nn.relu, name='conv2_2', trainable=trainable)
        nn = tf.layers.conv1d(nn, filters=128, kernel_size=5, activation=tf.nn.relu, name='conv2_3', trainable=trainable)
        nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2, name = 'max_pool2')
        nn = tf.layers.batch_normalization(nn, name='batchnorm_2', trainable=trainable)

        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu, name='conv3_1', trainable=trainable)
        nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2, name = 'max_pool3')
        nn = tf.layers.batch_normalization(nn, name='batchnorm_3', trainable=trainable)

        # New Layers
        nn = tf.layers.dropout(nn, 0.5)

        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu)
        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu)
        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu)
        nn = tf.layers.batch_normalization(nn)

        nn = tf.layers.flatten(nn, name='flatten_1')

        nn = tf.layers.dropout(nn, rate=0.5)
        nn = tf.layers.batch_normalization(nn)

        nn = tf.layers.dense(nn, 4096, activation=tf.nn.relu)
        nn = tf.layers.dropout(nn, rate=0.5)
        nn = tf.layers.batch_normalization(nn)

        nn = tf.layers.dense(nn, 4096, activation=tf.nn.relu)
        nn = tf.layers.dropout(nn, rate=0.5)
        nn = tf.layers.batch_normalization(nn)

        # nn = tf.layers.dense(nn, 512, activation=tf.nn.relu, name='dense_3')
        # nn = tf.layers.dropout(nn, rate=0.5)

        nn = tf.layers.dense(nn, 26, activation=tf.nn.sigmoid)

    return nn
