import tensorflow as tf
import time
import numpy as np

import DeepLearning.Tensorflow_Base_Functions.initialization as tfInitial

def Conv_Dense_0001_a(DataCenter):

    tfInitial.initialize_placeholders(DataCenter)

    scope = DataCenter.new_model_scope
    last_layer = DataCenter.train_output_data.shape[1]
    with tf.variable_scope(scope):
        x = DataCenter.x_placeholder

        nn = tf.layers.conv1d(x, filters=64, kernel_size=5, name='conv1_1')
        nn = tf.layers.conv1d(nn, filters=64, kernel_size=5, name='conv1_2')
        nn = tf.layers.conv1d(nn, filters=64, kernel_size=5, name='conv1_3')
        nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2, name = 'max_pool1')
        nn = tf.layers.batch_normalization(nn, name='batchnorm_1')

        nn = tf.layers.conv1d(nn, filters=128, kernel_size=5, activation=tf.nn.relu, name='conv2_1')
        nn = tf.layers.conv1d(nn, filters=128, kernel_size=5, activation=tf.nn.relu, name='conv2_2')
        nn = tf.layers.conv1d(nn, filters=128, kernel_size=5, activation=tf.nn.relu, name='conv2_3')
        nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2, name = 'max_pool2')
        nn = tf.layers.batch_normalization(nn, name='batchnorm_2')

        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu, name='conv3_1')
        nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2, name = 'max_pool3')
        nn = tf.layers.batch_normalization(nn, name='batchnorm_3')

        nn = tf.layers.flatten(nn, name='flatten_1')

        nn = tf.layers.dense(nn, 512, activation=tf.nn.relu, name='dense_1')

        nn = tf.layers.dense(nn, last_layer, activation=tf.nn.sigmoid, name='dense_3')

    return nn

