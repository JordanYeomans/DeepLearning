import tensorflow as tf
import time
import numpy as np

import DeepLearning.Tensorflow_Base_Functions.initialization as tfInitial

def Conv_Dense_0001_a(DataCenter):

    tfInitial.initialize_placeholders(DataCenter)

    scope = DataCenter.transfer_model_scope

    with tf.variable_scope(scope):
        x = DataCenter.x_placeholder

        nn = tf.layers.conv1d(x, filters=64, kernel_size=5, activation=tf.nn.relu, name='conv1', trainable=False)

        nn = tf.layers.max_pooling1d(nn,pool_size=4, strides=2, name = 'max_pool1')

        nn = tf.layers.conv1d(nn, filters=128, kernel_size=5, activation=tf.nn.relu, name='conv2', trainable=False)
        nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2, name = 'max_pool2')

        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu, name='conv3', trainable=False)
        nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2, name = 'max_pool3')

        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu, name='conv4', trainable=False)
        nn = tf.layers.flatten(nn, name='flatten_1')

        nn = tf.layers.dense(nn, 128, activation=tf.nn.relu, name='dense_1')
        nn = tf.layers.dense(nn, 128, activation=tf.nn.relu, name='dense_2')

        nn = tf.layers.dense(nn, 30, activation=tf.nn.sigmoid, name='dense_3')

    return nn