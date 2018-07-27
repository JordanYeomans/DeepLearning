import tensorflow as tf
import time
import numpy as np

import DeepLearning.Tensorflow_Base_Functions.initialization as tfInitial

def Siamese_Conv_Dense_0001_a(DataCenter):

    def model(x, scope, reuse):
        with tf.variable_scope(scope):

            nn = tf.layers.conv1d(x, filters=64, kernel_size=15, name='conv1_1', reuse=reuse)
            nn = tf.layers.conv1d(nn, filters=64, kernel_size=15, name='conv1_2', reuse=reuse)
            nn = tf.layers.conv1d(nn, filters=64, kernel_size=15, name='conv1_3', reuse=reuse)

            nn = tf.layers.flatten(nn)

            nn = tf.layers.dense(nn, 64, activation=tf.nn.relu, name='dense_1', reuse=reuse)
            nn = tf.layers.dense(nn, 64, activation=tf.nn.relu, name='dense_2', reuse=reuse)
            nn = tf.layers.dense(nn, 64, activation=tf.nn.relu, name='dense_3', reuse=reuse)

            nn = tf.layers.dense(nn, 1, name='output', reuse=reuse)

        return nn

    tfInitial.initialize_siamese_placeholders(DataCenter)

    scope = DataCenter.new_model_scope

    model_left = model(DataCenter.x_left_placeholder, scope,  reuse=False)
    model_right = model(DataCenter.x_right_placeholder, scope, reuse=True)

    return model_left, model_right
