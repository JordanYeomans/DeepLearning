import tensorflow as tf
import time
import numpy as np

def three_dense_layers(DataCenter):

    x = DataCenter.x_placeholder

    nn = tf.layers.conv1d(x, filters=64, kernel_size=5, activation=tf.nn.relu)
    nn = tf.layers.max_pooling1d(nn,pool_size=4, strides=2)

    nn = tf.layers.conv1d(nn, filters=128, kernel_size=5, activation=tf.nn.relu)
    nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2)

    nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu)
    nn = tf.layers.max_pooling1d(nn, pool_size=4, strides=2)

    nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu)
    nn = tf.layers.flatten(nn)

    nn = tf.layers.dense(nn, 4096, activation=tf.nn.relu)
    nn = tf.layers.dense(nn, 4096, activation=tf.nn.relu)


    nn = tf.layers.dense(nn, 30, activation=tf.nn.sigmoid)

    return nn

def Conv_LSTM_Dense_0001_a():

    return