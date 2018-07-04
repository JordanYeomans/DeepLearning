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

        nn = tf.layers.dense(nn, last_layer, activation=tf.nn.softmax, name='dense_3')

    return nn

def Conv_Dense_0001_b(DataCenter):

    tfInitial.initialize_placeholders(DataCenter)

    scope = DataCenter.new_model_scope
    last_layer = DataCenter.train_output_data.shape[1]

    activ = tf.nn.relu

    with tf.variable_scope(scope):
        x = DataCenter.x_placeholder

        conv1_7_7 = tf.layers.conv1d(x, 64, 7, strides=2, padding='same', activation=activ, name='conv1_7_7_s2')

        pool1_3_3 = tf.layers.max_pooling1d(conv1_7_7, 3, strides=2)
        pool1_3_3 = tf.layers.batch_normalization(pool1_3_3, name = 'batchnorm_1')

        conv2_3_3_reduce = tf.layers.conv1d(pool1_3_3, 64, 1, padding='same', activation=activ, name='conv2_3_3_reduce')
        conv2_3_3 = tf.layers.conv1d(conv2_3_3_reduce, 192, 3, padding='same', activation=activ, name='conv2_3_3')
        conv2_3_3 = tf.layers.batch_normalization(conv2_3_3)

        pool2_3_3 = tf.layers.max_pooling1d(conv2_3_3, pool_size=3, strides=2, name='pool2_3_3_s2')

        inception_3a_1_1 = tf.layers.conv1d(pool2_3_3, 64, 1, padding='same', activation=activ, name='inception_3a_1_1')

        inception_3a_3_3_reduce = tf.layers.conv1d(pool2_3_3, 96, 1, padding='same', activation=activ, name='inception_3a_3_3_reduce')
        inception_3a_3_3 = tf.layers.conv1d(inception_3a_3_3_reduce, 128, padding='same', kernel_size = 3, activation=activ, name='inception_3a_3_3')

        inception_3a_5_5_reduce = tf.layers.conv1d(pool2_3_3, 16, padding='same', kernel_size=1, activation=activ, name='inception_3a_5_5_reduce')
        inception_3a_5_5 = tf.layers.conv1d(inception_3a_5_5_reduce, 32, padding='same', kernel_size=5, activation=activ, name='inception_3a_5_5')

        inception_3a_pool = tf.layers.max_pooling1d(pool2_3_3, pool_size=3, strides=1, padding='same')
        inception_3a_pool_1_1 = tf.layers.conv1d(inception_3a_pool, 32, kernel_size=1, padding='same', activation=activ, name='inception_3a_pool_1_1')

        # merge the inception_3a__
        inception_3a_output = tf.concat([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], axis=2)

        inception_3b_1_1 = tf.layers.conv1d(inception_3a_output, 128, kernel_size=1, padding='same', activation=activ, name='inception_3b_1_1')
        inception_3b_3_3_reduce = tf.layers.conv1d(inception_3a_output, 128, kernel_size=1, padding='same', activation=activ, name='inception_3b_3_3_reduce')
        inception_3b_3_3 = tf.layers.conv1d(inception_3b_3_3_reduce, 192, kernel_size=3, padding='same', activation=activ, name='inception_3b_3_3')
        inception_3b_5_5_reduce = tf.layers.conv1d(inception_3a_output, 32, kernel_size=1, padding='same', activation=activ, name='inception_3b_5_5_reduce')
        inception_3b_5_5 = tf.layers.conv1d(inception_3b_5_5_reduce, 96, kernel_size=5, padding='same', activation=activ, name='inception_3b_5_5')
        inception_3b_pool = tf.layers.max_pooling1d(inception_3a_output, pool_size=3, strides=1, padding='same', name='inception_3b_pool')
        inception_3b_pool_1_1 = tf.layers.conv1d(inception_3b_pool, 64, kernel_size=1, padding='same', activation=activ, name='inception_3b_pool_1_1')

        # merge the inception_3b_*
        inception_3b_output = tf.concat([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], axis=2, name='inception_3b_output')

        pool3_3_3 = tf.layers.max_pooling1d(inception_3b_output, pool_size=3, strides=2, name='pool3_3_3')

        inception_4a_1_1 = tf.layers.conv1d(pool3_3_3, 192, kernel_size=1, padding='same', activation=activ, name='inception_4a_1_1')
        inception_4a_3_3_reduce = tf.layers.conv1d(pool3_3_3, 96, kernel_size=1, padding='same', activation=activ, name='inception_4a_3_3_reduce')
        inception_4a_3_3 = tf.layers.conv1d(inception_4a_3_3_reduce, 208, kernel_size=3, padding='same', activation=activ, name='inception_4a_3_3')
        inception_4a_5_5_reduce = tf.layers.conv1d(pool3_3_3, 16, kernel_size=1, padding='same', activation=activ, name='inception_4a_5_5_reduce')
        inception_4a_5_5 = tf.layers.conv1d(inception_4a_5_5_reduce, 48, kernel_size=5, padding='same', activation=activ, name='inception_4a_5_5')
        inception_4a_pool = tf.layers.max_pooling1d(pool3_3_3, pool_size=3, strides=1, padding='same', name='inception_4a_pool')
        inception_4a_pool_1_1 = tf.layers.conv1d(inception_4a_pool, 64, kernel_size=1, padding='same', activation=activ, name='inception_4a_pool_1_1')

        inception_4a_output = tf.concat([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], axis=2, name='inception_4a_output')

        inception_4b_1_1 = tf.layers.conv1d(inception_4a_output, 160, kernel_size=1, padding='same', activation=activ, name='inception_4b_1_1')
        inception_4b_3_3_reduce = tf.layers.conv1d(inception_4a_output, 112, kernel_size=1, padding='same', activation=activ, name='inception_4b_3_3_reduce')
        inception_4b_3_3 = tf.layers.conv1d(inception_4b_3_3_reduce, 224, kernel_size=3, padding='same', activation=activ, name='inception_4b_3_3')
        inception_4b_5_5_reduce = tf.layers.conv1d(inception_4a_output, 24, kernel_size=1, padding='same', activation=activ, name='inception_4b_5_5_reduce')
        inception_4b_5_5 = tf.layers.conv1d(inception_4b_5_5_reduce, 64, kernel_size=5, padding='same', activation=activ, name='inception_4b_5_5')

        inception_4b_pool = tf.layers.max_pooling1d(inception_4a_output, pool_size=3, strides=1, padding='same', name='inception_4b_pool')
        inception_4b_pool_1_1 = tf.layers.conv1d(inception_4b_pool, 64, kernel_size=1, padding='same', activation=activ, name='inception_4b_pool_1_1')

        inception_4b_output = tf.concat([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], axis=2, name='inception_4b_output')

        inception_4c_1_1 = tf.layers.conv1d(inception_4b_output, 128, kernel_size=1, padding='same', activation=activ, name='inception_4c_1_1')
        inception_4c_3_3_reduce = tf.layers.conv1d(inception_4b_output, 128, kernel_size=1, padding='same', activation=activ, name='inception_4c_3_3_reduce')
        inception_4c_3_3 = tf.layers.conv1d(inception_4c_3_3_reduce, 256, kernel_size=3, padding='same', activation=activ, name='inception_4c_3_3')
        inception_4c_5_5_reduce = tf.layers.conv1d(inception_4b_output, 24, kernel_size=1, padding='same', activation=activ, name='inception_4c_5_5_reduce')
        inception_4c_5_5 = tf.layers.conv1d(inception_4c_5_5_reduce, 64, kernel_size=5, padding='same', activation=activ, name='inception_4c_5_5')

        inception_4c_pool = tf.layers.max_pooling1d(inception_4b_output, pool_size=3, strides=1, padding='same')
        inception_4c_pool_1_1 = tf.layers.conv1d(inception_4c_pool, 64, kernel_size=1, padding='same', activation=activ, name='inception_4c_pool_1_1')

        inception_4c_output = tf.concat([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], axis=2, name='inception_4c_output')

        inception_4d_1_1 = tf.layers.conv1d(inception_4c_output, 112, kernel_size=1, padding='same', activation=activ, name='inception_4d_1_1')
        inception_4d_3_3_reduce = tf.layers.conv1d(inception_4c_output, 144, kernel_size=1, padding='same', activation=activ, name='inception_4d_3_3_reduce')
        inception_4d_3_3 = tf.layers.conv1d(inception_4d_3_3_reduce, 288, kernel_size=3, padding='same', activation=activ, name='inception_4d_3_3')
        inception_4d_5_5_reduce = tf.layers.conv1d(inception_4c_output, 32, kernel_size=1, padding='same', activation=activ, name='inception_4d_5_5_reduce')
        inception_4d_5_5 = tf.layers.conv1d(inception_4d_5_5_reduce, 64, kernel_size=5, padding='same', activation=activ, name='inception_4d_5_5')
        inception_4d_pool = tf.layers.max_pooling1d(inception_4c_output, pool_size=3, strides=1, padding='same', name='inception_4d_pool')
        inception_4d_pool_1_1 = tf.layers.conv1d(inception_4d_pool, 64, kernel_size=1, padding='same', activation=activ, name='inception_4d_pool_1_1')

        inception_4d_output = tf.concat([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], axis=2, name='inception_4d_output')

        inception_4e_1_1 = tf.layers.conv1d(inception_4d_output, 256, kernel_size=1, padding='same', activation=activ, name='inception_4e_1_1')
        inception_4e_3_3_reduce = tf.layers.conv1d(inception_4d_output, 160, kernel_size=1, padding='same', activation=activ, name='inception_4e_3_3_reduce')
        inception_4e_3_3 = tf.layers.conv1d(inception_4e_3_3_reduce, 320, kernel_size=3, padding='same', activation=activ, name='inception_4e_3_3')
        inception_4e_5_5_reduce = tf.layers.conv1d(inception_4d_output, 32, kernel_size=1, padding='same', activation=activ, name='inception_4e_5_5_reduce')
        inception_4e_5_5 = tf.layers.conv1d(inception_4e_5_5_reduce, 128, kernel_size=5, padding='same', activation=activ, name='inception_4e_5_5')
        inception_4e_pool = tf.layers.max_pooling1d(inception_4d_output, pool_size=3, strides=1, padding='same', name='inception_4e_pool')
        inception_4e_pool_1_1 = tf.layers.conv1d(inception_4e_pool, 128, kernel_size=1, padding='same', activation=activ, name='inception_4e_pool_1_1')

        inception_4e_output = tf.concat([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=2)
        pool4_3_3 = tf.layers.max_pooling1d(inception_4e_output, pool_size=3, strides=2, name='pool_3_3')

        inception_5a_1_1 = tf.layers.conv1d(pool4_3_3, 256, kernel_size=1, padding='same', activation=activ, name='inception_5a_1_1')
        inception_5a_3_3_reduce = tf.layers.conv1d(pool4_3_3, 160, kernel_size=1, padding='same', activation=activ, name='inception_5a_3_3_reduce')
        inception_5a_3_3 = tf.layers.conv1d(inception_5a_3_3_reduce, 320, kernel_size=3, padding='same', activation=activ, name='inception_5a_3_3')
        inception_5a_5_5_reduce = tf.layers.conv1d(pool4_3_3, 32, kernel_size=1, padding='same', activation=activ, name='inception_5a_5_5_reduce')
        inception_5a_5_5 = tf.layers.conv1d(inception_5a_5_5_reduce, 128, kernel_size=5, padding='same', activation=activ, name='inception_5a_5_5')
        inception_5a_pool = tf.layers.max_pooling1d(pool4_3_3, pool_size=3, strides=1, padding='same', name='inception_5a_pool')
        inception_5a_pool_1_1 = tf.layers.conv1d(inception_5a_pool, 128, kernel_size=1, padding='same', activation=activ, name='inception_5a_pool_1_1')

        inception_5a_output = tf.concat([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=2)

        inception_5b_1_1 = tf.layers.conv1d(inception_5a_output, 384, kernel_size=1, padding='same', activation=activ, name='inception_5b_1_1')
        inception_5b_3_3_reduce = tf.layers.conv1d(inception_5a_output, 192, kernel_size=1, padding='same', activation=activ, name='inception_5b_3_3_reduce')
        inception_5b_3_3 = tf.layers.conv1d(inception_5b_3_3_reduce, 384, kernel_size=3, padding='same', activation=activ, name='inception_5b_3_3')
        inception_5b_5_5_reduce = tf.layers.conv1d(inception_5a_output, 48, kernel_size=1, padding='same', activation=activ, name='inception_5b_5_5_reduce')
        inception_5b_5_5 = tf.layers.conv1d(inception_5b_5_5_reduce, 128, kernel_size=5, padding='same', activation=activ, name='inception_5b_5_5')
        inception_5b_pool = tf.layers.max_pooling1d(inception_5a_output, pool_size=3, strides=1, padding='same', name='inception_5b_pool')
        inception_5b_pool_1_1 = tf.layers.conv1d(inception_5b_pool, 128, kernel_size=1, padding='same', activation=activ, name='inception_5b_pool_1_1')
        inception_5b_output = tf.concat([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=2)

        pool5_7_7 = tf.layers.max_pooling1d(inception_5b_output, pool_size=7, strides=1)

        nn = tf.layers.flatten(pool5_7_7)

        nn = tf.layers.dense(nn, 1024)
        nn = tf.layers.dense(nn, 1024)

        nn = tf.layers.dense(nn, last_layer, activation=tf.nn.softmax, name='output')

    return nn

def Conv_Dense_0001_c(DataCenter):

    tfInitial.initialize_placeholders(DataCenter)

    scope = DataCenter.new_model_scope
    last_layer = DataCenter.train_output_data.shape[1]
    with tf.variable_scope(scope):
        x = DataCenter.x_placeholder

        nn = tf.layers.conv1d(x, filters=64, kernel_size=9, name='conv1_1')
        nn = tf.layers.conv1d(nn, filters=64, kernel_size=9, name='conv1_2')
        nn = tf.layers.conv1d(nn, filters=64, kernel_size=9, name='conv1_3')
        nn = tf.layers.max_pooling1d(nn, pool_size=2, strides=2, name='max_pool1')
        nn = tf.layers.batch_normalization(nn, name='batchnorm_1')

        nn = tf.layers.conv1d(nn, filters=128, kernel_size=7, activation=tf.nn.relu, name='conv2_1')
        nn = tf.layers.conv1d(nn, filters=128, kernel_size=7, activation=tf.nn.relu, name='conv2_2')
        nn = tf.layers.conv1d(nn, filters=128, kernel_size=7, activation=tf.nn.relu, name='conv2_3')
        nn = tf.layers.max_pooling1d(nn, pool_size=2, strides=2, name='max_pool2')
        nn = tf.layers.batch_normalization(nn, name='batchnorm_2')

        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu, name='conv3_1')
        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu, name='conv3_2')
        nn = tf.layers.conv1d(nn, filters=256, kernel_size=5, activation=tf.nn.relu, name='conv3_3')
        nn = tf.layers.max_pooling1d(nn, pool_size=2, strides=2, name='max_pool3')
        nn = tf.layers.batch_normalization(nn, name='batchnorm_3')

        nn = tf.layers.flatten(nn, name='flatten_1')

        nn = tf.layers.dropout(nn, rate=0.5)
        nn = tf.layers.batch_normalization(nn, name='batchnorm_6')

        nn = tf.layers.dense(nn, 1024, activation=tf.nn.relu, name='dense_1')
        nn = tf.layers.dropout(nn, 0.3)

        nn = tf.layers.dense(nn, 1024, activation=tf.nn.relu, name='dense_2')
        nn = tf.layers.dropout(nn, 0.3)

        nn = tf.layers.dense(nn, 1024, activation=tf.nn.relu, name='dense_3')
        nn = tf.layers.dropout(nn, 0.3)

        nn = tf.layers.dense(nn, last_layer, activation=tf.nn.softmax, name='output')

    return nn