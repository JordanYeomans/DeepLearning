import tensorflow as tf

def adam_optimizer(DataCenter, cost):

    return tf.train.AdamOptimizer(DataCenter.learning_rate).minimize(cost)