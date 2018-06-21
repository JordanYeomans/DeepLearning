import tensorflow as tf

def adam_optimizer(DataCenter, cost):
    return tf.train.AdamOptimizer(DataCenter.learning_rate).minimize(cost)

def adam_optimizer_w_lr_decay(DataCenter, cost):

    global_step = tf.Variable(0, trainable=False)

    lr = tf.train.exponential_decay(DataCenter.lr_start, global_step, DataCenter.num_train_batches * DataCenter.lr_step_rate, DataCenter.lr_decay_rate,
                                    staircase=True)

    learning_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.7, beta2=0.3, name='Adam-op')
    optimizer = learning_step.minimize(cost, global_step=global_step)

    return learning_step, optimizer