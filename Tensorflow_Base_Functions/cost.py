import tensorflow as tf

def categorical_cross_entropy(DataCenter, model):
    'Cost function for categorical cross entropy'
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=DataCenter.y_placeholder))

def mean_squared_error(DataCenter, model):
    return tf.losses.mean_squared_error(predictions=model, labels=DataCenter.y_placeholder)

def siamese_loss(DataCenter, nn_left, nn_right):

    label = DataCenter.y_placeholder
    lr = DataCenter.learning_rate
    margin = 1

    d = tf.reduce_sum(tf.square(tf.subtract(nn_left, nn_right)), 1)
    dw = tf.sqrt(d + 1e-6)

    loss_eq = (0.5 * (1 - label) * tf.square(dw)) + (0.5 * label * tf.square(tf.maximum(0., margin - dw)))
    loss = tf.reduce_mean(loss_eq)

    optimiser = tf.train.AdamOptimizer(lr).minimize(loss)

    return optimiser, loss, loss_eq, d, dw
