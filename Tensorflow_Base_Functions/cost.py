import tensorflow as tf

def categorical_cross_entropy(DataCenter, model):
    'Cost function for categorical cross entropy'
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=DataCenter.y_placeholder))
