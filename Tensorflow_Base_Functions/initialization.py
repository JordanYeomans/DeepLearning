import tensorflow as tf

def initialize_placeholders(DataCenter):

    dims = len(DataCenter.train_input_data.shape)-1

    dim_1 = DataCenter.train_input_data.shape[1]

    if dims == 1:
        x = tf.placeholder('float', [None, dim_1], name='Input')

    if dims == 2:
        dim_2 = DataCenter.train_input_data.shape[2]
        x = tf.placeholder('float', [None, dim_1, dim_2], name='Input')

    if dims == 3:
        dim_2 = DataCenter.train_input_data.shape[2]
        dim_3 = DataCenter.train_input_data.shape[3]
        x = tf.placeholder('float', [None, dim_1, dim_2, dim_3], name='Input')

    y = tf.placeholder('float', name='Output')

    DataCenter.x_placeholder = x
    DataCenter.y_placeholder = y

    return

def initialize_predict_placeholders(DataCenter):

    dims = len(DataCenter.predict_input_data.shape)-1

    dim_1 = DataCenter.predict_input_data.shape[1]

    if dims == 1:
        x = tf.placeholder('float', [None, dim_1], name='Input')

    if dims == 2:
        dim_2 = DataCenter.predict_input_data.shape[2]
        x = tf.placeholder('float', [None, dim_1, dim_2], name='Input')

    if dims == 3:
        dim_2 = DataCenter.predict_input_data.shape[2]
        dim_3 = DataCenter.predict_input_data.shape[3]
        x = tf.placeholder('float', [None, dim_1, dim_2, dim_3], name='Input')

    y = tf.placeholder('float', name='Output')

    DataCenter.x_placeholder = x
    DataCenter.y_placeholder = y

    return

def initialize_siamese_placeholders(DataCenter):

    dims = len(DataCenter.train_input_data.shape)-1

    dim_1 = DataCenter.train_input_data.shape[1]

    if dims == 1:
        x_left = tf.placeholder('float', [None, dim_1], name='Input_left')
        x_right = tf.placeholder('float', [None, dim_1], name='Input_right')

    if dims == 2:
        dim_2 = DataCenter.train_input_data.shape[2]
        x_left = tf.placeholder('float', [None, dim_1, dim_2], name='Input_left')
        x_right = tf.placeholder('float', [None, dim_1, dim_2], name='Input_right')

    if dims == 3:
        dim_2 = DataCenter.train_input_data.shape[2]
        dim_3 = DataCenter.train_input_data.shape[3]
        x_left = tf.placeholder('float', [None, dim_1, dim_2, dim_3], name='Input_left')
        x_right = tf.placeholder('float', [None, dim_1, dim_2, dim_3], name='Input_right')

    y = tf.placeholder('float', name='Output')

    DataCenter.x_left_placeholder = x_left
    DataCenter.x_right_placeholder = x_right
    DataCenter.y_placeholder = y

    return