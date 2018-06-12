import tensorflow as tf
import numpy as np

def vis_1D_Conv_layer(DataCenter, layer_name):

    with tf.Session() as sess:

        with tf.variable_scope(DataCenter.new_model_scope + '/' + layer_name, reuse=True):
            weight = tf.get_variable('kernel')
            bias = tf.get_variable('bias')

            weight_name = str(weight.name)
            bias_name = str(bias.name)

            weight_name = weight_name[weight_name.find('/'):]
            bias_name = bias_name[bias_name.find('/'):]

            weight_pos = np.where(DataCenter.transfer_var_names == weight_name)[0][0]
            bias_pos = np.where(DataCenter.transfer_var_names == bias_name)[0][0]

    # To Do:

def extract_variables(DataCenter):
    ''' Dulplicate Function from Transfer Learning'''
    with tf.Session() as sess:
        # Load Model
        new_saver = tf.train.import_meta_graph(DataCenter.model_load_folder + DataCenter.model_load_name + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(DataCenter.model_load_folder))

        # Get All Variables
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        var_names = []
        var_values = []

        for i in range(np.array(variables).shape[0]):

            variable = variables[i]

            name = variable.name
            value = sess.run(variable)

            var_names.append(name[name.find('/'):])
            var_values.append(value)

    sess.close()
    tf.reset_default_graph()

    DataCenter.transfer_var_names = np.array(var_names)
    DataCenter.transfer_var_values = var_values

    return
