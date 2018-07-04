import DeepLearning.Tensorflow_Base_Functions.evaluation as tfEval

import numpy as np
import tensorflow as tf

def predict_network(DataCenter, model, layers, layer_types):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Loading Weights")

        # Load Weights
        for i in range(len(layers)):
            # Get Layer Type
            type = layer_types[i]

            with tf.variable_scope(DataCenter.transfer_model_scope + '/' + layers[i], reuse=True):
                # Convolution or Dense Layers
                if type == 'conv' or type == 'dense':
                    weight = tf.get_variable('kernel')
                    bias = tf.get_variable('bias')

                    weight_name = str(weight.name)
                    bias_name = str(bias.name)

                    weight_name = weight_name[weight_name.find('/'):]
                    bias_name = bias_name[bias_name.find('/'):]

                    weight_pos = np.where(DataCenter.transfer_var_names == weight_name)[0][0]
                    bias_pos = np.where(DataCenter.transfer_var_names == bias_name)[0][0]

                    sess.run(tf.assign(weight, DataCenter.transfer_var_values[weight_pos]))
                    sess.run(tf.assign(bias, DataCenter.transfer_var_values[bias_pos]))

                elif type == 'batchnorm':
                    gamma = tf.get_variable('gamma')
                    beta = tf.get_variable('beta')

                    gamma_name = str(gamma.name)
                    beta_name = str(beta.name)

                    gamma_name = gamma_name[gamma_name.find('/'):]
                    beta_name = beta_name[beta_name.find('/'):]

                    gamma_pos = np.where(DataCenter.transfer_var_names == gamma_name)[0][0]
                    beta_pos = np.where(DataCenter.transfer_var_names == beta_name)[0][0]

                    sess.run(tf.assign(gamma, DataCenter.transfer_var_values[gamma_pos]))
                    sess.run(tf.assign(beta, DataCenter.transfer_var_values[beta_pos]))

                print('Loaded {}% of Weights'.format(np.round(i / len(layers) * 100)))

        # Make Prediction
        predictions = tfEval.predict(DataCenter, model, DataCenter.predict_input_data)

        DataCenter.predict_output_data = predictions

        return predictions