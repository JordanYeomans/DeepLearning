import tensorflow as tf
import numpy as np
import DeepLearning.DataCenter.DataProcessing as DataProcess

def prediction_accuracy(DataCenter, model, x_data, y_data):
    ''' Calculate prediction accuracy between x_data and y_data from a NN model

    '''
    x = DataCenter.x_placeholder
    y = DataCenter.y_placeholder

    acc = 0
    acc_rec = np.zeros(x_data.shape[0])

    correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1), name='Correct')
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    for i in range(x_data.shape[0]):
        val_x = x_data[i]
        val_y = y_data[i]

        acc_rec[i] = accuracy.eval({x: val_x, y: val_y})
        acc = np.mean(acc_rec)

    return acc

def mse(DataCenter, cost):
    ''' Evaluate Mean Squared Error
    '''

    x = DataCenter.x_placeholder
    y = DataCenter.y_placeholder

    mse = np.zeros(DataCenter.val_input_batches.shape[0])
    for i in range(DataCenter.val_input_batches.shape[0]):
        mse[i] = cost.eval(feed_dict={x: DataCenter.val_input_batches[0], y: DataCenter.val_output_batches[0]})

    mse = np.mean(mse, axis=0)
    return mse

def predict(DataCenter, model, x_data):
    ''' Prediction for x_data through NN model
    '''
    x = DataCenter.x_placeholder

    predictions = model.eval(feed_dict={x: x_data[0]})

    for i in range(1, x_data.shape[0]):
        pred = model.eval(feed_dict={x: x_data[i]})
        predictions = np.concatenate([predictions, pred], axis=0)

    return predictions

def true_outputs(y_data):
    ''' For Tensorflow based mini-batches, this function concatenates different batches into a single axis
    '''
    true = y_data[0]
    for i in range(1, y_data.shape[0]):
        true = np.concatenate([true, y_data[i]], axis=0)
    return true

def predict_train_val_eval(DataCenter, model):
    ''' Prediction for all training, validation and evaluation batches
    '''

    print('Predicting Training, Validation & Evaluation Batches')
    train_x_all = DataCenter.train_input_batches
    train_y_all = DataCenter.train_output_batches

    val_x_all = DataCenter.val_input_batches
    val_y_all = DataCenter.val_output_batches

    eval_x_all = DataCenter.eval_input_batches
    eval_y_all = DataCenter.eval_output_batches

    DataCenter.train_true = true_outputs(train_y_all)
    DataCenter.train_predictions = predict(DataCenter, model, train_x_all)

    DataCenter.val_true = true_outputs(val_y_all)
    DataCenter.val_predictions = predict(DataCenter, model, val_x_all)

    DataCenter.eval_true = true_outputs(eval_y_all)
    DataCenter.eval_predictions = predict(DataCenter, model, eval_x_all)

def export_val_one_hot_predictions(DataCenter, model):
    val_predictions = predict(DataCenter, model, DataCenter.val_input_batches)
    val_true = DataProcess.combine_batches(DataCenter.val_output_batches)
    val_true_arg_max = np.argmax(val_true, axis=1)
    return np.concatenate([val_true_arg_max, val_predictions], axis=1)