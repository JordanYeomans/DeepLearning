import numpy as np
import pandas as pd
import time

def load_data(filepath):
    return np.array(pd.read_csv(filepath, header=None))


def split_input_output_data(data, num_outputs, output_first):
    # Split data into output and input data
    if output_first == True:
        output_data = data[:,:num_outputs]
        input_data = data[:,num_outputs:]

    else:
        input_data = data[:, :num_outputs]
        output_data = data[:, num_outputs:]

    return input_data, output_data


def scale_input(data, scale = None):

    if scale == None:
        scale_min = np.abs(np.min(data))
        scale_max = np.abs(np.max(data))

        scale = np.max([scale_min, scale_max])

    data = np.divide(data,scale)

    return data, scale


def one_hot_output(data, column, concat=True):
    '''This function converts an output column to a 1 hot array

    :param data: Output Data
    :param column:
    :param concat:
    :return:
    '''

    column_data = data[:, column]

    # Create One Hot Array
    one_hot_labels = np.unique(column_data)
    one_hot_array = np.zeros((column_data.shape[0], one_hot_labels.shape[0]))

    for i in range(one_hot_array.shape[0]):
        value = column_data[i]

        idx = np.where(value == one_hot_labels)

        one_hot_array[i][idx] = 1

    if concat == True:
        output_data = np.concatenate([data[:,:column],one_hot_array,data[:,column:]], axis=1)
    else:
        output_data = one_hot_array

    return output_data, one_hot_labels


def scale_outputs(data):

    scale_record = np.zeros((data.shape[1]))
    for i in range(data.shape[1]):

        scale_min = np.abs(np.min(data[:, i]))
        scale_max = np.abs(np.max(data[:, i]))
        scale = np.max([scale_min, scale_max])

        scale_record[i] = scale

        data[:, i] = np.divide(data[:, i], scale)

    data = np.nan_to_num(data)

    return data, scale_record


def shuffle_input_output(input_data, output_data):

    s = np.arange(input_data.shape[0])
    np.random.shuffle(s)

    input_data = input_data[s]
    output_data = output_data[s]

    return input_data, output_data


def cut_input_data_seq_length(input_data, out_length = None):
    ''' This function adds zeros to the 1st dimension of an array

    Input: Data of shape (Samples, 800)
    Output: Data of shape (Samples, 1000) with zeros from 800 - 1000
    '''
    if out_length == None:
        out_length = input_data.shape[1]

    new_input_data = np.zeros((input_data.shape[0], out_length))

    for i in range(input_data.shape[0]):
        if input_data[i].shape[0] <= out_length:
            new_input_data[i,:input_data[i].shape[0]] = input_data[i,:input_data[i].shape[0]]
        else:
            new_input_data[i, :out_length] = input_data[i, :out_length]

    return new_input_data


def reshape_1D_input(input_data):
    return input_data.reshape(input_data.shape[0], input_data.shape[1], 1)


def convert_to_tensorflow_minbatch(input_data, output_data, batch_size):
    ''' Convert an array from [Samples, data size] - [Batches, Batch_size, Data size]

    '''
    print('Splitting Into Batches')

    num_samples = int(input_data.shape[0]/batch_size)

    new_input_data = np.array([input_data[:batch_size]])
    new_output_data = ([output_data[:batch_size]])

    if num_samples >=2:
        for i in range(1,num_samples):
            data_in = np.array([input_data[batch_size * i : batch_size * (i+1)]])
            data_out = np.array([output_data[batch_size * i : batch_size * (i+1)]])

            new_input_data = np.concatenate([new_input_data, data_in], axis=0)
            new_output_data = np.concatenate([new_output_data, data_out], axis=0)

            print(i/num_samples)

    return np.array(new_input_data), np.array(new_output_data)
