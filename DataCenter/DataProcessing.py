import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy.signal
import skimage.transform

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

    num_batches = int(input_data.shape[0]/batch_size)

    new_input_data = np.array([input_data[:batch_size]])
    new_output_data = np.array(([output_data[:batch_size]]))

    # Note:
    # Input and Output data dimensions will change depending on data type. An image output for say image segmentation
    # will have a different output dimension to categories.

    # Create Input Batch Array
    if new_input_data.ndim == 4:
        new_input_data = np.zeros((num_batches, new_input_data.shape[1], new_input_data.shape[2], new_input_data.shape[3]))
    else:
        raise 'Need to add functionality for Not 2 Input dimensions'

    # Create Output Batch Array
    if new_output_data.ndim == 3:
        new_output_data = np.zeros((num_batches, new_output_data.shape[1], new_output_data.shape[2]))
    else:
        raise 'Need to add functionality for Not 1 Output dimensions'

    if num_batches > 1:
        for i in range(num_batches):
            new_input_data[i] = np.array([input_data[batch_size * i : batch_size * (i+1)]])
            new_output_data[i] = np.array([output_data[batch_size * i : batch_size * (i+1)]])

    return new_input_data, new_output_data

def augment_1D_left_right(input_data, output_data, left, right, step):

    assert (left + right)%step == 0, 'Non integer steps. Consider using integers and step = 1'

    augment_range = np.arange(-left, right+step, step)
    augmented_data = numpy_zeros_extended(input_data, augment_range.shape[0], type='extend_dim_1')
    new_output_data = numpy_zeros_extended(output_data, augment_range.shape[0], type='extend_dim_1')

    num_samples = input_data.shape[0]

    count = 0

    for i in augment_range:

        sample_start = count * num_samples
        sample_end = (count + 1) * num_samples

        if i < 0:
            new_data = input_data[:, np.abs(i):]
            augmented_data[sample_start:sample_end, :new_data.shape[1]] = new_data
        elif i > 0:
            new_data = input_data[:, :-np.abs(i)]
            augmented_data[sample_start:sample_end, i:] = new_data
        else:
            augmented_data[sample_start:sample_end, i:] = input_data

        new_output_data[sample_start:sample_end,:] = output_data

        count += 1

    assert len(np.where(~augmented_data.any(axis=1))[0]) == 0, 'Input Data contains Data that is all Zeros'

    return augmented_data, new_output_data

def augment_1D_squeeze_stretch(input_data, output_data, squeeze, stretch, steps):


    squeeze = np.floor(input_data.shape[1] * squeeze)
    squeeze = squeeze - input_data.shape[1]

    stretch = np.ceil(input_data.shape[1] * stretch)
    stretch = stretch - input_data.shape[1]

    step = (stretch-squeeze)/steps
    augment_range = np.arange(squeeze, squeeze + (steps * (step+1)), step)

    augmented_data = numpy_zeros_extended(input_data, augment_range.shape[0], type='extend_dim_1')
    new_output_data = numpy_zeros_extended(output_data, augment_range.shape[0], type='extend_dim_1')

    num_samples = input_data.shape[0]
    orig_signal_len = input_data.shape[1]
    count = 0

    for i in augment_range:
        signal_num=0
        sample_start = count * num_samples
        sample_end = (count + 1) * num_samples

        for sample in range(sample_start,sample_end):

            signal = input_data[signal_num]

            if i < 0:
                signal = np.concatenate([signal, np.zeros((int(np.abs(i)),1))])
            elif i> 0:
                signal = signal[:-int(i)]

            if i == 0:
                signal = signal

            new_data = skimage.transform.resize(signal, (orig_signal_len,1))

            augmented_data[sample] = new_data
            new_output_data[sample] = output_data[signal_num]

            signal_num += 1
        count += 1

    assert len(np.where(~augmented_data.any(axis=1))[0]) == 0, 'Input Data contains Data that is all Zeros'

    return augmented_data, new_output_data

def augment_1D_squash_pull(input_data, output_data, squash, pull, steps, type = 'multiply'):

    step = (pull - squash) / steps

    augment_range = np.arange(squash, squash+(steps*step), step)
    augment_range = np.concatenate([augment_range, np.array(0).reshape(-1)])

    augmented_data = numpy_zeros_extended(input_data, augment_range.shape[0], type='extend_dim_1')
    new_output_data = numpy_zeros_extended(output_data, augment_range.shape[0], type='extend_dim_1')

    num_samples = input_data.shape[0]
    count = 0

    zeros = np.where(input_data == 0)

    print(augmented_data.shape)
    for i in augment_range:

        sample_start = count * num_samples
        sample_end = (count + 1) * num_samples

        if i != 0:
            if type == 'multiply':
                augmented_data[sample_start:sample_end] = np.multiply(input_data,i)
            elif type == 'add':
                augmented_data[sample_start:sample_end] = np.add(input_data, i-1)
            new_output_data[sample_start:sample_end] = output_data
        else:
            augmented_data[sample_start:sample_end] = input_data
            new_output_data[sample_start:sample_end] = output_data

        augmented_data[sample_start:sample_end][zeros] = 0
        count += 1

    assert len(np.where(~augmented_data.any(axis=1))[0]) == 0, 'Input Data contains Data that is all Zeros'

    return augmented_data, new_output_data

def augment_add_noise(input_data, std_dev):

    zeros = np.where(input_data==0)

    noise = np.random.normal(loc = 0, scale=std_dev, size=(input_data.shape[0], input_data.shape[1],1))
    input_data = np.add(input_data,noise)

    input_data[zeros] = 0

    return input_data

def numpy_zeros_extended(data, dimension_multiplier, type = 'extend_dim_1'):

    shape = None
    if type == 'extend_dim_1':
        shape = np.array(data.shape).flatten()
        shape[0] = shape[0] * dimension_multiplier

    if type == 'new_dim_1':
        shape = (dimension_multiplier,) + data.shape

    return np.zeros(shape)