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


def scale_multi_chan_input(data, scale = None):

    channels = data.shape[2]

    scale_sheet = []

    if scale == None:

        for channel in range(channels):
            scale_min = np.abs(np.min(data[:, :, channel]))
            scale_max = np.abs(np.max(data[:, :, channel]))
            scale = np.max([scale_min, scale_max])
            data[:, :, channel] = np.divide(data[:, :, channel],scale)
            scale_sheet.append(scale)

    return data, scale_sheet


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
    print(new_output_data.shape)
    if new_output_data.ndim == 3:
        new_output_data = np.zeros((num_batches, new_output_data.shape[1], new_output_data.shape[2]))
    else:
        raise 'Need to add functionality for Not 1 Output dimensions'

    if num_batches > 1:
        for i in range(num_batches):
            new_input_data[i] = np.array([input_data[batch_size * i : batch_size * (i+1)]])
            new_output_data[i] = np.array([output_data[batch_size * i : batch_size * (i+1)]])
            print('Splitting into batches {}%'.format(np.round(i/num_batches*100), 2))
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


def augment_1D_squash_pull(input_data, output_data, squash, pull, steps, type='multiply'):

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


def numpy_zeros_extended(data, dimension_multiplier, type='extend_dim_1'):

    shape = None
    if type == 'extend_dim_1':
        shape = np.array(data.shape).flatten()
        shape[0] = shape[0] * dimension_multiplier

    if type == 'new_dim_1':
        shape = (dimension_multiplier,) + data.shape

    return np.zeros(shape)


def restrict_to_ids(all_input_data, all_output_data, ids, column):

    for i in range(len(ids)):

        id_pos = np.array(np.where(all_output_data[:,column] == ids[i])[0])

        if i == 0:
            id_record = id_pos
        else:
            id_record = np.concatenate([id_record, id_pos], axis = 0)

    all_input_data = all_input_data[id_record]
    all_output_data = all_output_data[id_record]

    return all_input_data, all_output_data


def integrate_input_curve(all_input_data, col_start = None, col_end = None):

    all_output_data = np.zeros((all_input_data.shape[0], all_input_data.shape[2]))

    for i in range(all_input_data.shape[2]):
        all_output_data[:, i] = np.sum(all_input_data[:, col_start:col_end, i], axis=1)

    return all_output_data


def balance_batch_for_one_hot(all_input_data, all_output_data, min_samples):

    current_min = all_output_data.shape[0]

    temp_input_data = np.zeros_like(all_input_data)
    temp_output_data = np.zeros_like(all_output_data)

    # Get number of minimum samples
    for i in range(all_output_data.shape[1]):
        num_samples = np.array(np.where(all_output_data[:,i] == 1)[0]).shape[0]
        if num_samples < current_min and num_samples > min_samples:
            current_min = num_samples

    num_samples = current_min
    current_sample = 0

    # Randomly allocate balanced number of samples
    for i in range(all_output_data.shape[1]):
        # Get all samples for index, shuffle and restrict
        sample_idx = np.array(np.where(all_output_data[:, i] == 1)[0])
        np.random.shuffle(sample_idx)
        sample_idx = sample_idx[:num_samples]

        # If enough samples, add as many as we can
        if sample_idx.shape[0] >= num_samples:
            temp_input_data[current_sample: current_sample + num_samples] = all_input_data[sample_idx]
            temp_output_data[current_sample: current_sample + num_samples] = all_output_data[sample_idx]
            current_sample += num_samples

        else:
            temp_input_data[current_sample: current_sample + sample_idx.shape[0]] = all_input_data[sample_idx]
            temp_output_data[current_sample: current_sample + sample_idx.shape[0]] = all_output_data[sample_idx]
            current_sample += sample_idx.shape[0]

    temp_input_data = temp_input_data[:current_sample]
    temp_output_data = temp_output_data[:current_sample]

    all_input_data, all_output_data = shuffle_input_output(temp_input_data, temp_output_data)

    return all_input_data, all_output_data


def create_continuous_one_hot_array(all_output_data, val_min = None, val_max = None, one_hot_length = 10):
    ''' This function creates a continuous valued one hot array, with even increments between each jump

    Example Code:

    output = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    one_hot, true_range = data.continuous_one_hot_array(output, val_min=2, val_max=4, one_hot_length=11)

    One_hot =
        [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]

    True Range =
        [2.  2.2 2.4 2.6 2.8 3.  3.2 3.4 3.6 3.8 4. ]

    :param all_output_data: A numpy array containing single valued outputs
    :param val_min: Optional - If specified, values lower than this value will be put in the first index
    :param val_max: Optional - If specified, values greater than this value will be put in the last index
    :param one_hot_length: Desired one-hot-array length. Often a odd number to include 0 and a even final number
    :return:
    '''

    if val_min is None:
        val_min = np.min(all_output_data)
    else:
        # Any values less than the lowest value, set to the lowest value
        print('Creating Continuous 1 Hot Array with pre-set min value. Values lower than the preset value are stored in the first index')
        neg_idx = np.array(np.where(all_output_data < val_min)[0])
        all_output_data[neg_idx] = val_min

    if val_max is None:
        val_max = np.max(all_output_data)

    else:
        # Any values greater than the max value, set to the max value
        print('Creating Continuous 1 Hot Array with pre-set max value. Values lower than the preset value are stored in the last index')
        pos_idx = np.array(np.where(all_output_data > val_max)[0])
        all_output_data[pos_idx] = val_max

    one_hot_range = val_max - val_min
    assert one_hot_range != 0, print('All outputs are the same number')

    dist_from_min = np.subtract(all_output_data, val_min)
    perc_of_range = np.divide(dist_from_min, one_hot_range)

    one_hot_idx = np.round(np.multiply(one_hot_length-1, perc_of_range),0)

    continuous_one_hot = np.zeros((all_output_data.shape[0], one_hot_length))

    for i in range(continuous_one_hot.shape[0]):
        idx = int(one_hot_idx[i])
        continuous_one_hot[i][idx] = 1

    true_range = np.linspace(val_min, val_max, one_hot_length)
    return continuous_one_hot, true_range


def continuous_mse_loss(all_output_data, base_width, power, top_width, offset = 0.5):
    ''' This function converts a continuous values one hot array into a trapezoidal shaped loss

    example code:

    output = np.array([0, 1, 2, 2.5, 3, 3.5, 4, 5])
    one_hot, true_range = create_continuous_one_hot_array(one_hot_length=51)
    new_output_data, shift = continuous_mse_loss(one_hot, base_width=100, top_width=5, power=2)

    :param all_output_data:
    :param base_width:
    :param power:
    :param top_width:
    :return:
    '''
    one_hot_length = all_output_data.shape[1]
    padding = (one_hot_length * 2) + 1
    total_length = (padding * 2) + all_output_data.shape[1]

    # Create Top Section. Scale to 0.99 to ensure argmax picks middle
    top_section = np.ones(top_width) * 0.99
    top_half = int(np.floor(top_width / 2))
    top_section[top_half] = 1

    # Create LHS and RHS of trapezoid
    steps = int((base_width - top_width) / 2)
    trap_lhs = np.linspace(0, 0.99, steps)
    trap_rhs = np.linspace(0.99, 0, steps)

    # Trapezoid power. P>1 = Convex, 0<P<1 = Concave
    trap_lhs = np.power(trap_lhs, power)
    trap_rhs = np.power(trap_rhs, power)

    tail_zeros = (total_length - trap_lhs.shape[0] - top_width - trap_rhs.shape[0])
    assert tail_zeros % 2 == 0, print('Ensure One Hot Array is odd in length')

    # Create and fill new array
    new_output_data = np.zeros((all_output_data.shape[0], total_length))
    for i in range(all_output_data.shape[0]):
        idx = np.argmax(all_output_data[i])

        lhs_zeros = np.zeros(idx + padding - trap_lhs.shape[0] - int(top_width / 2))
        rhs_zeros = np.zeros(
            total_length - trap_rhs.shape[0] - top_section.shape[0] - trap_lhs.shape[0] - lhs_zeros.shape[0])

        new_output = np.concatenate([lhs_zeros, trap_lhs, top_section, trap_rhs, rhs_zeros], axis=0)

        ratio = (1-offset)
        new_output = (new_output * ratio) + offset
        new_output_data[i] = new_output

    shift = padding
    return new_output_data, shift


def padd_one_hot_array(all_output_data, pad_reduce = 0):
    one_hot_length = all_output_data.shape[1]
    padding = (one_hot_length * 2) + 1

    if pad_reduce != 0:
        padding = padding - int((pad_reduce/2))

    pad_zeros = np.zeros((all_output_data.shape[0], padding))

    all_output_data = np.concatenate([pad_zeros, all_output_data, pad_zeros], axis=1)

    if pad_reduce != 0 and pad_reduce % 2 != 0:
        all_output_data = all_output_data[:,1:]

    return all_output_data


def balance_batch_for_dual_sided_one_hot(all_input_data, all_output_data):

    c0_1 = np.where(all_output_data[:, 0] == 1)[0]
    c0_0 = np.where(all_output_data[:, 0] == 0)[0]

    c1_1 = np.where(all_output_data[:, 1] == 1)[0]
    c1_0 = np.where(all_output_data[:, 1] == 0)[0]

    Zero_Zero = np.intersect1d(c0_0, c1_0)
    One_Zero = np.intersect1d(c0_1, c1_0)
    Zero_One = np.intersect1d(c0_0, c1_1)

    One_One_idx = np.intersect1d(c0_1, c1_1)

    samples = [Zero_Zero.shape[0], One_Zero.shape[0], Zero_One.shape[0]]
    min_samples = np.min(samples)

    np.random.shuffle(Zero_Zero)
    np.random.shuffle(One_Zero)
    np.random.shuffle(Zero_One)

    Zero_Zero_idx = Zero_Zero[:min_samples]
    One_Zero_idx = One_Zero[:min_samples]
    Zero_One_idx = Zero_One[:min_samples]

    all_input_data = np.concatenate([all_input_data[Zero_Zero_idx], all_input_data[One_Zero_idx], all_input_data[Zero_One_idx], all_input_data[One_One_idx]], axis=0)
    all_output_data = np.concatenate([all_output_data[Zero_Zero_idx], all_output_data[One_Zero_idx], all_output_data[Zero_One_idx], all_output_data[One_One_idx]], axis=0)

    all_input_data, all_output_data = shuffle_input_output(all_input_data, all_output_data)

    return all_input_data, all_output_data

def calc_unique_ids(output_data):
    return np.unique(output_data)

def calc_siamese_batches(all_input_data, all_output_data, unique_ids, batches, batch_size, reshape=True):

    def get_random_sample(input_data, unique_id_idx_list, unique_id_idx):
        trace_idxs = unique_id_idx_list[unique_id_idx]
        np.random.shuffle(trace_idxs)
        return input_data[trace_idxs[0]]

    # Get an array of labels. Either 0 or 1
    total_samples = batches * batch_size

    # Get ID Array
    unique_id_idx_list = []
    for i in range(len(unique_ids)):
        id_array = np.array(np.where(all_output_data == unique_ids[i])[0])
        unique_id_idx_list.append(id_array)

    # Create Arrays
    siamese_left_id_idx = np.random.random_integers(0, len(unique_ids)-1, total_samples)

    siamese_right_id_idx = np.zeros_like(siamese_left_id_idx)
    labels = np.zeros_like(siamese_left_id_idx)

    input_data_batches_left = np.zeros((total_samples, all_input_data.shape[1], all_input_data.shape[2]))
    input_data_batches_right = np.zeros((total_samples, all_input_data.shape[1], all_input_data.shape[2]))

    # Iterate over all indexs
    for i in range(len(siamese_right_id_idx)):

        rand_choice = np.random.random()

        # If left and right are the same
        if rand_choice >= 0.5:
            siamese_right_id_idx[i] = siamese_left_id_idx[i]
            labels[i] = 1

        # If not, set right to be a different index
        else:
            siamese_right_id_idx[i] = np.random.randint(0, len(unique_ids))
            while siamese_right_id_idx[i] == siamese_left_id_idx[i]:
                siamese_right_id_idx[i] = np.random.randint(0, len(unique_ids))

            labels[i] = 0

        input_data_batches_left[i] = get_random_sample(all_input_data, unique_id_idx_list, siamese_left_id_idx[i])
        input_data_batches_right[i] = get_random_sample(all_input_data, unique_id_idx_list, siamese_right_id_idx[i])

    # If True, Split into batches
    if reshape is True:
        # Reshape Inputs
        input_data_batches_left = input_data_batches_left.reshape(batches, batch_size, all_input_data.shape[1], all_input_data.shape[2])
        input_data_batches_right = input_data_batches_right.reshape(batches, batch_size, all_input_data.shape[1],all_input_data.shape[2])
        labels = labels.reshape(batches, batch_size)
        siamese_left_id_idx = siamese_left_id_idx.reshape(batches, batch_size)
        siamese_right_id_idx = siamese_right_id_idx.reshape(batches, batch_size)

    return input_data_batches_left, input_data_batches_right, labels, siamese_left_id_idx, siamese_right_id_idx