import numpy as np


def validate_1hot_outputs(oh_labels):
    ''' Check that all output array's contain the same one-hot array values and order

    Example input:
        oh_labels = [X_Data.one_hot_labels,
                    Y_Data.one_hot_labels,
                    Z_Data.one_hot_labels]
    '''
    oh_labels = np.array(oh_labels)

    assert oh_labels[0] is not None, 'One Hot Output Labels are None'

    base = oh_labels[0].reshape(-1)

    for i in range(1, oh_labels.shape[0]):
        label = oh_labels[i].reshape(-1)
        for j in range(label.shape[0]):
            assert label[j] == base[j], 'One Hot Output Labels are different'


def combine_1D_Onehot_DC(input_arrays, output_arrays, samples=None):

    input_arrays = np.array(input_arrays)
    output_arrays = np.array(output_arrays)

    if samples is None:
        samples = input_arrays.shape[1]
    new_input = np.zeros((samples, input_arrays.shape[2], input_arrays.shape[0]))
    new_output = np.zeros((samples, output_arrays.shape[2]))

    ## Fill Index Array
    idx_array = np.zeros((input_arrays.shape[0], output_arrays.shape[2], input_arrays.shape[1]))
    # Iterate over all channels
    for channel in range(input_arrays.shape[0]):
        channel_input_data = input_arrays[channel]
        channel_output_data = output_arrays[channel]

        # Iterate over all training examples
        for i in range(channel_input_data.shape[0]):
            output_idx = np.argmax(channel_output_data[i])
            idx_array[channel][output_idx][i] = i

    ## Fill Samples
    # Iterate over all samples
    for sample in range(samples):
        print('{} Samples out of {}'.format(sample+1, samples))
        rand_output = np.random.randint(0, new_output.shape[1])

        # Iterate over all channels
        for channel in range(input_arrays.shape[0]):
            # Get random index for channel that has an output of rand_output
            rand_idx_array = idx_array[channel][rand_output]
            rand_idx_array = rand_idx_array[rand_idx_array != 0]
            np.random.shuffle(rand_idx_array)
            rand_idx = int(rand_idx_array[0])

            # Get input and output data for index
            in_data = input_arrays[channel][rand_idx]
            out_data = output_arrays[channel][rand_idx]

            # Update complete input/output arrays
            new_input[sample][:,channel] = in_data.reshape(-1)
            new_output[sample] = out_data

    return new_input, new_output
