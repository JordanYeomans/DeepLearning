import numpy as np

import report_writing_functions.research.figures as research
import DeepLearning.DataCenter.DataProcessing as data

def combined_1D_onehot(DataCenter, folder_paths, samples = 4):
    folder_paths = np.array(folder_paths)

    channels = folder_paths.shape[0]

    plot = research.SubPlot(samples, channels)

    update_plot = 1

    # Iterate over all samples
    for sample in range(samples):

        for channel in range(channels):

            one_hot_labels = data.load_data(folder_paths[channel] + 'one_hot_labels.csv')
            max_curves = data.load_data(folder_paths[channel] + 'max_array_record.csv')
            min_curves = data.load_data(folder_paths[channel] + 'min_array_record.csv')

            participants = max_curves[:, 0]

            max_curves = max_curves[:, 1:]
            min_curves = min_curves[:, 1:]

            # Randomly Select Input Data
            rand_idx = np.random.randint(DataCenter.all_input_data.shape[0])

            # Get Output idx
            output_idx = np.argmax(DataCenter.all_output_data[rand_idx])

            # Get Participant ID from One Hot Labels
            participant = one_hot_labels[output_idx]

            curve_id = np.array(np.where(participants == participant))[0][0]

            max_curve = np.divide(max_curves[curve_id],DataCenter.input_scale[channel])
            min_curve = np.divide(min_curves[curve_id], DataCenter.input_scale[channel])

            # Plot Max/Min Curves
            plot.current_plot = update_plot
            plot.add_subplot_data(max_curve, add_data_to=update_plot, color='black')
            plot.add_subplot_data(min_curve, add_data_to=update_plot, color='black')

            # Plot Input Data
            input_data = DataCenter.all_input_data[rand_idx][:,channel]
            plot.add_subplot_data(input_data, add_data_to=update_plot, color='blue')

            update_plot += 1

    plot.show_plow()