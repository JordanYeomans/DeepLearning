import numpy as np
import matplotlib.pyplot as plt

import copy
import report_writing_functions.research.figures as research
import report_writing_functions.academic.figures as academic
import DeepLearning.DataCenter.DataProcessing as data

class Graph():
    def __init__(self):
        self.graph_type = 'Research'

        # Data
        self.train_input_data = None
        self.train_output_data = None
        self.val_input_data = None
        self.val_output_data = None
        self.eval_input_data = None
        self.eval_output_data = None

        # Predictions
        self.train_true = None
        self.train_predictions = None
        self.val_input_data = None
        self.val_true = None
        self.val_predictions = None
        self.eval_input_data = None
        self.eval_true = None
        self.eval_predictions = None


def correlation_train_val_eval(DataCenter, plot_outputs = None, samples = 1000, shuffle = True):

    # Initiate Graph Data
    GraphData = graph_data(DataCenter)

    # Restrict Samples
    restrict_prediction_samples(GraphData, samples)


    # Todo Shuffle
    if plot_outputs == None:
        # Define Graph Type
        if GraphData.graph_type == 'Research':
            plot = research.SubPlot(1,3)

        elif GraphData.graph_type == 'Academic':
            plot = academic.SubPlot(1, 3)

        # Plot Data
        print(GraphData.train_true.shape)
        print(GraphData.train_predictions.shape)

        plot.add_subplot_data(GraphData.train_true, GraphData.train_predictions, type='scatter')
        plot.add_subplot_data(GraphData.val_true, GraphData.val_predictions, type='scatter')
        try:
            plot.add_subplot_data(GraphData.eval_true, GraphData.eval_predictions, type='scatter')
        except:
            print('No Evaluation Data')

    else:
        # Define Graph Type
        if GraphData.graph_type == 'Research':
            plot = research.SubPlot(len(plot_outputs), 3)

        elif GraphData.graph_type == 'Academic':
            plot = academic.SubPlot(len(plot_outputs), 3)

        for i in range(len(plot_outputs)):
            column = plot_outputs[i]

            plot.add_subplot_data(GraphData.train_true[:,column], GraphData.train_predictions[:,column], type='scatter')
            plot.add_subplot_data(GraphData.val_true[:,column], GraphData.val_predictions[:,column], type='scatter')
            try:
                plot.add_subplot_data(GraphData.eval_true[:,column], GraphData.eval_predictions[:,column], type='scatter')
            except:
                print('No Evaluation Data')

    return plot


def train_val_eval_samples(DataCenter, samples = 4, shuffle = True):

    # Initiate Graph Data
    GraphData = graph_data(DataCenter)

    # Shuffle Data
    if shuffle is True:
        shuffle_data_samples(GraphData)

    # Restrict Samples
    restrict_data_samples(GraphData, samples)

    # Define Graph Type
    if GraphData.graph_type == 'Research':
        plot = research.SubPlot(2, 3)

    elif GraphData.graph_type == 'Academic':
        plot = academic.SubPlot(2, 3)

    # Plot Input Data
    for i in range(samples):
        plot.current_plot = 1
        plot.add_subplot_data(GraphData.train_input_data[i], add_data_to=1, title='Training Input Examples')

        plot.current_plot = 2
        plot.add_subplot_data(GraphData.val_input_data[i], add_data_to=2, title='Validation Input Examples')

        plot.current_plot = 3
        plot.add_subplot_data(GraphData.eval_input_data[i], add_data_to=3, title='Evaluation Input Examples')

        plot.current_plot = 4
        plot.add_subplot_data(GraphData.train_output_data[i], add_data_to=4, title='Training Output Examples')

        plot.current_plot = 5
        plot.add_subplot_data(GraphData.val_output_data[i], add_data_to=5, title='Validation Output Examples')

        plot.current_plot = 6
        plot.add_subplot_data(GraphData.eval_output_data[i], add_data_to=6, title='Evaluation Output Examples')

    return plot


def train_val_samples(DataCenter, samples = 4, shuffle = True):

    # Initiate Graph Data
    GraphData = graph_data(DataCenter)

    # Shuffle Data
    if shuffle is True:
        shuffle_data_samples(GraphData)

    # Restrict Samples
    restrict_data_samples(GraphData, samples)

    # Define Graph Type
    if GraphData.graph_type == 'Research':
        plot = research.SubPlot(2, 2)

    elif GraphData.graph_type == 'Academic':
        plot = academic.SubPlot(2, 2)

    # Plot Input Data
    for i in range(samples):
        plot.current_plot = 1
        plot.add_subplot_data(GraphData.train_input_data[i], add_data_to=1, title='Training Input Examples')

        plot.current_plot = 2
        plot.add_subplot_data(GraphData.val_input_data[i], add_data_to=2, title='Validation Input Examples')

        plot.current_plot = 3
        plot.add_subplot_data(GraphData.train_output_data[i], add_data_to=3, title='Training Output Examples')

        plot.current_plot = 4
        plot.add_subplot_data(GraphData.val_output_data[i], add_data_to=4, title='Validation Output Examples')

    return plot


def reshape_channel(data):
    channels = data.shape[-1]

    if channels == 1:
        if data.shape[channels+1] == 1:
            data = data.reshape(data.shape[0], data.shape[1])

    else:
        raise 'To Do: Code needed for more than 1 dimension'

    return data


def graph_data(DataCenter):
    ''' Object to copy data from DataCenter.
    - We copy it so we can manipulate it without changing the core data
    - We can't copy the entire object as deepcopy can't copy Tensorflow Tensors

    '''
    GraphData = Graph()

    # Copy Data
    GraphData.train_input_data = copy.deepcopy(DataCenter.train_input_data)
    GraphData.val_input_data = copy.deepcopy(DataCenter.val_input_data)
    GraphData.eval_input_data = copy.deepcopy(DataCenter.eval_input_data)

    GraphData.train_output_data = copy.deepcopy(DataCenter.train_output_data)
    GraphData.val_output_data = copy.deepcopy(DataCenter.val_output_data)
    GraphData.eval_output_data = copy.deepcopy(DataCenter.eval_output_data)

    # Copy Predictions
    try:
        GraphData.train_true = copy.deepcopy(DataCenter.train_true)
        GraphData.train_predictions = copy.deepcopy(DataCenter.train_predictions)

        GraphData.val_input_data = copy.deepcopy(DataCenter.val_input_data)
        GraphData.val_true = copy.deepcopy(DataCenter.val_true)
        GraphData.val_predictions = copy.deepcopy(DataCenter.val_predictions)

        GraphData.eval_input_data = copy.deepcopy(DataCenter.eval_input_data)
        GraphData.eval_true = copy.deepcopy(DataCenter.eval_true)
        GraphData.eval_predictions = copy.deepcopy(DataCenter.eval_predictions)

    except AttributeError:
        print('Just letting you know... No Predictions in DataCenter Object')

    return GraphData


def restrict_data_samples(GraphData, samples):
    ''' Restrict the number of data samples
    '''

    GraphData.train_input_data = GraphData.train_input_data[:samples]
    GraphData.train_output_data = GraphData.train_output_data[:samples]
    GraphData.val_input_data = GraphData.val_input_data[:samples]
    GraphData.val_output_data = GraphData.val_output_data[:samples]
    GraphData.eval_input_data = GraphData.eval_input_data[:samples]
    GraphData.eval_output_data = GraphData.eval_output_data[:samples]


def restrict_prediction_samples(GraphData, samples):
    ''' Restrict the number of prediction samples
    '''

    GraphData.train_true = GraphData.train_true[:samples]
    GraphData.train_predictions = GraphData.train_predictions[:samples]

    # Check if enough validation samples, if not, restrict to number of validation samples
    if GraphData.val_input_data.shape[0] > samples:
        GraphData.val_true = GraphData.val_true[:samples]
        GraphData.val_predictions = GraphData.val_predictions[:samples]
    else:
        GraphData.val_true = GraphData.val_true[:GraphData.val_input_data.shape[0]]
        GraphData.val_predictions = GraphData.val_predictions[:GraphData.val_input_data.shape[0]]

    # Check if enough evaluation samples, if not, restrict to number of evaluation samples
    if GraphData.eval_input_data.shape[0] > samples:
        GraphData.eval_true = GraphData.eval_true[:samples]
        GraphData.eval_predictions = GraphData.eval_predictions[:samples]
    else:
        GraphData.eval_true = GraphData.eval_true[:GraphData.eval_input_data.shape[0]]
        GraphData.eval_predictions = GraphData.eval_predictions[:GraphData.eval_input_data.shape[0]]


def shuffle_data_samples(GraphData):
    ''' Shuffle Data
    '''

    GraphData.train_input_data, GraphData.train_output_data = data.shuffle_input_output(GraphData.train_input_data,
                                                                                        GraphData.train_output_data)

    GraphData.val_input_data, GraphData.val_output_data = data.shuffle_input_output(GraphData.val_input_data,
                                                                                        GraphData.val_output_data)

    GraphData.eval_input_data, GraphData.eval_output_data = data.shuffle_input_output(GraphData.eval_input_data,
                                                                                        GraphData.eval_output_data)