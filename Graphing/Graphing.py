import numpy as np
import matplotlib.pyplot as plt
import copy
import report_writing_functions.research.figures as research
import report_writing_functions.academic.figures as academic

class Graph():
    def __init__(self):
        self.graph_type = 'Research'

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
    restrict_samples(GraphData, samples)

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
        plot.add_subplot_data(GraphData.eval_true, GraphData.eval_predictions, type='scatter')

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
            plot.add_subplot_data(GraphData.eval_true[:,column], GraphData.eval_predictions[:,column], type='scatter')

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

    GraphData = Graph()

    GraphData.train_true = copy.deepcopy(DataCenter.train_true)
    GraphData.train_predictions = copy.deepcopy(DataCenter.train_predictions)

    GraphData.val_input_data = copy.deepcopy(DataCenter.val_input_data)
    GraphData.val_true = copy.deepcopy(DataCenter.val_true)
    GraphData.val_predictions = copy.deepcopy(DataCenter.val_predictions)

    GraphData.eval_input_data = copy.deepcopy(DataCenter.eval_input_data)
    GraphData.eval_true = copy.deepcopy(DataCenter.eval_true)
    GraphData.eval_predictions = copy.deepcopy(DataCenter.eval_predictions)

    return GraphData

def restrict_samples(GraphData, samples):
    # Restrict to number of samples
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