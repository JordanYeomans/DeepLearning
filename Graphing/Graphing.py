import numpy as np
import matplotlib.pyplot as plt
import copy
import report_writing_functions.research.figures as research
import report_writing_functions.academic.figures as academic


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
        plot.add_subplot_data(GraphData.train_output_batches, GraphData.train_output_batches_pred, type='scatter')
        plot.add_subplot_data(GraphData.val_output_batches, GraphData.val_outputs_batches_pred, type='scatter')
        plot.add_subplot_data(GraphData.eval_output_batches, GraphData.eval_outputs_batches_pred, type='scatter')

    else:
        # Define Graph Type
        if GraphData.graph_type == 'Research':
            plot = research.SubPlot(len(plot_outputs), 3)

        elif GraphData.graph_type == 'Academic':
            plot = academic.SubPlot(len(plot_outputs), 3)

        for i in range(len(plot_outputs)):
            column = plot_outputs[i]

            plot.add_subplot_data(GraphData.train_output_batches[:,column], GraphData.train_output_batches_pred[:,column], type='scatter')
            plot.add_subplot_data(GraphData.val_output_batches[:,column], GraphData.val_outputs_batches_pred[:,column], type='scatter')
            plot.add_subplot_data(GraphData.eval_output_batches[:,column], GraphData.eval_outputs_batches_pred[:,column], type='scatter')

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
    GraphData = copy.deepcopy(DataCenter)

    return GraphData

def restrict_samples(GraphData, samples):
    # Restrict to number of samples
    GraphData.train_output_batches = GraphData.train_output_batches[:samples]
    GraphData.train_output_batches_pred = GraphData.train_output_batches_pred[:samples]

    # Check if enough validation samples, if not, restrict to number of validation samples
    if GraphData.val_input_data.shape[0] > samples:
        GraphData.val_output_batches = GraphData.val_output_batches[:samples]
        GraphData.val_outputs_batches_pred = GraphData.val_outputs_batches_pred[:samples]
    else:
        GraphData.val_output_batches = GraphData.val_output_batches[:GraphData.val_input_data.shape[0]]
        GraphData.val_outputs_batches_pred = GraphData.val_outputs_batches_pred[:GraphData.val_input_data.shape[0]]

    # Check if enough evaluation samples, if not, restrict to number of evaluation samples
    if GraphData.eval_input_data.shape[0] > samples:
        GraphData.eval_output_batches = GraphData.eval_output_batches[:samples]
        GraphData.eval_outputs_batches_pred = GraphData.eval_outputs_batches_pred[:samples]
    else:
        GraphData.eval_output_batches = GraphData.eval_output_batches[:GraphData.eval_input_data.shape[0]]
        GraphData.eval_outputs_batches_pred = GraphData.eval_outputs_batches_pred[:GraphData.eval_input_data.shape[0]]

# # Reshape Inputs
# Graph.val_input_batches = reshape_channel(Graph.val_input_batches)
# Graph.eval_input_batches = reshape_channel(Graph.eval_input_batches)
