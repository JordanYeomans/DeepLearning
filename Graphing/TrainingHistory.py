import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import report_writing_functions.research.figures as research
import report_writing_functions.academic.figures as academic
import report_writing_functions.presentations.figures_4_by_3 as pres4by3
import DeepLearning.DataCenter.DataProcessing as data


class Graph():
    def __init__(self):
        self.graph_type = 'research'
        self.graph_save_location = './'
        self.model_save_location = './'
        self.log_prefix = ''
        self.plot = None

    def load_training_history(self):
        self.loss_train_log = np.array(pd.read_csv(self.model_save_location + self.log_prefix + 'Train_Loss_Log.csv', header=None))
        self.loss_val_log = np.array(pd.read_csv(self.model_save_location + self.log_prefix + 'Val_Loss_Log.csv', header=None))
        self.acc_train_log = np.array(pd.read_csv(self.model_save_location + self.log_prefix + 'Train_Acc_Log.csv', header=None))
        self.acc_val_log = np.array(pd.read_csv(self.model_save_location + self.log_prefix + 'Val_Acc_Log.csv', header=None))

    def plot_train_loss_history(self):

        self.format_data()
        if self.graph_type == 'research':
            self.plot = research.SubPlot(1,1)
        elif self.graph_type == 'pres4by3':
            self.plot = pres4by3.SubPlot(1,1)

        self.plot.add_subplot_data(self.loss_train_log)
        self.plot.show_plow()

    def plot_train_loss_val_acc_history(self, x_lim):
        self.avg_train_batch_loss()
        self.format_data()
        if self.graph_type == 'research':
            self.plot = research.SubPlot(1,2)
        elif self.graph_type == 'pres4by3':
            self.plot = pres4by3.SubPlot(1,2)

        self.plot.current_plot=1
        self.plot.add_subplot_data(self.loss_train_log, add_data_to=1, xlim=x_lim)
        self.plot.current_plot = 2
        self.plot.add_subplot_data(self.acc_val_log, ylim=[0,1], add_data_to=2, xlim=x_lim)

        self.plot.save_figure(self.graph_save_location)

    def show_plot(self):
        self.plot.show_plow()

    ## Functions to format Data
    def avg_train_batch_loss(self):
        self.loss_train_log = np.mean(self.loss_train_log, axis=1)

    def format_data(self):
        self.loss_train_log = self.loss_train_log[self.loss_train_log!=0]
        self.acc_val_log = self.acc_val_log[self.acc_val_log!=0]
