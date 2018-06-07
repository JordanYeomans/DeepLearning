import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Reshape
from keras.optimizers import Adam
import keras.backend as K

def replace_dense(model,num_remove, new_layer_sizes=None, learning_rate = 0.0005, loss=None, optimizer=None, activation = None, basemodel_trainable = False):

    model.layers = model.layers[:-num_remove-1]

    if basemodel_trainable == False:
        for layer in model.layers:
            layer.trainable = False

    #model.add(Flatten())
    model.add(Dense(128, name='Extra_Layer_1', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, name='Extra_Layer_2', activation='relu'))
    model.add(Dropout(0.5))

    if activation == True:
        model.add(Dense(29, name='Extra_Layer_3', activation=activation))
    else:
        model.add(Dense(29, name='Extra_Layer_3', activation=activation))

    if optimizer == None:
        optimizer = Adam(learning_rate)

    if loss == None:
        loss = 'mean_squared_error'

    model.compile(loss = loss, optimizer=optimizer)

    print('Transfer Learning Model after modifications')

    model.summary()

    return model

def turn_off_layer_training(model, layer_range):

    layer_range = np.array(layer_range)

    for i in layer_range:
        layer = model.layers[i]
        layer.trainable = False
        print('Freezing Layer #{}: {}'.format(i,layer.name))

    model.compile()
    return model

def turn_on_layer_training(model, layer_range):

    layer_range = np.array(layer_range).reshape(-1)

    for i in layer_range:
        layer = model.layers[i]
        layer.trainable = True
        print('Unfreezing Layer #{}: {}'.format(i,layer.name))


    return model