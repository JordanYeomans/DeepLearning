from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Reshape
from keras.optimizers import Adam
import keras.backend as K

def TimeSeries_Conv_LSTM_Dense_0001_a(input_data, output_data, batch_size, lr):

    model = Sequential()

    ## Convolutional Network
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu',
                     batch_input_shape=(batch_size, input_data.shape[1], input_data.shape[2]),
                     kernel_initializer='TruncatedNormal'))

    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=256, kernel_size=7, padding='valid', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(MaxPooling1D(2))

    model.add(LSTM(256, return_sequences=True, kernel_initializer='TruncatedNormal'))
    model.add(LSTM(256, return_sequences=True, kernel_initializer='TruncatedNormal'))
    model.add(LSTM(256))

    # Multi Layer Perceptron Network
    model.add(Dense(4096, activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(Dense(4096, activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(Dense(4096, activation='relu', kernel_initializer='TruncatedNormal'))

    model.add(Dense(output_data.shape[1]))

    # Define Optimiser
    optimizer = Adam(lr)

    # Compile Model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model