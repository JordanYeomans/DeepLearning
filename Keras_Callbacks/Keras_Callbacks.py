from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

# Model Save Path
def standard_callbacks(model_filepath = './Saved_models/latest_new_model.hdf5', patience = 10):
    #checkpoint = save_best(model_filepath)
    tensorboard = start_tensorboard()
    earlystopping = early_stopping(patience)

    return [tensorboard, earlystopping] #[checkpoint, tensorboard, earlystopping]

def save_best(model_filepath):
    return ModelCheckpoint(filepath = model_filepath, verbose = 1, save_best_only = True)

def start_tensorboard():
    # Command Line: tensorboard --logdir ./Graph
    return TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True)

def early_stopping(patience = 10):
    return EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')