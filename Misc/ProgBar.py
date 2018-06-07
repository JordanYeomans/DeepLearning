import tensorflow.contrib.keras as keras
import time

def new_prog_bar():
    return keras.utils.Progbar(target=1)

def update_prog_bar(progress_bar, update):
    progress_bar.update(update)