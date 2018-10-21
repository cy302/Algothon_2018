import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
import datetime as dt
import os
import math
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import time

data = pd.read_csv('~/Downloads')

class deeper_neural_conv:
    def __init__(self, Nlayers, layer_dims, learning_rate, train_data, test_data):
        self.Nlayers = Nlayers
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.train_data = train_data
        self.test_data = test_data


    def neural_network(self):
        nn = keras.Sequential()
        for i in range(self.Nlayers-1):
            nn.add(keras.layers.Dense(self.layer_dims[i], activation=tf.nn.relu))
        nn.add(keras.layers.Dense(self.layer_dims[-1], activation=keras.activations.linear))
        nn.compile(optimizer=tf.train.AdamOptimizer, loss = keras.losses.mean_squared_error, metrics=['accuracy'])


def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

################################## LSTM neural network ###########################################

class Time:
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        stop_dt = dt.datetime.now()
        print('Time taken %s' %(stop_dt-self.start_dt))

def read_into_array(filename):
    data = pd.read_csv(filename)
    data_matrix = data.as_matrix()
    annotated_array = []
    for i in range(len(data_matrix)):
        a = np.array(data_matrix[i][0].split(' '))
        annotated_array.append(np.ndarray.tolist(np.delete(a, np.where(a=='')[0])))
    annotated_array = np.array(annotated_array)
    return(annotated_array)


# class loadingData:
#     def __init__(self, data, training_rate):
#         ind_split = int(len(data)*training_rate)
#         self.train_data = data[:ind_split, :]
#         self.test_data = data[ind_split:, :]
#         self.len_train = len(self.train_data)
#         self.len_test = len(self.test_data)
#
#     def normalise_window(self, window_data, single_window = False):
#         normalised_data = []
#         window_data = [window_data] if single_window else window_data
#         for window in window_data:
#             normalised_data = []
#             for i in range(window.shape[1]):
#                 normalised_col = [((float(p)/float(window[0])))]


class LSTM_model:

    def __init__(self):
        self.lstm_nn = Sequential()

    def load_model(self, filepath):
        print("Loading trained model from %s" %filepath)
        self.lstm_nn = load_model(filepath)

    def build_nn(self, configs):
        time = Time()
        time.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layers['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layers['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'Dense':
                self.lstm_nn.add(Dense(neurons, activation=activation))

            if layer['type'] == 'lstm':
                self.lstm_nn.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.lstm_nn.add(Dropout(neurons, dropout_rate=dropout_rate))

        self.lstm_nn.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimiser'])

        print('LSTM neural network finished compiling')
        time.stop()

    def train(self, x, y, epochs, batch_size, sav_dir):
        time = Time()
        time.start()
        print('LSTM network training starts, with %s epochs and batchsize %s' %(epochs, batch_size))
        sav_filename = os.path.join(sav_dir, '%s-e%s.h5' %(dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=sav_filename, monitor='val_loss', save_best_only=True)
        ]
        self.lstm_nn.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        self.lstm_nn.save(sav_filename)
        print('LSTM network training completed. Model saved as %s' %sav_filename)
        time.stop()

    def train_generator(self, x_gen, epochs, batch_size, steps_per_epoch, sav_dir):
        time = Time()
        time.start()
        print('LSTM network training starts, with %s epochs, %s batchsize and %s batched per epoch' %(epochs,
        batch_size, steps_per_epoch))
        sav_filename = os.path.join(sav_dir, '%s-e%s.h5' %(dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

        callbacks = [ModelCheckpoint(filepath=sav_filename, monitor='loss', save_best_only=True)]

        self.lstm_nn.fit(x_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, workers=1)

        print('LSTM network training completed. Model saved as %s' %sav_filename)
        time.stop()

    def predict_point(self, data):
        print('Trained LSTM network predictings sequentially')
        predictions = self.lstm_nn.predict(data)
        predictions = np.reshape(predictions, (predictions.size, ))
        return predictions

    def predict_full(self, data):
        curr_frame = data[0]
        prediction = []
        for i in range(len(data)):
            prediction.append(self.lstm_nn.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], prediction[-1], axis=0)
        return prediction


def plot_predictions(predictions, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True data')
    ax.plot(predictions, label='Prediction')
    plt.legend()
    plt.show()


def main():
    configs = json.load(open('configs.json', 'r'))
    if not os.path.exists(configs['model']['sav_dir']): os.mkdir(configs['model']['sav_dir'])

    # please provide a data loader for this particular task, depending on the RAM, create functions with either
    # complete import or generative sequential import

    # assume we have data X here

    data = ...

    lstm = LSTM_model()
    lstm.build_nn(configs)

    # suppose here we have the sequential feed of the input data, but please also write functions for generative
    # update
    steps_per_epoch = math.ceil((train_len - configs['data']['sequence_length'])/configs['training']['batch_size'])
    lstm.train_generator(x_gen=..., epochs=configs['model']['epochs'], steps_per_epoch=steps_per_epoch,
                         batch_size=configs['model']['batch_size'], sav_dir=configs['model']['sav_dir'])

    # here use your test data to test your model performance
    x_test, y_test = ..., ...
    predictions = lstm.predict_point(x_test)
    plot_predictions(predictions, y_test)

if __name__ == '__main__':
    main()










