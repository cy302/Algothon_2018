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
import warnings


warnings.filterwarnings("ignore")


class Time:
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        stop_dt = dt.datetime.now()
        print('Time taken %s' % (stop_dt-self.start_dt))


class data_load:

    def __init__(self, filename, training_rate):
        df = pd.read_csv(filename)
        ind_split = int(len(filename)*training_rate)
        self.train_data = df.get('mid-price').values[:ind_split]
        self.test_data = df.get('mid-price').values[ind_split:]
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        self.train_window_len = None

    def get_train_data(self, seq_len, normalise):
        train_x, train_y = [], []
        for i in range(self.train_len-seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            train_x.append(x)
            train_y.append(y)
        return np.array(train_x), np.array(train_y)

    def get_test_data(self, seq_len, normalise):
        data_window = []

        for i in range(self.train_len-seq_len):
            data_window.append(self.test_data[i:i+seq_len])

        data_window = np.array(data_window).astype(float)
        data_window = self.normalise_window(data_window, single_window=False) if normalise else data_window

        x = data_window[:, :-1]
        y = data_window[:, -1, [0]]

        return x, y

    def generate_train_batch(self, seq_len, batch_size, normalise):
        i = 0
        while i < (self.train_len - seq_len):
            batch_x, batch_y = [], []
            for b in range(batch_size):
                if i >= (self.train_len-seq_len):
                    yield batch_x, batch_y
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                batch_x.append(x)
                batch_y.append(y)
                i += 1
            yield np.array(batch_x), np.array(batch_y)

    def _next_window(self, i, seq_len, normalise):
        window = self.train_data[i:i+seq_len]
        window = self.normalise_window(window, single_window=True) if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_window(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for i in range(window.shape[1]):
                normalised = [((float(p) / float(window[0, i])) - 1) for p in window[:, i]]
                normalised_window.append(normalised)

            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)


class LSTM_model:

    def __init__(self):
        self.lstm_nn = Sequential()

    def load_model(self, filepath):
        print("Loading trained model from %s" % filepath)
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

            if layer['type'] == 'dense':
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
        print('LSTM network training starts, with %s epochs and batchsize %s' % (epochs, batch_size))
        sav_filename = os.path.join(sav_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

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
        sav_filename = os.path.join(sav_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

        callbacks = [ModelCheckpoint(filepath=sav_filename, monitor='loss', save_best_only=True)]

        self.lstm_nn.fit(x_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, workers=1)

        print('LSTM network training completed. Model saved as %s' % sav_filename)
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

    data = data_load(
        os.path.join('data', configs['data']['filename']),
        configs['data']['training_rate'],
    )

    lstm = LSTM_model()
    lstm.build_nn(configs)

    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=true
    )

    # model training when we feed the entire dataset into the workspace
    # lstm.train(x, y, epochs=configs['training']['epochs'], batch_size=configs['training']['batch_size'],
    #            sav_dir=configs['model']['sav_dir'])

    # suppose here we have the sequential feed of the input data, but please also write functions for generative
    # update
    steps_per_epoch = math.ceil((train_len - configs['data']['sequence_length'])/configs['training']['batch_size'])
    lstm.train_generator(x_gen=data.generate_train_batch(seq_len=configs['data']['sequence_length'],
                                                         batch_size=configs['training']['batch_size'],
                                                         normalise=True),
                         epochs=configs['model']['epochs'],
                         steps_per_epoch=steps_per_epoch,
                         batch_size=configs['model']['batch_size'],
                         sav_dir=configs['model']['sav_dir'])

    # here use your test data to test your model performance
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=True
    )

    predictions = lstm.predict_point(x_test)
    plot_predictions(predictions, y_test)


if __name__ == '__main__':
    main()