from __future__ import print_function
from gensim.models import word2vec
import os
import numpy as np
import re
import itertools
from collections import Counter
from tensorflow import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM, Activation
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers
import datetime as dt
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import time
import warnings


def train_word2vec(s_matrix, v_dict, num_features, min_words, context):
    '''
    :param s_matrix:
    :param v_dict:
    :param num_features:
    :param min_words:
    :param context:
    :return:
    '''
    model_dir = 'model'
    model_name = "{:d}_features_{:d}minwords_{:d}context".format(num_features, min_words, context)
    model_name = os.path.join(model_dir, model_name)
    if os.path.exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model')
    else:
        num_workers = 2
        downsampling = 1e-3

        print('Training the model...')
        sentences = [[v_dict[w] for w in s] for s in s_matrix]
        embedding_model = word2vec.Word2Vec(sentences=sentences, workers=num_workers, size=num_features,
                                            min_count=min_words, window=context, sample=downsampling)

        embedding_model.init_sims(replace=True)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        embedding_model.save(model_name)

    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                            np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                       for key, word in enumerate(v_dict.items())}
    return embedding_weights, model_name


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_labels(positives, negatives):
    positive_examples = list(open(positives, 'r', encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negatives, 'r', encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    text = positive_examples+negative_examples
    text = [clean_str(s) for s in text]

    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [text, y]


def padding(sentences, pad = '<PAD>'):
    max_len = max([len(s) for s in sentences])
    padded_sentences = []
    for s in sentences:
        s += pad*(max_len-len(s))
        padded_sentences.append(s)
    return padded_sentences


def build_library(sentences):
    word_count = Counter(itertools.chain(*sentences))
    v_inv = [_[0] for _ in word_count.most_common()]
    v_dict = {v: i for i, v in enumerate(v_inv)}
    v_inv_dict = {i: v for v, i in v_dict.items()}
    return v_dict, v_inv_dict


def load_actual_data(positives, negatives):
    sentences, labels = load_data_labels(positives, negatives)
    padded_sentences = padding(sentences)
    v_dic, v_inv_dic = build_library(padded_sentences)
    x, y = np.array([[v_dic[i] for i in s] for s in sentences]), np.array(labels)
    return [x, y, v_dic, v_inv_dic]


def batch_iter(data, batch_size, epochs, shuffle=True):
    data = np.array(data)
    batches_per_epoch = int((len(data)-1)/batch_size)+1
    for i in range(epochs):
        if shuffle:
            shuffled_data = data[np.random.permutation(np.arange(len(data)))]
        else:
            shuffled_data = data
        for j in range(batches_per_epoch):
            yield shuffled_data[j*batch_size:min((j+1)*batch_size, len(data))]


############ LSTM network begins here ###########################################

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


##### turn sentiment vectors into predictions
class CNN:
    def __init__(self):
        self.model = None

    def build_cnn(self, Nlayers, num_neurons):
        self.model = Sequential()
        for i in range(Nlayers):
            self.model.add(Dense(num_neurons[i], activation='relu'))
        self.model.add(Dense(num_neurons[-1], activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, x, y, epochs=60, batch_size=10):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)


class logistic_regression:
    def __init__(self):
        self.model = None

    def model(self):
        self.model = Sequential([Dense(64, activation='sigmoid')])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, x, y, epochs=60, batch_size=10):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    model_type = 'CNN_non_static'

    embedding_dims = 50
    filter_size = (3, 7)
    num_filters = 16
    dropout_prob = (0.5, 0.8)
    hidden_neurons = 64

    batch_size = 60
    epochs = 10

    sequence_length = 4000
    max_words = 5000

    min_word = 1
    context = 10

    positives, negatives = ..., ...

    x, y, v_dict, v_inv_dict = load_actual_data(positives, negatives)
    y = y.argmax(axis=1)

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]

    if sequence_length != x_test.shape[1]:
        sequence_length = x_test.shape[1]

    if model_type == 'CNN_non_static':
        embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                           min_words=min_word_count, context=context)

    if model_type == 'CNN_static':
        input_shape = (sequence_length, embedding_dims)
    else:
        input_shape = (sequence_length,)

    model_input = Input(input_shape)

    if model_type == 'CNN_static':
        z = model_input
    else:
        z = Embedding(len(v_inv_dict), embedding_dims, input_length=sequence_length, name="embedding")(model_input)

    z = Dropout(dropout_prob[0])(z)

    conv_blocks = []
    for f in filter_size:
        conv = Convolution1D(filters=num_filters, kernel_size=f, padding='valid', activation='relu',
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_neurons, activation='relu')(z)

    model_output = Dense(1, activation='sigmoid')(z)

    model = Model(model_input, model_output)
    model.compile(optimzer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if model_type == "CNN-non-static":
        weights = np.array([v for v in embedding_weights.values()])
        embedding_layer = model.get_layer("embedding")
        embedding_layer.set_weights([weights])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)

    news_old = ...
    predictions_sentiment_vectors = model.predict(news_old)




#### LSTM main function start here
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
    # plot_predictions(predictions, y_test)

    cnn = CNN()
    cnn.build_cnn(5, [1024, 512, 256, 128, 1])
    cnn.fit(news_old, predictions_sentiment_vectors)
    predictions_sentiment_price_new = cnn.predict(news)
    predictions_sentiment_price_old = cnn.predict(news_old)

    old_lstm_prices = lstm.predict_point(x[:-5064])
    new_lstm_prices = lstm.predict_point(x[-5064:])

    logist = logistic_regression()
    logist.model()
    logist.fit(np.concatenate([old_lstm_prices, predictions_sentiment_price_old]), y[:-5064])
    logist.predict(np.concatenate([new_lstm_prices, predictions_sentiment_price_new]))

