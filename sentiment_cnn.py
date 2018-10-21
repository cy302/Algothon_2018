import numpy as np
from nlp_algothon import train_word2vec
import data_load
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence

# np.random.seed(0)

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

positives, negatives = ..., ... # specify the directory for positive and negative words files


def load_data(filename):
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

    return x_train, y_train, x_test, y_test, v_inv_dict


x_train, y_train, x_test, y_test, v_inv_dict = load_data(filename="some shits")

if sequence_length != x_test.shape[1]:
    sequence_length = x_test.shape[1]


if model_type == 'CNN_non_static':
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
# elif model_type == 'CNN_static':
#     embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
#                                        min_word_count=min_word_count, context=context)
#     x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
#     x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
# elif model_type == 'CNN_rand':
#     embedding_weights = None
# else:
#     raise ValueError("Unknown model type")
#
if model_type == 'CNN_static':
    input_shape = (sequence_length, embedding_dims)
else:
    input_shape = (sequence_length, )

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




