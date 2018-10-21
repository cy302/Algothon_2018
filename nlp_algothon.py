from __future__ import print_function
from gensim.models import word2vec
import os
import numpy as np
import re
import itertools
from collections import Counter


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


if __name__ == '__main__':
    import data_load

    x, _, _, v_inv_dict = data_helpers.load_data()
    w, model_name = train_word2vec(x, v_inv_dict, num_features=300, min_words=1, context=10)
    model_word2vec = word2vec.Word2Vec.load(model_name)
    output_vec = [w[s] for s in sentences]
