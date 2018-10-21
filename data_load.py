import numpy as np
import re
import itertools
from collections import Counter


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


