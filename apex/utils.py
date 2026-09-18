"""APEX-pathogen helpers (vendored). Trimmed to the inference-time functions only.

Original imports scipy/scikit-learn for training utilities; those are unused at inference, so
they are dropped here to keep the vendored environment minimal.
"""

import csv

import numpy as np


def make_vocab():
    # 0: pad, 1: start, 2: end
    word2idx = {"0": 0, "1": 1, "2": 2}
    for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY", start=3):
        word2idx[aa] = i
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def AAindex(path, word2idx):
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        AAindex_dict = {}
        AAindex_matrix = []
        skip = 1
        for row in reader:
            if skip == 1:
                skip = 0
                header = np.array(row)[1:].tolist()
                continue
            tmp = []
            for j in np.array(row)[1:]:
                try:
                    tmp.append(float(j))
                except ValueError:
                    tmp.append(0)
            AAindex_matrix.append(np.array(tmp))
        dim = np.shape(AAindex_matrix)[0]
        AAindex_matrix = np.array(AAindex_matrix)
        for i in range(len(header)):
            AAindex_dict[header[i]] = AAindex_matrix[:, i]
    emb = np.zeros((len(word2idx), dim))
    for key, value in word2idx.items():
        if key in AAindex_dict:
            emb[value] = AAindex_dict[key]
    return emb, AAindex_dict


def onehot_encoding(seq_list_, max_len, word2idx):
    # 0: pad, 1: start, 2: end
    seq_list = [i for i in seq_list_]
    X = np.zeros((len(seq_list), max_len)).astype(int)
    for i in range(len(seq_list)):
        if len(seq_list[i]) >= max_len - 2:
            a_seq = "1" + seq_list[i][:max_len - 2].upper() + "2"
        else:
            a_seq = "1" + seq_list[i].upper() + "2"
        iter_num = max_len if len(a_seq) > max_len else len(a_seq)
        for j in range(iter_num):
            if a_seq[j] in word2idx:
                X[i, j] = word2idx[a_seq[j]]
    return np.array(X)
