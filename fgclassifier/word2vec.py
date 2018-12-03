#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word Embedding features
"""
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

# size: ncol of word vector
# min_count: the minimum appearence of a word be count in the model
# window: Maximum distance between the current and predicted word within a sentence.
# iter: Number of iterations (epochs) over the corpus.
DEFAULT_ZH_EMBEDDER = Word2Vec(size=256, window=10, min_count=5, iter=10)
DEFAULT_EN_EMBEDDER = Word2Vec(size=256, window=10, min_count=5, iter=10)

class Word2VecTransformer():

    def __init__(self, embedder=DEFAULT_ZH_EMBEDDER):
        self.embedder = embedder

    def fit(self, X):
        self.embedder.train(X)
        return self

    def transform(self, X):
        contents_vector = np.zeros((len(contents), size))
        for i in range(len(contents)):
            n = 0
            segs = contents[i].split()
            vector_sum = np.zeros(size)
            for j in range(len(segs)):
                try:
                    vector_sum = vector_sum + model[segs[j]]  # Sum word vector
                    n = n + 1
                except KeyError:
                    pass
            contents_vector[i] = vector_sum/n
        return contents_vector