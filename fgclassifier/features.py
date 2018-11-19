#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipelines for building features
"""
import logging
import numpy as np
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation, SparsePCA

from fgclassifier.utils import ensure_named_steps, read_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DummyTransform():
    """Return content length as features.
    do nothing to the labels"""

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array(X['content'].str.len())[:, None]


class Tfidf(TfidfVectorizer):

    def fit(self, *args, **kwargs):
        logging.info('Fitting TF-IDF...')
        return super().fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        logging.info('Transforming TF-IDF...')
        return super().transform(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        logging.info('Fit & Transform TF-IDF...')
        return super().fit_transform(*args, **kwargs)


class SVD(TruncatedSVD):

    def fit(self, *args, **kwargs):
        logging.info('Fitting TruncatedSVD...')
        return super().fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        logging.info('Transforming TruncatedSVD...')
        return super().transform(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        logging.info('Fit & Transform TruncatedSVD...')
        return super().fit_transform(*args, **kwargs)


class SparseToSense():
    """Return content length as features.
    do nothing to the labels"""

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.toarray()


# ------- Feature Builder -----------------
class FeaturePipeline(Pipeline):

    # Default feature pipeline
    FEATURE_PIPELINE = [
        # from one of our sample, words like 味道 (taste), 好 (good)
        # are also in the most frequent words, but they are actually relevant
        # in terms of sentiment, so we keep max_df as 1.0
        ('tfidf', Tfidf(analyzer='word', ngram_range=(1, 5),
                        min_df=0.01, max_df=1.0, norm='l2')),
        # Must has some sort of dimension reduction, otherwise feature steps
        # will be really slow...
        ('reduce_dim', SVD(n_components=1000))
    ]

    def __init__(self, steps=None):
        steps = steps or self.FEATURE_PIPELINE
        super().__init__(ensure_named_steps(steps))


# ------- Additional helpers and basic pipelines ---------

def build_features(df_train, df_test, steps=None):
    X_train, y_train = read_data(df_train)
    X_test, y_test = read_data(df_test)
    feature = FeaturePipeline(steps)
    X_train = feature.fit_transform(X_train)
    X_test = feature.transform(X_test)
    return X_train, y_train, X_test, y_test

