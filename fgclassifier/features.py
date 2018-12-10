#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipelines for building features
"""
import logging
import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

from fgclassifier.embedding import Text2Tokens, W2VTransformer
from fgclassifier.embedding import tokenize_zh, tokenize_en


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DummyTransform(BaseEstimator):
    """Return content length as features.
    do nothing to the labels"""

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array(X.str.len())[:, None]


class Tfidf(TfidfTransformer):

    def fit_transform(self, *args, **kwargs):
        logger.debug('Fit & Transform TF-IDF...')
        return super().fit_transform(*args, **kwargs)


class SVD(TruncatedSVD):

    def fit_transform(self, *args, **kwargs):
        logger.debug('Fit & Transform TruncatedSVD...')
        return super().fit_transform(*args, **kwargs)


class SmartSVD(TruncatedSVD):

    def __init__(self, p_components):
        self.p_components = p_components
    
    def fit_transform(self, X, y=None, **kwargs):
        if self.p_components == 1:
            return X
        self.n_components = int(X.shape[1] * self.p_components)
        logger.info('SVD feature size %s', self.n_components)
        return super().fit_transform(X, y, **kwargs)


class Count(CountVectorizer):

    def transform(self, raw_documents):
        logger.debug(f'Transform with CountVectorizer...')
        ret = super().transform(raw_documents)
        logger.debug('Vocab Size: %s', len(self.vocabulary_))
        return ret

    def fit_transform(self, raw_documents, y=None):
        logger.debug(f'Fit & Transform CountVectorizer...')
        ret = super().fit_transform(raw_documents, y=y)
        logger.info('Vocab Size: %s', len(self.vocabulary_))
        return ret


class CountChinese(Count):

    def fit_transform(self, raw_documents, y=None):
        # a precomputed vocabulary should make things
        self.vocabulary = set(' '.join(raw_documents).split())
        self._validate_vocabulary()
        return super().fit_transform(raw_documents, y=y)


class SparseToDense(BaseEstimator):
    """Return content length as features.
    do nothing to the labels"""

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.toarray()


# ------- Feature Builder -----------------
def is_list_or_tuple(obj):
    return isinstance(obj, tuple) or isinstance(obj, list)


# Feature model specifications
# For Chinese
fm_spec = {
    'hashing': HashingVectorizer(tokenizer=tokenize_zh),
    'count': Count(ngram_range=(1, 5), min_df=5, max_df=0.9,
                   max_features=4000, tokenizer=tokenize_zh),
    'tfidf': ['count', Tfidf()],
    'tfidf_dense': ['tfidf', SparseToDense()],
    'lsa_200': ['tfidf', SVD(n_components=200)],
    'lsa_500': ['tfidf', SVD(n_components=500)],
    'lsa_1k': ['tfidf', SVD(n_components=1000)],
    'lsa_500_minmax': ['lsa_500', MinMaxScaler()],
    'lsa_1k_minmax': ['lsa_1k', MinMaxScaler()],

    # smaller vocabulary (removed more stop and infrequent words)
    'count_sv': Count(ngram_range=(1, 5), min_df=5, max_df=0.6,
                      max_features=2000, tokenizer=tokenize_zh),
    'tfidf_sv': ['count_sv', Tfidf()],
    'tfidf_sv_dense': ['tfidf_sv', SparseToDense()],
    'tfidf_sv_minmax': ['tfidf_sv_minmax', MinMaxScaler()],
    'lsa_200_sv': ['tfidf_sv', SVD(n_components=200)],
    'lsa_500_sv': ['tfidf_sv', SVD(n_components=500)],
    'lsa_1k_sv': ['tfidf_sv', SVD(n_components=1000)],
    'lsa_500_sv_minmax': ['lsa_500_sv', MinMaxScaler()],
    'lsa_1k_sv_minmax': ['lsa_1k_sv', MinMaxScaler()],

    'count_tiny': Count(ngram_range=(1, 5), min_df=0.03, max_df=0.6,
                        tokenizer=tokenize_zh),
    'tfidf_tiny': ['count_tiny', Tfidf()],
    'tfidf_tiny_dense': ['tfidf_tiny', SparseToDense()],
    'tfidf_tiny_minmax': ['tfidf_tiny_minmax', MinMaxScaler()],
    'lsa_200_tiny': ['tfidf_tiny', SVD(n_components=200)],
    'lsa_500_tiny': ['tfidf_tiny', SVD(n_components=500)],

    'word2vec': [Text2Tokens(),
                 W2VTransformer(size=300, min_count=5, max_vocab_size=50000,
                                sample=0.5, window=10, iter=10)],
    'word2vec_minmax': ['word2vec_minmax', MinMaxScaler()],

    'word2vec_en': [Text2Tokens(tokenizer=tokenize_en),
                    W2VTransformer(size=300, min_count=3, window=10, iter=10)],
    'word2vec_en_minmax': ['word2vec_en', MinMaxScaler()],

    # For English
    'count_en': Count(ngram_range=(1, 6), min_df=3, stop_words='english',
                      max_features=4000),
    'tfidf_en': ['count_en', Tfidf()],
    'tfidf_en_dense': ['tfidf_en', SparseToDense()],
    'lsa_500_en': ['tfidf_en', SVD(n_components=500)],
    'lsa_1k_en': ['tfidf_en', SVD(n_components=1000)],

    'count_en_sv': Count(ngram_range=(1, 6), min_df=3, stop_words='english',
                         max_features=2000),
    'tfidf_en_sv': ['count_en_sv', Tfidf()],
    'tfidf_en_sv_dense': ['tfidf_en_sv', SparseToDense()],
    'lsa_500_en_sv': ['tfidf_en_sv', SVD(n_components=500)],
    'lsa_1k_en_sv': ['tfidf_en_sv', SVD(n_components=1000)],
}


def ensure_named_steps(steps, spec=fm_spec, cache=None):
    """make sure steps are named tuples.
    Also handles dependencies in steps.
    """
    if not isinstance(steps, list):
        steps = [steps]

    # make a copy of the steps
    if isinstance(steps, list):
        steps = steps.copy()
    elif isinstance(steps, tuple):
        steps = list(steps)

    steps_ = []

    # while steps is not empty
    while steps:
        name, estimator = None, steps.pop(0)
        if isinstance(estimator, str):
            # if string, look it up from cache or spec
            if cache and estimator in cache:
                # if in cache, return cache
                name, estimator = estimator, cache[estimator]['model']
            else:
                # otherwise resolve spec
                name, estimator = estimator, spec[estimator]
        elif is_list_or_tuple(estimator):
            # when estimator has name already, expand it
            name, estimator = estimator

        # If already a Pipeline, name it with the last step
        if isinstance(estimator, Pipeline):
            name = estimator.steps[-1][0]

        # if is an array in cache
        if isinstance(estimator, list):
            # make sure current name is used for the last step
            # in the cached spec
            if not isinstance(estimator[-1], tuple):
                estimator[-1] = (name, estimator[-1])
            # add back to list
            steps = estimator + steps
            continue

        # Initialize estimator if necessary
        if callable(estimator):
            estimator = estimator()

        # if still haven't figured out step name
        if name is None:
            # get the name from class name
            name = estimator.__class__.__name__

        steps_.append((name, estimator))
    return steps_


def fit_cache(action_name):
    """A decorator to enable cache"""

    def wrapper(func):

        def wrapped(self, X, *args, **kwargs):
            cache_name = self._final_estimator_name
            cache = self.cache
            cachekey = f'{action_name}_{len(X)}_i{X.index[0]}'

            if cache and cachekey in cache:
                logger.info(f'  {cache_name}: {action_name} use cache.')
                return cache[cachekey]

            ret = func(self, X, *args, **kwargs)
            if cache is not None:
                cache[cachekey] = ret
            return ret

        return wrapped

    return wrapper


class FeaturePipeline(Pipeline):
    """
    FeaturePipeline with spec and cache support.

    Usage:

        fm_spec = {
            'count': CountVectorizer(ngram_range=(1, 4), min_df=0.01,
                                     max_df=0.99),
            'tfidf': ['count', TfidfTransformer],
        }
        fm = defaultdict(dict)
        model = FeaturePipeline('tfidf', spec=fm_spec, cache=fm)
        model.fit_transform(X_train)
        model.transform(X_test)

    Generates:

        > fm['tfidf']
            {'model': FeaturePipeline(...),
            'train': numpy.array,
            'test': numpy.array}
        > fm['count']
            {'model': FeaturePipeline(...), ...}

    Parameters
    ----------
        spec:   a dictionary of specs matching count to id
        cache:  a defaultdict to store estimator and train/test results
    """

    @classmethod
    def from_spec(cls, name, spec=fm_spec, cache=None, **kwargs):
        if cache is not None and name in cache:
            return cache[name]['model']
        return cls(name, spec, cache, **kwargs)

    def __init__(self, steps='tfidf_sv', spec=fm_spec, cache=None, **kwargs):
        self._steps = steps  # steps name

        steps = ensure_named_steps(steps, spec, cache)
        super().__init__(steps, **kwargs)

        # if speficied cache, save self to cache
        if cache is not None:
            self.cache = cache[self._final_estimator_name]
            self.cache['model'] = self
        else:
            self.cache = None

    def __repr__(self):
        return f'FeaturePipeline(steps={self._steps})'

    @property
    def _final_estimator_name(self):
        return self.steps[-1][0]
    
    @property
    def name(self):
        """Feature model name is just the final estimator name"""
        return self._final_estimator_name

    @fit_cache('fit_transform')
    def fit_transform(self, X, y=None, **fit_params):
        """Fit transform the training data and save the results in cache"""
        return super().fit_transform(X, y, **fit_params)

    @fit_cache('transform')
    def transform(self, X):
        """Transform the testing data and save the results in cache"""
        return super().transform(X)


# ------- Additional helpers and basic pipelines ---------

def build_features(X_train, X_test, steps='tfidf_sv', spec=fm_spec, **kwargs):
    # if provided both training and testing dataset
    # otherwise, load it from cache
    feature = FeaturePipeline(steps, spec=spec, **kwargs)
    X_train = feature.fit_transform(X_train)
    X_test = feature.transform(X_test)
    return X_train, X_test

