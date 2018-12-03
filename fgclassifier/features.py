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
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation as LatentDirichlet

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
        logging.info('Fit & Transform TF-IDF...')
        return super().fit_transform(*args, **kwargs)


class SVD(TruncatedSVD):

    def fit_transform(self, *args, **kwargs):
        logging.info('Fit & Transform TruncatedSVD...')
        return super().fit_transform(*args, **kwargs)


class Count(CountVectorizer):

    def fit_transform(self, raw_documents, y=None):
        logging.info(f'Fit & Transform CountVectorizer...')
        ret = super().fit_transform(raw_documents, y=y)
        logging.info(f'Vocab Size: {len(self.vocabulary_)}')
        return ret


class SparseToDense(BaseEstimator):
    """Return content length as features.
    do nothing to the labels"""

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.toarray()


class OverSample(BaseEstimator):

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)


# ------- Feature Builder -----------------
def is_list_or_tuple(obj):
    return isinstance(obj, tuple) or isinstance(obj, list)


# Feature model specifications
# For Chinese
fm_spec = {
    'count': Count(ngram_range=(1, 5), min_df=0.001, max_df=0.99),
    'tfidf': ['count', Tfidf()],
    'lsa_200': ['tfidf', SVD(n_components=200)],
    'lsa_500': ['tfidf', SVD(n_components=500)],
    'lsa_1k': ['tfidf', SVD(n_components=1000)],

    # smaller vocabulary (removed more stop and infrequent words)
    'count_sv': Count(ngram_range=(1, 5), min_df=0.02, max_df=0.99),
    'tfidf_sv': ['count_sv', Tfidf()],
    'tfidf_sv_dense': ['tfidf_sv', SparseToDense()],
    'lsa_200_sv': ['tfidf_sv', SVD(n_components=200)],
    'lsa_500_sv': ['tfidf_sv', SVD(n_components=500)],

    # For English
    'count_en': Count(ngram_range=(1, 4), min_df=0.01, stop_words='english'),
    'tfidf_en': ['count_en', Tfidf()],
    'tfidf_en_dense': ['tfidf_en', SparseToDense()],
    'lsa_200_en': ['tfidf_en', SVD(n_components=200)],
    'lsa_500_en': ['tfidf_en', SVD(n_components=500)],
    'lsa_1k_en': ['tfidf_en', SVD(n_components=1000)],

    'count_en_sv': Count(ngram_range=(1, 4), min_df=0.02, stop_words='english'),
    'tfidf_en_sv': ['count_en_sv', Tfidf()],
    'tfidf_en_sv_dense': ['tfidf_en_sv', SparseToDense()],
    'lsa_200_en_sv': ['tfidf_en_sv', SVD(n_components=200)],
    'lsa_500_en_sv': ['tfidf_en_sv', SVD(n_components=500)],
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

    def fit_transform(self, X, y=None, **fit_params):
        """Fit transform the training data and save the results in cache"""
        cache_name = self._final_estimator_name
        cache = self.cache
        if cache and 'train' in cache:
            logger.info(f'  {cache_name}: fit_transform use cache.')
            return cache['train']
        Xt = super().fit_transform(X, y, **fit_params)
        if cache is not None:
            cache['train'] = Xt
        return Xt

    def transform(self, X):
        """Transform the testing data and save the results in cache"""
        cache_name = self._final_estimator_name
        cache = self.cache
        if cache and 'test' in cache:
            logger.info(f'  {cache_name}: transform use cache.')
            return cache['test']
        Xt = super().transform(X)
        if cache is not None:
            cache['test'] = Xt
        return Xt


# ------- Additional helpers and basic pipelines ---------

def build_features(X_train, X_test, steps='tfidf_sv', spec=fm_spec, **kwargs):
    # if provided both training and testing dataset
    # otherwise, load it from cache
    feature = FeaturePipeline(steps, spec=spec, **kwargs)
    X_train = feature.fit_transform(X_train)
    X_test = feature.transform(X_test)
    return X_train, X_test

