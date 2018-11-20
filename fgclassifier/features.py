#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipelines for building features
"""
import logging
import numpy as np
import re

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

from fgclassifier.utils import read_data

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


class Tfidf(TfidfTransformer):

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
def is_list_or_tuple(obj):
    return isinstance(obj, tuple) or isinstance(obj, list)


def ensure_named_steps(steps, spec=None, cache=None):
    """make sure steps are named tuples.
    Also handles dependencies in steps.
    """
    if not isinstance(steps, list):
        steps = [steps]
    # make a copy of the steps
    if is_list_or_tuple(steps):
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

    # Default feature pipeline
    FEATURE_PIPELINE = [
        # from one of our sample, words like 味道 (taste), 好 (good)
        # are also in the most frequent words, but they are actually relevant
        # in terms of sentiment, so we keep max_df as 1.0
        ('count', CountVectorizer(analyzer='word', ngram_range=(1, 5),
                                  min_df=0.01, max_df=1.0)),
        ('tfidf', TfidfTransformer(norm='l2')),
        # Must has some sort of dimension reduction, otherwise feature steps
        # will be really slow...
        ('reduce_dim', SVD(n_components=1000))
    ]

    def __init__(self, steps=None, spec=None, cache=None, **kwargs):
        steps = steps or self.FEATURE_PIPELINE
        steps = ensure_named_steps(steps, spec, cache)
        super().__init__(steps, **kwargs)
        # if speficied cache, save self to cache
        if cache is not None:
            self.cache = cache[self._final_estimator_name]
            self.cache['model'] = self

    @property
    def _final_estimator_name(self):
        return self.steps[-1][0]

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


# -------- For Word Embeddings ---------
RE_EXCL = re.compile('！+')
RE_QUES = re.compile('？+')

def split_by(s, regexp, char):
    if char in s.strip(char):
        tmp = regexp.split(s)
        last = tmp.pop()
        ret = [x + char for x in tmp]
        ret.append(last)  # add last sentence back
        return ret


def article_to_sentences(articles, split_sentence=True):
    sentences, aids, slens = [], [], []
    for aid, article in enumerate(articles):
        if not split_sentence:
            tokens = article.split()
            sentences.append(tokens)
            aids.append(aid)
            slens.append(len(tokens))
            continue
        ss = article.split('。')
        while ss:
            s = ss.pop(0).strip()
            if not s:
                continue
            tmp = split_by(s, RE_EXCL, '！')
            if tmp:
                ss = tmp + ss
                continue
                
            tmp = split_by(s, RE_QUES, '？')
            if tmp:
                ss = tmp + ss
                continue
                
            tokens = s.split()
            sentences.append(tokens)
            # keep a record of article ids and sentence length
            # so that we know which sentence/word belongs to
            # which article
            aids.append(aid)
            slens.append(len(tokens))
    return sentences, aids, slens


# ------- Additional helpers and basic pipelines ---------

def build_features(df_train, df_test, steps=None):
    X_train, y_train = read_data(df_train)
    X_test, y_test = read_data(df_test)
    feature = FeaturePipeline(steps)
    X_train = feature.fit_transform(X_train)
    X_test = feature.transform(X_test)
    return X_train, y_train, X_test, y_test
