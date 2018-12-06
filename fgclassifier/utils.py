#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for reading data, etc.
"""
import os
import logging
import jieba
import csv
import pandas as pd
import numpy as np
import hashlib

import threading
from collections import defaultdict
from functools import lru_cache, _make_key

import config
from datetime import datetime
from collections import Counter
from sklearn.externals import joblib
from tqdm import tqdm

logging.getLogger('jieba').setLevel(logging.INFO)
logger = logging.getLogger(__name__)
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def threadsafe_lru(*cache_args, **cache_kwargs):
    cache_wrap = lru_cache(*cache_args, **cache_kwargs)

    def wrapper(func):
        func = cache_wrap(func)
        lock_dict = defaultdict(threading.Lock)

        def _thread_lru(*args, **kwargs):
            key = _make_key(args, kwargs, typed=False)
            with lock_dict[key]:
                return func(*args, **kwargs)

        return _thread_lru

    return wrapper


if os.name != 'nt':
    jieba.enable_parallel(4)


def load_user_dict():
    fpath = os.path.join(CURRENT_PATH, 'jieba/user_dict.txt')
    for word in open(fpath, encoding='utf-8').read().strip().split('\n'):
        jieba.add_word(word)
        jieba.suggest_freq(word, True)


load_user_dict()
stop_words = open(
    os.path.join(CURRENT_PATH, 'jieba/stop_words.txt')
).read().strip().split('\n')

# Blank space in Chinese can be meaningful,
# so we create a unique word to replace blankspaces
# BLANKSPACE = 'BBLANKK'
# NEWLINE = 'NNEWLINEE'
# jieba.add_word(BLANKSPACE)
# jieba.add_word(NEWLINE)


USELESS_CHARS = ' "\'\n\r\t【-#：:;/\\…●^'


def tokenize(s, rejoin_by=' ', HMM=True, flavor='tokenized'):
    """Tokenize text string"""
    if 'keep_case' not in flavor:
        # everything is case insensitive
        s = s.lower()
    if 'keep_useless' not in flavor:
        # Remove useless characters
        s = s.strip(USELESS_CHARS)
    tokens = [token for token in jieba.cut(s, HMM=HMM)
              if token not in stop_words and token.strip(USELESS_CHARS)]
    return rejoin_by.join(tokens)


def read_csv(filename, flavor=None):
    """Load data from CSV"""
    if flavor and flavor is not 'raw':
        orig_filename = filename
        filename = f'{filename}.{flavor}.csv'
        if not os.path.exists(filename):
            return tokenize_csv(orig_filename, flavor=flavor)
    logger.info(f'Reading {filename}..')
    return pd.read_csv(filename, encoding='utf-8')


def tokenize_csv(filename, flavor='tokenized'):
    """Tokenize all data"""
    output = f'{filename}.{flavor}.csv'
    if os.path.exists(output):
        logger.warn('Tokenized csv already exists. '
                    'Please use read_data() to read it.')
        return read_csv(output)

    df = read_csv(filename)
    logger.info(f'Segmenting {filename}..')
    df['content'] = [tokenize(s, flavor=flavor)
                     for s in tqdm(df['content'])]

    df.to_csv(output, index=False,
              quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    logger.info('Saved to {output}.')
    return df


def read_data(data_path, flavor='tokenized', return_df=False,
              sample_n=None, random_state=1, **kwargs):
    """Load data, return X, y"""
    if isinstance(data_path, pd.DataFrame):
        # flavor does not matter when passing in a DataFrame
        df = data_path
    else:
        df = read_csv(data_path, flavor=flavor, **kwargs)

    if sample_n:
        logger.info(
            f'Take {sample_n} samples with random state {random_state}')
        df = df.sample(sample_n, random_state=random_state)

    # X is just 1D strings, y is 20-D labels
    X, y = df['content'], df.drop(['id', 'content'], axis=1)

    # remove other columns, as well
    if 'content_raw' in y.columns:
        y = y.drop(['content_raw'], axis=1)

    if return_df:
        return X, y, df.copy()
    return X, y


@threadsafe_lru(maxsize=10)
def get_dataset(dataset, include_raw=True, keyword=None):
    """Get a dataset DataFrame, including the raw content and tokenized content
    and filter it by keywords"""
    data_path = getattr(config, f'{dataset}_data_path')
    if 'english' in data_path:
        df = read_csv(data_path)
        if include_raw or keyword:
            # For english, the raw content is the same
            df['content_raw'] = df['content']
    else:
        # if Chinese, needs to add a raw column
        df = read_csv(data_path, 'tokenized')
        if include_raw or keyword:
            df['content_raw'] = read_csv(data_path, 'raw')[
                'content'].str.strip('"')
    if keyword:
        df = df[df['content_raw'].str.contains(keyword, case=False)]
    return df


def persistent(subdir, storage_path=config.data_root, ttl=604800):
    """Save function output in disk"""
    outdir = os.path.join(storage_path, subdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    def wrapper(func):
        def wraped(*args):
            # cache for 1 week
            ts = str(round(datetime.now().timestamp() / ttl))
            key = '--'.join([*args, ts])
            # key = hashlib.md5(key.encode('utf-8')).hexdigest()[:7]
            fpath = os.path.join(outdir, f'{key}.pkl')
            if os.path.exists(fpath):
                logger.info('Load cache %s', fpath)
                return joblib.load(fpath)
            ret = func(*args)
            joblib.dump(ret, fpath)
            logger.info('Saved cache %s', fpath)
            return ret
        return wraped
    return wrapper


def label2dist(y):
    """y (nx20x4) labels to distributions (20x4)"""
    counts = [Counter(y[col]) for col in y.columns]
    counts = [[c.get(x, 0) for x in [-2, -1, 0, 1]] for c in counts]
    return (np.array(counts) / len(y)).tolist()


@threadsafe_lru(10)
@persistent('stats')
def get_stats(dataset, fm, clf):
    """Get performance stats of a model on the selected dataset"""
    X, y = read_data(get_dataset(dataset, include_raw=False))
    model = load_model(fm, clf)
    logging.info('Running stats for %s...', model.name)
    y_pred = pd.DataFrame(
        data=model.predict(X), index=y.index, columns=y.columns)
    ret = {
        'dataset': dataset,
        'fm': fm,
        'clf': clf,
        'predict_dist': label2dist(y_pred),
    }

    # if this is a test dataset, there is no true labels,
    # hence no scores
    if 'test' in dataset:
        return ret

    scores = model.scores(X, y)

    return {
        **ret,
        'true_dist': label2dist(y),
        'avg_score': np.mean(scores),
        'scores': scores,
    }


def save_model(model, model_save_path=config.model_save_path):
    """Save model to disk, so we can reuse it later"""
    filename = f'{model.name}.pkl'
    model_path = os.path.join(model_save_path, filename)
    logger.info("Saving model to %s..." % model_path)
    pathdir = os.path.dirname(model_path)
    if not os.path.exists(pathdir):
        os.makedirs(pathdir)
    joblib.dump(model, model_path)
    logger.info("Saving model... Done.")


@threadsafe_lru(maxsize=10)
def load_model(feature_model, classifier, model='Baseline',
               model_save_path=config.model_save_path):
    if model == 'Baseline':
        filename = f'{feature_model}_{classifier}.pkl'
    else:
        filename = f'{feature_model}_{model}_{classifier}.pkl'

    # if file does not exist, try load the SGD version
    if classifier in ('SVC', 'Logistic') and not os.path.exists(filename):
        filename = filename.replace(classifier, 'SGD_' + classifier)

    model_path = os.path.join(model_save_path, filename)
    logger.info('Loading model %s', filename)
    model = joblib.load(model_path)
    logger.info('Loading %s done.', filename)
    return model
