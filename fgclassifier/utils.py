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

from sklearn.externals import joblib

from tqdm import tqdm

logging.getLogger('jieba').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

jieba.enable_parallel(4)


def load_user_dict():
    fpath = os.path.join(CURRENT_PATH, 'jieba/user_dict.txt')
    for word in open(fpath).read().strip().split('\n'):
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
        filename = f'{filename.replace}.{flavor}.csv'
        if not os.path.exists(filename):
            return tokenize_csv(orig_filename, flavor=flavor)
    logger.info(f'Reading {filename}..')
    return pd.read_csv(filename, encoding='utf-8')


def tokenize_csv(filename, flavor='tokenized'):
    """Tokenize all data"""
    output = f'{filename}.{flavor}.tsv'
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
        df = data_path
    else:
        df = read_csv(data_path, flavor=flavor, **kwargs)
    if sample_n:
        df = df.sample(sample_n)
    # X is just 1D strings, y is 20-D labels
    X, y = df['content'], df.drop(['id', 'content'], axis=1)
    if return_df:
        return X, y, df.copy()
    return X, y


def save_model(model, filepath):
    """Save model to disk, so we can reuse it later"""
    logger.info("Saving model to %s..." % filepath)
    pathdir = os.path.dirname(filepath)
    if not os.path.exists(pathdir):
        os.makedirs(pathdir)
    joblib.dump(model, filepath)
    logger.info("Saving model... Done.")

