#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for reading data, etc.
"""
import os
import logging
import pandas as pd
import jieba
from tqdm import tqdm

from sklearn.metrics import f1_score as f1_score_

logger = logging.getLogger(__name__)

# Blank space in Chinese can be meaningful,
# so we create a unique word to replace blankspaces
BLANKSPACE = 'BBLANKK'
NEWLINE = 'NNEWLINEE'
jieba.add_word(BLANKSPACE)
jieba.add_word(NEWLINE)

def read_csv(filename, *args, seg_words=True, sample_n=None, **kwargs):
    """Load data from CSV"""
    if seg_words:
        segged_file = '{}.segged_sample_{}.tsv'.format(filename, sample_n)
        if os.path.exists(segged_file):
            logger.info('Read cache %s..' % segged_file)
            df = pd.read_csv(segged_file, engine='python', sep='\t',
                             encoding='utf-8', keep_default_na=False)
            df['content'] = df['content'].fillna('N/A')
            return df

    logger.info('Reading %s..', filename)
    df = pd.read_csv(filename, *args, encoding='utf-8', **kwargs)
    if sample_n:
        logger.info('Pick a sample of %d', sample_n)
        df = df.head(sample_n)
    if seg_words:
        # Separate chinese words,
        # re-join with space,
        # remove extraneous quotes
        logger.info('Segmenting %s..', filename)
        df['content'] = [
            ' '.join(jieba.lcut(
                s.strip('"').replace(' ', BLANKSPACE).replace('\n', NEWLINE)
            ))
            for s in tqdm(df['content'])]
        df.to_csv(segged_file, index=False, sep='\t', encoding='utf-8')
        logger.info('Saved cached %s.', segged_file)
    return df

def f1_score(model, X, y):
    """Multi-class F1 score with Macro average"""
    return f1_score_(y, model.predict(X), average='macro')
