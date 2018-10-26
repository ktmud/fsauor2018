#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for reading data, etc.
"""
import logging
import pandas as pd
import jieba

from sklearn.metrics import f1_score as f1_score_

logger = logging.getLogger(__name__)

def read_csv(file, *args, seg_words=True, sample_n=100, **kwargs):
    """Load data from CSV"""
    logger.info('Reading %s..', file)
    df = pd.read_csv(file, *args, **kwargs)
    if sample_n:
        df = df.head(sample_n)
    if seg_words:
        # Separate chinese words,
        # re-join with space,
        # remove extraneous quotes
        logger.info('Segmenting %s..', file)
        df['content'] = [' '.join(jieba.lcut(s.strip('"')))
                         for s in df['content']]
    return df

def f1_score(model, X, y):
    """Multi-class F1 score with Macro average"""
    return f1_score_(y, model.predict(X), average='macro')
