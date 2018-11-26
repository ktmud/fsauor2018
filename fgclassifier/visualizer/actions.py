#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actions users can take
"""
from flask import request

from fgclassifier.visualizer.config import dataset_choices, fm_choices
from fgclassifier.visualizer.config import clf_choices
from fgclassifier.utils import get_dataset, load_model, read_data


def parse_inputs(dataset='train_en', keyword=None,
                 fm='tfidf', clf='lda', seed='42', **kwargs):
    """Predict sentiments for one single review"""
    if keyword is None or isinstance(keyword, str):
        keyword = [keyword]
    # DataFrames for all keywords
    keywords = keyword
    dfs = [get_dataset(dataset, keyword=x) for x in keywords]
    seed = int(seed) if seed.isdigit() else 42
    return dict(locals())


def pick_review(dfs, seed, **kwargs):
    """Pick a random review"""
    # A random review from the first keyword
    random_review = dfs[0].sample(1, random_state=seed)
    return {
        'review': random_review.to_dict('records')[0],
    }


def predict_one(dfs, seed, fm, clf, **kwargs):
    """Predict for a random review"""
    # get the random review
    random_review = dfs[0].sample(1, random_state=seed)
    # split to feature and labels
    X, y = read_data(random_review)
    model = load_model(fm, clf)
    probas = model.proba(model)
    return {
        'label_columns': y.columns, 
        'probas': probas
    }

