#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actions users can take
"""
from flask import request, render_template

from fgclassifier.visualizer.config import dataset_choices, fm_choices
from fgclassifier.visualizer.config import clf_choices
from fgclassifier.utils import get_dataset, load_model, read_data


def parse_inputs(dataset='train_en', keyword=None,
                 fm='lsa_1k_en', clf='lda', seed='42', **kwargs):
    """Predict sentiments for one single review"""
    if keyword is None or isinstance(keyword, str):
        keyword = [keyword]
    # DataFrames for all keywords
    keywords = keyword

    # make sure values are valid
    if dataset not in dataset_choices:
        dataset = 'train_en'
    if fm not in fm_choices:
        fm = 'lsa_1k_en'
    if clf not in clf_choices:
        clf = 'LDA'

    dfs = [get_dataset(dataset, keyword=x) for x in keywords]  # filtered reviews
    totals = [df.shape[0] for df in dfs]  # total number of reviews
    seed = int(seed) if seed.isdigit() else 42
    return dict(locals())


def predict_one(dfs, totals, seed, fm, clf, **kwargs):
    """Predict for a random review"""
    # get the random review
    if totals[0] == 0:
        X, y = read_data(dfs[0])
        review = {
            'id': 'N/A',
            'content_html': '--  No matching reviews found. Please remove keyword. --'
        }
        true_labels, probas = None, None
    else:
        random_review = dfs[0].sample(1, random_state=seed)
        # split to feature and labels
        X, y = read_data(random_review)
        model = load_model(fm, clf)
        probas = model.predict_proba(X)
        review = random_review.to_dict('records')[0]
        # Add highlighted HTML's
        review['content_html'] = review['content_raw'].replace('\n', '<br>')
        true_labels = y.values.tolist()
        probas = [x.tolist() for x in probas]

    label_names = y.columns.tolist()
    return {
        'review': review,
        'label_names': label_names,
        'true_labels': true_labels,
        'probas': probas,
        'filter_results': render_template(
            'single/filter_results.jinja', **{**kwargs, **locals()})
    }

