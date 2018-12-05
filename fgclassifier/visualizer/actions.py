#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actions users can take
"""
import numpy as np
import pandas as pd

from flask import request, render_template
from collections import Counter

from fgclassifier.visualizer.options import dataset_choices, fm_choices
from fgclassifier.visualizer.options import clf_choices
from fgclassifier.visualizer.highlight import highlight_noun_chunks
from fgclassifier.utils import get_dataset, load_model, read_data


def parse_inputs(dataset='train_en', keyword=None,
                 fm='lsa_1k_en', clf='lda', seed='49', **kwargs):
    """Predict sentiments for one single review"""
    # make sure values are valid
    if dataset not in dataset_choices:
        dataset = 'train_en'
    if fm not in fm_choices:
        fm = 'lsa_1k_en'
    if clf not in clf_choices:
        clf = 'LDA'

    # handle language
    #   - if dataset is not English
    #   - remove _en from model names
    if '_en' not in dataset:
        dataset = dataset.replace('_en', '')
        fm = fm.replace('_en', '')
        print(fm)

    if keyword is None or isinstance(keyword, str):
        keyword = [keyword]
    # DataFrames for all keywords
    keywords = keyword

    dfs = [get_dataset(dataset, keyword=x) for x in keywords]  # filtered reviews
    totals = [df.shape[0] for df in dfs]  # total number of reviews
    seed = int(seed) if seed.isdigit() else 42
    return dict(locals())


def predict_proba(clf, model, X):
    if clf in ('Logistic', 'LDA', 'SDG_Logistic'):
        # Only LogisticRegression and LinearDiscriminantAnalysis supports
        # the probablisitc view.
        probas = model.predict_proba(X)
        probas = [x.tolist() for x in probas]
    else:
        probas = None
    return probas


def predict_one(dataset, dfs, totals, seed, fm, clf, **kwargs):
    """Predict for a random review"""
    lang = 'en' if '_en' in dataset else 'zh'
    X, y = read_data(dfs[0])
    if totals[0] == 0:
        review = {
            'id': 'N/A',
            'content_html': '--  No matching reviews found. Please remove keyword. --'
        }
        true_labels, probas = None, None
    else:
        # get a random review
        random_review = dfs[0].sample(1, random_state=seed)
        # split to feature and labels
        X, y = read_data(random_review)
        model = load_model(fm, clf)
        review = random_review.to_dict('records')[0]
        # Add highlighted HTML's
        review['content_html'] = highlight_noun_chunks(
            review['content_raw'], lang
        ).replace('\n', '<br>')
        del review['content_raw']
        probas = predict_proba(clf, model, X)
        
        true_labels = y.values
        predict_labels = model.predict(X)
        # number of correct predictions
        n_correct_labels = np.sum(true_labels == predict_labels,
                                  axis=1).tolist()
        true_labels = true_labels.tolist()
        predict_labels = predict_labels.tolist()
        true_label_counts = [Counter(x) for x in true_labels]
        predict_label_counts = [Counter(x) for x in predict_labels]

    label_names = y.columns.tolist()
    n_total_labels = len(label_names)  # number of labels to predict
    return {
        'review': review,
        'label_names': label_names,
        'n_total_labels': n_total_labels,
        'n_correct_labels': n_correct_labels,
        'n_correct_labels_html': render_template(
            'single/correct_count.jinja', **locals()
        ),
        'true_label_counts': true_label_counts,
        'predict_label_counts': predict_label_counts,
        'true_labels': true_labels,
        'predict_labels': predict_labels,
        'probas': probas,
        'filter_results': render_template(
            'single/filter_results.jinja', **{**kwargs, **locals()})
    }


def predict_text(text, fm, clf, **kwargs):
    """Predict for user inputed text"""
    X = pd.Series([text], name='content')
    model = load_model(fm, clf)
    predict_labels = model.predict(X)
    probas = predict_proba(clf, model, X)
    predict_labels = predict_labels.tolist()
    predict_label_counts = [Counter(x) for x in predict_labels]
    return {
        'predict_label_counts': predict_label_counts,
        'predict_labels': predict_labels,
        'probas': probas,
    }
