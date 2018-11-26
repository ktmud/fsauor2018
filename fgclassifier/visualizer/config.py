#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Visualizer
"""
from collections import OrderedDict

dataset_choices = OrderedDict([
    ('train_en', 'Training (8,000 reviews)'),
    ('valid_en', 'Validation (2,000 reviews)'),
    ('train', 'Chinese Training (100,000)'),
    ('valid', 'Chinese Validation (20,000)'),
])

# Feature model choices -------
fm_choices = OrderedDict([
    ('tfidf', 'TF-IDF (min_df=0.001)'),
    ('tfidf_sv', 'TF-IDF (min_df=0.02)'),
])

# Classifier choices ---------
clf_choices = OrderedDict([
    ('LinearDiscriminantAnalysis', 'Linear Discriminant Analysis'),
    ('Logistic', 'Logistic Regression'),
    ('Ridge', 'Ridge Classifier'),
])
