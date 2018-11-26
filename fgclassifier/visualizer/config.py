#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Visualizer
"""
from collections import OrderedDict

dataset_choices = OrderedDict([
    ('train_en', 'Training'),
    ('valid_en', 'Validation'),
    ('train', 'Training (Chinese)'),
    ('valid', 'Validation (Chinese)'),
])

# Feature model choices -------
fm_choices = OrderedDict([
    ('lsa_200_en', 'TF-IDF(min_df=0.01) -> SVD(200)'),
    ('lsa_500_en', 'TF-IDF(min_df=0.01) -> SVD(500)'),
    ('lsa_1k_en', 'TF-IDF(min_df=0.01) -> SVD(1000)'),
    ('tfidf_en_sv_dense', 'TF-IDF(min_df=0.02)'),
    ('lsa_200_en_sv', 'TF-IDF(min_df=0.02) -> SVD(200)'),
    ('lsa_500_en_sv', 'TF-IDF(min_df=0.02) -> SVD(500)'),
])

# Classifier choices ---------
clf_choices = OrderedDict([
    # Only LDA and Logistic supports predict a probability
    ('LDA', 'Linear Discriminant Analysis'),
    ('Logistic', 'Logistic Regression'),
    # ('LinearSVC', 'Linear SVM Classifier'),
    # ('Ridge', 'Ridge Classifier'),
])
