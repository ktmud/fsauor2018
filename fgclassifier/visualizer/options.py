#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Visualizer
"""
from collections import OrderedDict

dataset_choices = OrderedDict([
    ('train_en', 'Training'),
    ('valid_en', 'Validation'),
    ('test_en', 'Testing'),
    ('train', 'Training (Chinese)'),
    ('valid', 'Validation (Chinese)'),
    ('test', 'Testing (Chinese)'),
])

# Feature model choices -------
fm_choices = OrderedDict([
    ('count_en', 'Word Count'),
    ('count_en_sv', 'Word Count(SM)'),
    ('lsa_500_en', 'TF-IDF -> SVD(500)'),
    ('lsa_1k_en', 'TF-IDF -> SVD(1000)'),
    ('lsa_500_en_sv', 'TF-IDF(SM) -> SVD(500)'),
    ('tfidf_en_sv_dense', 'TF-IDF(SM)'),
    ('word2vec_en', 'Word2Vec'),
])

# Classifier choices ---------
clf_choices = OrderedDict([
    # Only LDA and Logistic supports predict a probability
    ('LDA', 'Linear Discriminant Analysis'),
    ('SGD_Logistic', 'Logistic Regression'),
    ('SGD_SVC', 'Linear SVM Classifier'),
    ('Ridge', 'Ridge Classifier'),
    ('ComplementNB', 'Complement Naive Bayes'),
])
