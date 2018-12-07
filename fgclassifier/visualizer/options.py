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
    ('count_en', 'Word Count(4K)'),
    ('count_en_sv', 'Word Count(2K)'),
    ('tfidf_en_sv', 'TF-IDF(4K)'),
    ('lsa_500_en', 'TF-IDF(4K) -> SVD(500)'),
    ('lsa_1k_en', 'TF-IDF(4K) -> SVD(1K)'),
    ('tfidf_en_sv', 'TF-IDF(2K)'),
    ('lsa_500_en_sv', 'TF-IDF(2K) -> SVD(500)'),
    ('lsa_1k_en_sv', 'TF-IDF(2K) -> SVD(1K)'),
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
