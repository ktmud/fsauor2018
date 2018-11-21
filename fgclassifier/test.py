#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load the model and run on test data
"""
import os
import argparse

from sklearn.externals import joblib
from fgclassifier import classifiers
from fgclassifier.features import FeaturePipeline
from fgclassifier.utils import read_data, read_csv

try:
    import local_config as config
except ImportError:
    import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    classifier_choices = [x for x in dir(classifiers) if not x.startswith('_')]
    parser.add_argument('-fm', '--feature-model', default='tfidf_sv',
                        help='Feature Model')
    parser.add_argument('-m', '--model', default='Baseline',
                        help='Basis of the classifiers')
    parser.add_argument('-c', '--classifier',
                        default='LinearDiscriminantAnalysis',
                        choices=classifier_choices,
                        help='Classifier used for each aspect')
    parser.add_argument('-t', '--test-set', default='testa',
                        choices=['testa', 'testb'],
                        help='Which test set to use')
    args = parser.parse_args()

    filename = f'{args.feature_model}_{args.model}_{args.classifier}.pkl'
    fm_filename = f'{args.feature_model}.pkl'
    model_fpath = os.path.join(config.model_save_path, filename)
    model = joblib.load(model_fpath)

    fpath_key = f'{args.test_set}_data_path'
    fpath_key_out = f'{args.test_set}_predict_out_path'
    fpath = getattr(config, fpath_key)
    fpath_out = getattr(config, fpath_key_out)
    model.predict_df(fpath, save_to=fpath_out)
