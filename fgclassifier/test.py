#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load the model and run on test data
"""
import os
import argparse

from sklearn.externals import joblib
from fgclassifier import classifiers

try:
    import local_config as config
except ImportError:
    import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    classifier_choices = [x for x in dir(classifiers) if not x.startswith('_')]
    parser.add_argument('-c', '--classifier', default='LinearDiscriminantAnalysis',
                        choices=classifier_choices,
                        help='Classifier used for each aspect')
    parser.add_argument('-m', '--model', default='Indie',
                        help='Classifier used for each aspect')
    args = parser.parse_args()

    model_fpath = os.path.join(config.model_save_path, args.classifier + '.pkl')
    model = joblib.load(model_fpath)
    df_test = model.read(config.testa_data_path, sample_n=None)
    model.predict(df_test, save_to=config.testa_predict_out_path)
