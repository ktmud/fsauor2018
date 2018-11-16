"""
Load data and train the model
"""
import os
import argparse
import logging

from sklearn.externals import joblib

from fgclassifier import models, classifiers
from fgclassifier.utils import read_data, save_model

try:
    import local_config as config
except ImportError:
    import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    classifier_choices = [x for x in dir(classifiers) if not x.startswith('_')]
    parser.add_argument('-m', '--model', default='Baseline',
                        help='Top-level model, including how to do '
                             'feature engineering')
    parser.add_argument('-c', '--classifier', default='SVC',
                        choices=classifier_choices,
                        help='Classifier used by the model')
    parser.add_argument('--train', default=10000,
                        help='Number of training sample to use')
    parser.add_argument('--valid', default=1000,
                        help='Number of validation sample to use')
    args = parser.parse_args()

    Model = getattr(models, args.model)
    Classifier = getattr(classifiers, args.classifier)
    model = Model(classifier=Classifier)
    X_train, Y_train = read_data(config.train_data_path, sample_n=args.train)
    X_valid, Y_valid = read_data(config.valid_data_path, sample_n=args.valid)

    with joblib.parallel_backend('threading', n_jobs=5):
        model.fit(X_train, Y_train)
        score = model.score(X_valid, Y_valid)
        logging.info('')
        logging.info(f'Overall F1: {score:.4f}')
        logging.info('')

    save_model(model, os.path.join(config.model_save_path, model.name + '.pkl'))
