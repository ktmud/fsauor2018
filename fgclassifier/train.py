"""
Load data and train the model
"""
import os
import argparse
import logging
import numpy as np
import time

from collections import defaultdict
import joblib
from fgclassifier import models, classifiers
from fgclassifier.features import fm_spec
from fgclassifier.models import Baseline
from fgclassifier.utils import read_data, save_model, get_dataset

logger = logging.getLogger(__name__)


def fm_cross_check(fmns, clss, fm_cache=None, X_train=None, X_test=None,
                   y_train=None, y_test=None, model_cls=Baseline, results={}):
    """Feature Model Cross Check"""
    avg_test_scores = results['test_avg'] = results.get(
        'test_avg', defaultdict(dict))
    test_scores = results['test'] = results.get('test', {})
    avg_train_scores = results['train_avg'] = results.get(
        'train_avg', defaultdict(dict))
    train_scores = results['train'] = results.get('train', defaultdict(dict))
    # save modes as well
    models = results['models'] = results.get('models', defaultdict(dict))
    train_time = results['train_time'] = results.get(
        'train_time', defaultdict(dict))
    test_time = results['test_time'] = results.get(
        'test_time', defaultdict(dict))

    # Test for all Feature models
    for fmn in fmns:
        logger.info('')
        logger.info(f'============ Feature Model: {fmn} ============')
        logger.info('')
        cache = fm_cache[fmn]
        # Test on all major classifiers
        for clf in clss:
            tick = time.time()
            logger.info(f'Train for {fmn} -> {clf}...')

            Classifier = getattr(classifiers, clf)
            model = model_cls((clf, Classifier), fm=cache['model'])
            model.fit(X_train, y_train)

            train_scores[fmn][clf] = model.scores(X_train, y_train)
            train_f1 = avg_train_scores[fmn][clf] = np.mean(
                train_scores[fmn][clf])

            train_time[fmn][clf] = time.time() - tick
            tick = time.time()

            test_scores[fmn][clf] = model.scores(X_test, y_test)
            test_f1 = avg_test_scores[fmn][clf] = np.mean(
                test_scores[fmn][clf])
            test_time[fmn][clf] = time.time() - tick

            logger.info(
                '-------------------------------------------------------')
            logger.info(
                f'【{fmn} -> {clf}】 Train: {train_f1:.4f}, Test: {test_f1:.4f}')
            logger.info(
                '-------------------------------------------------------')
            models[model.name] = model

    return results


def train_and_save(train_file: str, valid_file: str, train_sample: int,
                   valid_sample: int, model: str, feature_model: str, classifier: str):
    Model = getattr(models, model)
    train_file, valid_file = 'train', 'valid'
    df_train = get_dataset(train_file)
    df_valid = get_dataset(valid_file)
    n_train_total, n_valid_total = df_train.shape[0], df_valid.shape[0]
    n_train_sample, n_valid_sample = train_sample, valid_sample
    if n_train_sample > n_train_total:
        logging.warning(f'Training sample size ({n_train_sample}) cannot be '
                        f'larger than the training dataset (n={n_train_total:,d}).')
        n_train_sample = n_train_total
    if n_valid_sample > n_valid_total:
        logging.warning(f'Validation sample size ({n_valid_sample}) cannot be '
                        f'larger than the validation dataset (n={n_valid_total:,d}).')
        n_valid_sample = n_valid_total
    X_train, Y_train = read_data(df_train, sample_n=n_train_sample)
    X_valid, Y_valid = read_data(df_valid, sample_n=n_valid_sample)
    model = Model(classifier=classifier, steps=[feature_model],
                  memory='data/feature_cache')

    with joblib.parallel_backend('threading', n_jobs=4):
        model.fit(X_train, Y_train)
        score = model.score(X_valid, Y_valid)
        logging.info('')
        logging.info(f'Overall F1: {score:.4f}')
        logging.info('')
    save_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    classifier_choices = [x for x in dir(classifiers) if not x.startswith('_')]
    parser.add_argument('-m', '--model', default='Baseline',
                        help='Top-level model, the basis for classifiers.')
    parser.add_argument('-fm', '--feature-model', default='tfidf_sv',
                        choices=fm_spec.keys(),
                        help='Which model to use for feature engineering')
    parser.add_argument('-c', '--classifier', default='SVC',
                        choices=classifier_choices,
                        help='Classifier used by the model')
    parser.add_argument('--train-file', default='train')
    parser.add_argument('--valid-file', default='valid')
    parser.add_argument('--train-sample', default=5000,
                        help='Number of training sample to use')
    parser.add_argument('--valid-sample', default=1000,
                        help='Number of validation sample to use')
    args = parser.parse_args()
    train_and_save(
        train_file=args.train_file,
        valid_file=args.valid_file,
        train_sample=args.train_sample,
        valid_sample=args.valid_sample,
        feature_model=args.feature_model,
        classifier=args.classifier,
        model=args.model
    )
