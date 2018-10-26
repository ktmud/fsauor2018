"""
Load data and train the model
"""
import os
import argparse

from fgclassifier.baseline import Indie
from fgclassifier import classifiers

try:
    import local_config as config
except ImportError:
    import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    classifier_choices = [x for x in dir(classifiers) if not x.startswith('_')]
    parser.add_argument('-c', '--classifier', default='MultinomialNB',
                        choices=classifier_choices,
                        help='Classifier used for each aspect')
    parser.add_argument('-m', '--model', default='no_interference',
                        help='Classifier used for each aspect')
    args = parser.parse_args()
    Classifier = getattr(classifiers, args.classifier)
    model = Indie(classifier=Classifier())
    X_train, Y_train = model.load(config.train_data_path, sample_n=None)
    X_validate, Y_validate = model.load(config.validate_data_path, sample_n=100)
    model.train(X_train, Y_train)
    model.validate(X_validate, Y_validate)
    model.save(os.path.join(config.model_save_path, args.classifier + '.pkl'))
