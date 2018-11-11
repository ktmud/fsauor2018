"""
Load data and train the model
"""
import os
import argparse

from fgclassifier import models, classifiers

try:
    import local_config as config
except ImportError:
    import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    classifier_choices = [x for x in dir(classifiers) if not x.startswith('_')]
    parser.add_argument('-m', '--model', default='Indie',
                        help='Top-level model, include how to do feature engineering')
    parser.add_argument('-c', '--classifier', default='ComplementNB',
                        choices=classifier_choices,
                        help='Classifier used by the model')
    args = parser.parse_args()

    Model = getattr(models, args.model)
    Classifier = getattr(classifiers, args.classifier)
    model = Model(classifier=Classifier)
    X_train, Y_train = model.load(config.train_data_path, sample_n=10000)
    X_validate, Y_validate = model.load(config.validate_data_path, sample_n=1000)

    model.train(X_train, Y_train)
    model.validate(X_validate, Y_validate)

    model.save(os.path.join(config.model_save_path, model.name + '.pkl'))
