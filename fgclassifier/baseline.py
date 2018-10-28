
"""
A Baseline Model.

TfIdfVectorizer + Classify aspects separately
"""
import os
import logging
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.base import clone as sk_clone

from fgclassifier import read_csv, f1_score
from fgclassifier import classifiers

logger = logging.getLogger(__name__)


class Baseline():
    """
    The Baseline model
    """

    # transform review content (string) to features (matrices)
    DEFAULT_TRANSFORMER = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2')
    # the model to run
    DEFAULT_CLASSIFIER = classifiers.MultinomialNB

    def __init__(self, transformer=None, classifier=None):
        # separate classifiers for each aspect
        self.classifiers = {}  # trained models
        self.scores = {}  # F1 scores to measure model performance

        # All aspects use the same feature transformer, but need
        # different instances of classifiers
        self.transformer = transformer or self.DEFAULT_TRANSFORMER
        self.classifier = classifier or self.DEFAULT_CLASSIFIER

        if callable(self.transformer):
            self.transformer = self.transformer()
        if callable(self.classifier):
            self.classifier = self.classifier()

    def load(self, data_path, fit=False, **kwargs):
        """Extract features and associated labels"""
        df = read_csv(data_path, **kwargs)
        if fit or not hasattr(self.transformer, 'vocabulary_'):
            logger.info('Fitting feature transformer...')
            self.transformer.fit(df['content'])
            logger.info('Fitted training features, vocabulary: %s',
                        len(self.transformer.vocabulary_.keys()))
        features = self.transformer.transform(df['content'])
        labels = df.drop(['id', 'content'], axis=1)
        return features, labels

    def save(self, filepath):
        """Save model to disk, so we can reuse it later"""
        logger.info("Saving model...")
        pathdir = os.path.dirname(filepath)
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)
        joblib.dump(self, filepath)
        logger.info("Saving model... Done.")

class Indie(Baseline):
    """
    Fine Grain classifier where we train and predict for
    each aspect independently.
    """

    def train(self, features, labels):
        """Train the model"""
        for aspect in labels.columns:
            logger.debug("[train] %s ", aspect)
            # Each aspect must use different estimators,
            # so we make a clone here
            model = sk_clone(self.classifier)
            model.fit(features, labels[aspect])
            self.classifiers[aspect] = model

    def validate(self, features, labels):
        """Validate trained models"""
        for aspect in labels.columns:
            logger.debug("[validate] %s ", aspect)
            model = self.classifiers[aspect]
            X, y = features, labels[aspect]
            self.scores[aspect] = f1_score(model, X, y)
        avg_score = np.mean(list(self.scores.values()))
        logger.info('[validate] \n' + '\n'.join('  {: <40s}\t{:.4f}'.format(aspect, self.scores[aspect])
                                                for aspect in labels.columns))
        logger.info("[validate] Final F1 Score: %s\n", avg_score)
        return avg_score, self.scores

    def predict(self, df, save_to=None):
        """Predict classes and update the output dataframe"""
        logger.info("compete predict test data.")
        features = self.transformer.transform(df['content'])
        for aspect, model in self.classifiers.items():
            df[aspect] = model.predict(features)
        if save_to:
            df.to_csv(save_to, encoding="utf_8_sig", index=False)
