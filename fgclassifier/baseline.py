
"""
A Baseline Model.

TfIdfVectorizer + Classify aspects separately
"""
import os
import logging
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.base import clone as sk_clone

from fgclassifier import read_csv, f1_score
from fgclassifier import classifiers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Classifiers that does not accept sparse matrix
# as features
NO_SPARSE_MATRIX_CLASSIFIERS = {
    'LinearDiscriminantAnalysis',
    'QuadraticDiscriminantAnalysis',
    'SVC'
}
NEED_STANDARDIZATION_CLASSIFIERS = {
    'LinearDiscriminantAnalysis',
    'QuadraticDiscriminantAnalysis',
    'SVC'
}

class Baseline():
    """
    The Baseline model
    """

    # transform review content (string) to features (matrices)
    DEFAULT_VECTORIZER = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 5), min_df=0.012, max_df=0.9,
        norm='l2')
    # the model to run
    DEFAULT_CLASSIFIER = classifiers.MultinomialNB

    def __init__(self, vectorizer=None, classifier=None):
        # separate classifiers for each aspect
        self.classifiers = {}  # trained models
        self.scores = {}  # F1 scores to measure model performance

        # All aspects use the same feature vectorizer, but need
        # different instances of classifiers
        self.vectorizer = vectorizer or self.DEFAULT_VECTORIZER
        self.classifier = classifier or self.DEFAULT_CLASSIFIER
        self.standardizer = StandardScaler()

        if callable(self.vectorizer):
            self.vectorizer = self.vectorizer()
        if callable(self.classifier):
            self.classifier = self.classifier()
    
    @property
    def classifier_class(self):
        return self.classifier.__class__.__name__ 

    def transform(self, content):
        """Transform text content to features"""
        features = self.vectorizer.transform(content)
        if self.classifier_class in NO_SPARSE_MATRIX_CLASSIFIERS:
            features = features.toarray()
            # print(np.where(np.isnan(features)))
            # print(np.where(np.isinf(features)))
        if self.classifier_class in NEED_STANDARDIZATION_CLASSIFIERS:
            if not hasattr(self.standardizer, 'scale_'):
                self.standardizer.fit(features)
            features = self.standardizer.transform(features)
        return features

    def load(self, data_path, fit=False, **kwargs):
        """Extract features and associated labels"""
        df = read_csv(data_path, **kwargs)
        if fit or not hasattr(self.vectorizer, 'vocabulary_'):
            logger.info('Fitting feature vectorizer...')
            self.vectorizer.fit(df['content'])
            # print(self.vectorizer.vocabulary_)
            logger.info('Fitted training features, vocabulary: %s',
                        len(self.vectorizer.vocabulary_.keys()))
        logger.info('Transform features...')
        features = self.transform(df['content'])
        labels = df.drop(['id', 'content'], axis=1)
        # Mark NA values "not mentioned"
        labels = labels.fillna(-2)
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
            logger.info("[train] %s ", aspect)
            # Each aspect must use different estimators,
            # so we make a clone here
            model = sk_clone(self.classifier)
            model.fit(features, labels[aspect])
            self.classifiers[aspect] = model

    def validate(self, features, labels):
        """Validate trained models"""
        for aspect in labels.columns:
            logger.info("[validate] %s ", aspect)
            model = self.classifiers[aspect]
            X, y = features, labels[aspect]
            self.scores[aspect] = f1_score(model, X, y)
        avg_score = np.mean(list(self.scores.values()))
        logger.info('[validate] \n' + '\n'.join('  {: <40s}\t{:.4f}'.format(aspect, self.scores[aspect])
                                                for aspect in labels.columns))
        logger.info("[validate] Final F1 Score: %s\n", avg_score)
        return avg_score, self.scores

    def predict(self, features, labels=None, save_to=None):
        """Predict classes and update the output dataframe"""
        if labels is None:
            labels = pd.DataFame()
        for aspect, model in self.classifiers.items():
            labels[aspect] = model.predict(features)
        logger.info("Complete predict for test data.")
        if save_to:
            labels.to_csv(save_to, encoding="utf_8_sig", index=False)
        return labels
