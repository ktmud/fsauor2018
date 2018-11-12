
"""
A Baseline Model.

TfIdfVectorizer + Classify aspects separately
"""
import os
import logging
import numpy as np

import _pickle as cPickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.base import clone as sk_clone
from sklearn.decomposition import TruncatedSVD

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

    Defaults to Tf-idf + TruncatedSVD, a.k.a, Latent Semantic Analysis
    Classifier uses Complement Naive Bayes.
    """

    # transform review content (string) to features (matrices)
    DEFAULT_VECTORIZER = lambda x: TfidfVectorizer(
        analyzer='word', ngram_range=(1, 5), min_df=0.01, max_df=0.95,
        norm='l2')
    # the model to run
    DEFAULT_CLASSIFIER = classifiers.ComplementNB
    DEFAULT_REDUCER = lambda x: TruncatedSVD(n_components=1000)

    def __init__(self, vectorizer=None, classifier=None, reducer=None):
        # separate classifiers for each aspect
        self.classifiers = {}  # trained models
        self.scores = {}  # F1 scores to measure model performance

        # All aspects use the same feature vectorizer, but need
        # different instances of classifiers
        self.vectorizer = self.DEFAULT_VECTORIZER if vectorizer is None else vectorizer
        self.classifier = self.DEFAULT_CLASSIFIER if classifier is None else classifier
        self.reducer = self.DEFAULT_REDUCER if reducer is None else reducer
        self.standardizer = StandardScaler()

        if callable(self.vectorizer):
            self.vectorizer = self.vectorizer()
        if callable(self.classifier):
            self.classifier = self.classifier()
        if callable(self.reducer):
            self.reducer = self.reducer()
    
    @property
    def classifier_class(self):
        return self.classifier.__class__.__name__ 

    @property
    def name(self):
        return self.__class__.__name__ + '_' + self.classifier_class

    def transform(self, content):
        """Transform text content to features"""
        logger.info('Transform features...')

        if not hasattr(self.vectorizer, 'vocabulary_'):
            logger.info('Fitting feature vectorizer...')
            self.vectorizer.fit(content)
            # print(self.vectorizer.vocabulary_)
            logger.info('Fitted training features, vocabulary: %s',
                        len(self.vectorizer.vocabulary_.keys()))

        features = self.vectorizer.transform(content)

        if self.classifier_class in NO_SPARSE_MATRIX_CLASSIFIERS:
            features = features.toarray()
            # print(np.where(np.isnan(features)))
            # print(np.where(np.isinf(features)))

        if self.classifier_class in NEED_STANDARDIZATION_CLASSIFIERS:
            if not hasattr(self.standardizer, 'scale_'):
                self.standardizer.fit(features)
            features = self.standardizer.transform(features)
        
        if self.reducer:
            if not hasattr(self.reducer, 'components_'):
                if self.reducer.n_components >= features.shape[1]:
                    self.reducer.n_components = features.shape[1] - 1
                self.reducer.fit(features)
            features = self.reducer.transform(features)

        return features

    def read(self, data_path, **kwargs):
        return read_csv(data_path, **kwargs)

    def load(self, data_path, **kwargs):
        """Extract features and associated labels"""
        return self.split_xy(self.read(data_path, **kwargs))

    def split_xy(self, df):
        return self.transform(df['content']), df.drop(['id', 'content'], axis=1)

    def save(self, filepath):
        """Save model to disk, so we can reuse it later"""
        logger.info("Saving model to %s..." % filepath)
        pathdir = os.path.dirname(filepath)
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)
        cPickle.dump(self, filepath)
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

    def predict(self, df, save_to=None):
        """Predict classes and update the output dataframe"""
        features = self.transform(df['content'])
        for aspect, model in self.classifiers.items():
            df[aspect] = model.predict(features)
        logger.info("[test] Complete predicting.")
        df['content'] = ''
        if save_to:
            df.to_csv(save_to, encoding="utf_8_sig", index=False)
        return df

class Dummy(Indie):
    
    def transform(self, content):
        return content

    def load(self, data_path, **kwargs):
        df = read_csv(data_path, **kwargs)
        return np.zeros(df.shape), df.drop(['id', 'content'], axis=1)
    