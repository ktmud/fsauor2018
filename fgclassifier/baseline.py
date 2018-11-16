
"""
A Baseline Model.

TfIdfVectorizer + Classify aspects separately
"""
import logging
import numpy as np

from sklearn.base import clone as sk_clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier

from fgclassifier.utils import read_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Tfidf(TfidfVectorizer):

    def fit(self, *args, **kwargs):
        logging.info('Fitting TF-IDF...')
        return super().fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        logging.info('Transforming TF-IDF...')
        return super().transform(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        logging.info('Fit & Transform TF-IDF...')
        return super().fit_transform(*args, **kwargs)


class SVD(TruncatedSVD):

    def fit(self, *args, **kwargs):
        logging.info('Fitting TruncatedSVD...')
        return super().fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        logging.info('Transforming TruncatedSVD...')
        return super().transform(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        logging.info('Fit & Transform TruncatedSVD...')
        return super().fit_transform(*args, **kwargs)


class Baseline(Pipeline):
    """
    The Baseline model

    Defaults to Tf-idf + TruncatedSVD, a.k.a, Latent Semantic Analysis
    Classifier uses Complement Naive Bayes.
    """
    # Default feature pipeline
    FEATURE_PIPELINE = [
        # from one of our sample, words like 味道 (taste), 好 (good)
        # are also in the most frequent words, but they are actually relevant
        # in terms of sentiment.
        ('tfidf', Tfidf(analyzer='word', ngram_range=(1, 5),
                        min_df=0.01, max_df=0.99, norm='l2')),
        ('reduce_dim', SVD(n_components=500))
    ]

    def __init__(self, classifier=None, steps=None):
        if steps is None:
            steps = [(name, sk_clone(estimator))
                     for name, estimator in self.FEATURE_PIPELINE]
        if classifier is not None:
            if callable(classifier):
                classifier = classifier()
            classifier_name = classifier.__class__.__name__
            classifier = MultiOutputClassifier(classifier)
            steps.append((classifier_name, classifier))
        super().__init__(steps)

    @property
    def classifier_name(self):
        return self.steps[-1][0]

    @property
    def name(self):
        return self.__class__.__name__ + '_' + self.classifier_name

    def f1_scores(self, X, y):
        """Return f1 score on a test dataset"""
        y_pred = self.predict(X)
        scores = []
        logger.info('[Validate]: F1 Scores')
        for i, label in enumerate(y.columns):
            score = f1_score(y[label], y_pred[:, i], average='macro')
            scores.append(score)
            logger.info('  {: <40s}\t{:.4f}'.format(label, score))
        return scores

    def score(self, X, y):
        scores = self.f1_scores(X, y)
        return np.mean(scores)

    def pred_df(self, df, save_to=None):
        """Make prediction on a data frame and save output"""
        # read_data returns a copy of df
        X, y, df = read_data(df, return_df=True)
        df['content'] = ''
        df[y.columns] = self.pred(X)
        if save_to:
            df.to_csv(save_to, encoding="utf_8_sig", index=False)
        return df


class SparseToSense():
    """Return content length as features.
    do nothing to the labels"""

    def __init__(self):
        pass

    def fit(self, X, y):
        return self
        
    def transform(self, X):
        return X.toarray()


class DummyTransform():
    """Return content length as features.
    do nothing to the labels"""

    def __init__(self):
        pass

    def fit(self, X, y):
        return self
        
    def transform(self, X):
        return np.array(X['content'].str.len())[:, None]


class Dummy(Baseline):

    def __init__(self, classifier, **kwargs):
        steps = [
            ('dummy_transform', DummyTransform()),
            ('classify', MultiOutputClassifier(classifier))
        ]
        super(Baseline, self).__init__(steps)
