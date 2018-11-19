
"""
A Baseline Model.

TfIdfVectorizer + Classify aspects separately
"""
import logging
import numpy as np

from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier

from fgclassifier.features import DummyTransform
from fgclassifier.utils import read_data, ensure_named_steps

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Baseline(Pipeline):
    """The Baseline model. Automatically ensuares MultiOutputClassifier

    Parameters
    --------------
        classifier:  the classifier to add to the final step
        name:        name of this model, useful when saving the model
    """

    def __init__(self, classifier=None, steps=None, name=None):
        steps = steps or []
        if classifier is not None:
            steps.append(classifier)
        steps = ensure_named_steps(steps)
        # Make sure last step is a MultiOutputClassifier
        if not isinstance(steps[-1][1], MultiOutputClassifier):
            steps[-1] = (steps[-1][0], MultiOutputClassifier(steps[-1][1]))
        super().__init__(steps)
        self._name = name

    @property
    def classifier_name(self):
        return self.steps[-1][0]

    @property
    def name(self):
        if self._name:
            return self._name
        return self.__class__.__name__ + '_' + self.classifier_name

    def scores(self, X, y):
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
        scores = self.scores(X, y)
        return np.mean(scores)

    def predict_df(self, df, save_to=None):
        """Make prediction on a data frame and save output"""
        # read_data returns a copy of df
        X, y, df = read_data(df, return_df=True)
        df['content'] = ''
        df[y.columns] = self.pred(X)
        if save_to:
            df.to_csv(save_to, encoding="utf_8_sig", index=False)
        return df


class Dummy(Baseline):

    def __init__(self, classifier, **kwargs):
        steps = [
            ('dummy_transform', DummyTransform()),
            ('classify', MultiOutputClassifier(classifier))
        ]
        super(Baseline, self).__init__(steps)

