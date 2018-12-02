"""
Different Classifier models
"""
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier

DummyStratified = DummyClassifier(strategy='stratified')
DummyMostFrequent = DummyClassifier(strategy='most_frequent')
ExtraTree = ExtraTreesClassifier(n_estimators=50, max_depth=10)
RandomForest = RandomForestClassifier(n_estimators=50, max_depth=10)

# These are default parameters,
# we initialize an instance here just to suppress warnings
Logistic = LogisticRegression(
    solver='lbfgs',
    multi_class='auto',
    n_jobs=-1,
    max_iter=200,
)
Ridge = RidgeClassifierCV(alphas=(0.01, 0.1, 0.5, 1.0, 5.0, 10.0))
LDA = LinearDiscriminantAnalysis()
QDA = QuadraticDiscriminantAnalysis()
RBF = SVC(kernel='rbf')


# Stochastic Gradient Descent with SVM
SGD_LinearSVC = SGDClassifier(
    max_iter=5000, tol=1e-6, alpha=1e-6)
