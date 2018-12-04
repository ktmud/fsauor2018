"""
Different Classifier models
"""
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier, RidgeClassifierCV
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
Logistic = LogisticRegressionCV(
    solver='lbfgs',
    multi_class='auto',
    n_jobs=-1,
    max_iter=200,
    class_weight='balanced'
)
Ridge = RidgeClassifierCV(
    alphas=(0.01, 0.1, 0.5, 1.0, 5.0, 10.0)
)
LDA = LinearDiscriminantAnalysis()
QDA = QuadraticDiscriminantAnalysis(reg_param=0.0001)
RBF = SVC(
    kernel='rbf', gamma='scale',
    C=0.1, class_weight='balanced', cache_size=1000,
    probability=True
)
SVM_Poly2 = SVC(
    kernel='poly', degree=2, gamma='scale',
    C=0.1, class_weight='balanced', cache_size=1000,
    probability=True
)
SVM_Poly3 = SVC(
    kernel='poly', degree=3, gamma='scale',
    C=0.1, class_weight='balanced', cache_size=1000,
    probability=True
)


# Stochastic Gradient Descent with SVM
SGD_LinearSVC = SGDClassifier(
    max_iter=5000, tol=1e-6, alpha=1e-6)
