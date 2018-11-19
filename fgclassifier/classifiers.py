"""
Different Classifier models
"""
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier

DummyStratified = DummyClassifier(strategy='stratified')
DummyMostFrequent = DummyClassifier(strategy='most_frequent')
ExtraTree = ExtraTreesClassifier(n_estimators=50)
