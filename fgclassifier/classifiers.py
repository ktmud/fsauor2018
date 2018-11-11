"""
Different Classifier models
"""
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

DummyStratified = DummyClassifier(strategy='stratified')
DummyMostFrequent = DummyClassifier(strategy='most_frequent')

SVC_rbf = SVC(kernel='rbf')
