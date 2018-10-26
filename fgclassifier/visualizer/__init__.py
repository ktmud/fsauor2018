import pandas as pd
import numpy as np

import os.path
from flask import Flask, send_from_directory, send_file
from flask import request, jsonify
from flask_cors import CORS

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, auc

app = Flask(__name__)
CORS(app)

def prepare():
    return X_train, y_train, X_test, y_test

# Assign globals
X_train, y_train, X_test, y_test = prepare()

def parse_args(preprocessing='none', model='logreg', c=1.00, kernel='linear', **kwargs):

    Y, Yt = y_train, y_test
    model_ = model  # save model name

    # must add certain preprocessing to SVM models
    if model == 'svm' and preprocessing == 'none':
        preprocessing = 'standaridization'

    # Preprocessing method
    if preprocessing == 'discretization':
        transformer = KBinsDiscretizer(n_bins=10)
        # some models (Naive Bayes) does not support sparse matrix
        X = transformer.fit_transform(X_train).toarray()
        Xt = transformer.transform(X_test).toarray()
    elif preprocessing == 'standardization':
        transformer = StandardScaler()
        X = transformer.fit_transform(X_train)
        Xt = transformer.transform(X_test)
    else:
        X, Xt = X_train, X_test

    # Hyperparameters
    params = {}
    # classfiers
    if classifier == 'svm':
        params['C'] = float(c)
        params['kernel'] = kernel
        classifier = SVC(**params, probability=True)
    elif classifier == 'nb':
        classifier = GaussianNB(**params)
    else:
        params['C'] = float(c)
        classifier = LogisticRegression(**params)

    # Expose all parameters
    params = {
        'preprocessing': preprocessing,
        'model': model_,
        **params
    }
    return params, model, X, Y, Xt, Yt

def run_model(**kwargs):
    params, model, X, Y, Xt, Yt = parse_args(**kwargs)
    model.train(X, Y)
    model.validate(X, Y)
    ret = {
        'params': params,
        'scores': model.scores
    }
    return ret

@app.route('/roc_curve')
def api_roc_curve():  # pass arguments
    ret = run_model(**{ key: val for key, val in request.args.items()})
    return jsonify(ret)


@app.route('/')
def index():
    return send_file('static/index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)
