#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Visualizer for Fine-grained Sentiment Classification
"""
import os

from flask import Flask, send_from_directory
from flask import request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO

from fgclassifier.visualizer import actions
from fgclassifier.visualizer.config import dataset_choices, fm_choices
from fgclassifier.visualizer.config import clf_choices


app = Flask(__name__)
app.config['SECRECT_KEY'] = os.environ.get('FLASK_SECRECT_KEY', 'keyboardcat!')
CORS(app)
socketio = SocketIO(app)


@app.route('/')
def index():
    """The Index Page"""
    
    inputs = actions.parse_inputs(**dict(request.args.items()))
    tmpl_data = {
        'dataset_choices': dataset_choices,
        'fm_choices': fm_choices,
        'clf_choices': clf_choices,
        **inputs,
        **actions.predict_one(**inputs)
    }
    return render_template('index.html', **tmpl_data)


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.route('/predict')
def predict():
    """Predict for one single review"""
    inputs = actions.parse_inputs(**dict(request.args.items()))
    return jsonify(actions.predict_one(**inputs))


if __name__ == '__main__':
    port = os.environ.get('DOKKU_PROXY_PORT', 5000)
    socketio.run(app, port=port)
