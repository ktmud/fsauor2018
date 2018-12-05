#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Visualizer for Fine-grained Sentiment Classification
"""
import os

from flask import Flask, send_from_directory
from flask import request, jsonify, render_template
from flask_socketio import SocketIO
from flask_assets import Environment
from flask_autoindex import AutoIndex

from fgclassifier.visualizer import actions
from fgclassifier.visualizer.options import dataset_choices, fm_choices
from fgclassifier.visualizer.options import clf_choices

from fgclassifier.utils import get_stats


app = Flask(__name__)
assets = Environment(app)
app.config['SECRECT_KEY'] = os.getenv('FLASK_SECRECT_KEY', 'keyboardcat!')
app.config['ASSETS_DEBUG'] = os.getenv('FLASK_ENV') == 'development'
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
    return render_template('index.jinja', **tmpl_data)


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.route('/predict')
def predict():
    """Predict for one single review"""
    inputs = actions.parse_inputs(**dict(request.args.items()))
    return jsonify(actions.predict_one(**inputs))


@app.route('/predict_text')
def predict_text():
    """Predict for user-inputted text"""
    inputs = dict(request.args.items())
    return jsonify(actions.predict_text(**inputs))


@app.route('/model_stats')
def model_stats():
    """Predict for user-inputted text"""
    args = request.args
    dataset = args.get('dataset')
    fm = args.get('fm')
    clf = args.get('clf')
    if not dataset or not fm or not clf:
        return jsonify({
            'error': 'wrong parameters'
        })
    return jsonify(get_stats(dataset, fm, clf))


# Add autoindex for data directory
files_index = AutoIndex(app, browse_root=os.getenv('DATA_ROOT', 'data'),
                        add_url_rules=False)


@app.route('/files')
@app.route('/files/')
@app.route('/files/<path:path>')
def autoindex(path='.'):
    return files_index.render_autoindex(path)

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    socketio.run(app, host='0.0.0.0', port=port)
