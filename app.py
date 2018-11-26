#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Start the Visualization App
"""
import os
import logging

from fgclassifier.visualizer.app import app, socketio

port = os.environ.get('DOKKU_PROXY_PORT', 5000)
debug = os.environ.get('FLASK_DEBUG') == '1'

if __name__ == '__main__':
    logging.info(f'Starting app at http://localhost:{port}')
    socketio.run(app, port=port, debug=debug)
