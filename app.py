#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Start the Visualization App
"""
import logging
import argparse

from fgclassifier.visualizer import app, socketio

parser = argparse.ArgumentParser(description='Development Server')
parser.add_argument('--port', '-p', type=int, default=5000)
args = parser.parse_args()

if __name__ == '__main__':
    logging.info(f'Starting dev server at http://0.0.0.0:{args.port}')
    socketio.run(app, port=args.port, debug=True)
