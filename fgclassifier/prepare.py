#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare visualizer environment, install 3rd-party libraries, etc.
"""
import nltk
from spacy.cli.download import download as spacy_download

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

spacy_download('en_core_web_sm')
