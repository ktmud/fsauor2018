#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare visualizer environment, install 3rd-party libraries, etc.
"""
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download("maxent_treebank_pos_tagger")
nltk.download("maxent_ne_chunker")
nltk.download(["tagsets", "universal_tagset"])
