#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For Word Embedding Features
"""
import re

from sklearn.base import BaseEstimator
from gensim.sklearn_api import W2VTransformer


class Text2Tokens(BaseEstimator):
    """Transform tokens to splitted"""

    def fit(self):
        """Fit the model"""
        return self

    def transform(self, X):
        """Transform"""
        return [x.split() for x in X]


# More robust split sentences
RE_SENTENCE = re.compile(r'.*?[。….？！?!；~～]+') 
RE_BLANK_AND_MARK = re.compile(r'\s+([。….？！?!；~～])')


def split_sentences(text):
    """Split Chinese sentences"""
    # replace consequetive "<space><mark>"
    text = RE_BLANK_AND_MARK.sub(r'\1', text)
    seen_one = False
    for sent in RE_SENTENCE.findall(text):
        seen_one = True
        yield sent.strip()
    if not seen_one:
        yield text.strip()


def content_to_corpus(df, txt_path, print_sample=10):
    """Convert Review content to text corpus for word embedding training"""
    sentences = df['content']
    if print_sample:
        print('\n'.join(sentences[:print_sample]))
    all_content = '\n'.join(sentences)
    with open(txt_path, 'w') as f:
        f.write(all_content + '\n')
