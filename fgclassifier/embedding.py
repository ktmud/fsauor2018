#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For Word Embedding Features
"""
import re
import numpy as np

from sklearn.base import BaseEstimator
from gensim.sklearn_api import W2VTransformer as W2VTransformer_
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import stem_text, strip_punctuation


EN_FILTERS = [stem_text, strip_punctuation]


def tokenize_zh(s):
    """Tokenize by splitting by space and remove punctuations"""
    tmp = (x.strip(' .。\'\"“”‘’~～') for x in s.lower().split())
    return [s for s in tmp if s]


def tokenize_en(s):
    return preprocess_string(s, EN_FILTERS)


class W2VTransformer(W2VTransformer_):

    def transform(self, X):
        """Make sure transform returns the same type of data as fit"""
        wv = self.gensim_model
        ret = []
        for words in X:
            vectors = np.vstack(
                wv[word][None, :] for word in words
                if word in wv
            )
            ret.append(np.mean(vectors, axis=0))
        return np.vstack(ret)


class Text2Tokens(BaseEstimator):
    """Transform tokens to splitted"""

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer or tokenize_zh

    def fit(self, X, y=None, **kwargs):
        """Fit the model"""
        return self

    def transform(self, X):
        """Transform"""
        tokenize = self._tokenizer
        return [tokenize(x) for x in X]


# More robust split sentences
RE_SENTENCE = re.compile(r'.*?[。….？！?!；~～]+[\)）\"]*')
# subsentence include splitting by commas
RE_SUBSENTENCE = re.compile(r'((.+?)([，,；:：。….？！?!；~～]+[\)）\"]*|$))')
RE_BLANK_AND_MARK = re.compile(r'\s+([。….？！?!；~～])')


def split_subsentences(text):
    """Split Chinese sentences"""
    # replace consequetive "<space><mark><space><mark>"
    # with just marks
    text = RE_BLANK_AND_MARK.sub(r'\1', text)
    seen_one = False
    for sent, g1, g2 in RE_SUBSENTENCE.findall(text):
        seen_one = True
        yield sent.strip(), g1, g2
    if not seen_one:
        yield text.strip(), text.strip(), ''


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
