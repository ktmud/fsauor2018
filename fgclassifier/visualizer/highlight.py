#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Highlight Sentiment Tags
"""
from functools import lru_cache

import spacy
from textblob import TextBlob
from snownlp import SnowNLP

from fgclassifier.embedding import split_sentences
from fgclassifier.embedding import RE_SUBSENTENCE, split_subsentences


@lru_cache(2)
def spacy_load(lang='en'):
    """Load spacy models"""
    if lang == 'en':
        lang = 'en_core_web_sm'
    elif lang == 'zh':
        lang = 'zh_core_web_sm'
    return spacy.load(lang)


def _sub_highlight(html, chunk, senti, replacements):
    # replace only the first occurance
    replacements.append(f'<span class="{senti}">{chunk}</span>')
    return html.replace(chunk, f'-CHUNK{len(replacements) - 1}-', 1)


def _show_highlight(html, replacements):
    for i, chunk in enumerate(replacements):
        html = html.replace(f'-CHUNK{i}-', chunk, 1)
    return html

        
def highlight_noun_chunks(text, lang='en'):
    """Highlight noun chunks with sentiments, wrap with HTML tags"""
    if lang == 'zh':
        return zh_highlight_noun_chunks(text)

    nlp = spacy_load(lang)
    html = ''
    replacements = []
    for sentence in split_sentences(text):
        score = TextBlob(sentence).sentiment.polarity
        sentiment = 'neutral'
        if score > 0.1:
            sentiment = 'positive'
        elif score < -0.1:
            sentiment = 'negative'
        # Find the longest noun_chunk in the text, assign the whole
        # sentence's sentiment to it.
        chunks = sorted(nlp(sentence).noun_chunks, key=lambda x: -len(x))

        # highlight the longest two noun chunks
        for chunk in chunks[:2]:
            # the chunk must has at least two words
            if len(chunk) > 1:
                sentence = _sub_highlight(sentence, chunk.text,
                                          sentiment, replacements)
        html += '<span class="sentence">' + sentence + '</span>'
    return _show_highlight(html, replacements)


def highlight_subsetence(text, lang='en'):
    """Highlight noun chunks with sentiments, wrap with HTML tags"""
    html = ''
    for sentence in split_sentences(text):
        html += '<span class="sentence">'
        # highlight the longest two noun chunks
        for subsentence, g1, g2 in split_subsentences(sentence):
            sentiment = None  # none is neutral
            if lang == 'zh':
                score = SnowNLP(subsentence).sentiments * 2 - 1
                if score > 0.5:
                    sentiment = 'positive'
                elif score < -0.5:
                    sentiment = 'negative'
            else:
                score = TextBlob(subsentence).sentiment.polarity
                if score > 0.29:
                    sentiment = 'positive'
                elif score < -0.29:
                    sentiment = 'negative'

            if sentiment:
                senti_s = (
                    f'<span class="{sentiment}" ' +
                    f'title="{score:.2f}">'
                )
                senti_e = f'</span>'
            else:
                senti_s = f'<span title="{score:.2f}">'
                senti_e = f'</span>'
            html += f'{senti_s}{g1}{senti_e}{g2}'
        html += '</span>'
    return html


def zh_noun_chunks_iterator(obj):
    """
    Iterate Chinse noun chunks
    """
    labels = ['nmod', 'punct', 'obj', 'nsubj',
              'dobj', 'nsubjpass', 'pcomp', 'pobj', 'dative', 'appos',
              'attr', 'ROOT']

    doc = obj.doc # Ensure works on both Doc and Span.
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add('conj')
    np_label = doc.vocab.strings.add('NP')
    
    seen = set()
    exclude = set(['，', ','])  # always exclude 「，」
    for i, word in enumerate(obj):
        # print(word, '\t', word.left_edge, word.tag_, word.dep_)
        if word.tag_ not in ('NNP', 'NN', 'RB'):
            continue
        # Prevent nested chunks from being produced
        if word.i in seen or word.text in exclude:
            continue
        if word.dep in np_deps:
            # print([w for w in word.subtree])
            if any((w.i in seen or w.text in exclude) for w in word.subtree):
                continue
            seen.update(j for j in range(word.left_edge.i, word.i+1))
            yield word.left_edge.i, word.i+1, np_label
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                if any(w.i in seen for w in word.subtree):
                    continue
                seen.update(j for j in range(word.left_edge.i, word.i+1))
                yield word.left_edge.i, word.i+1, np_label


def zh_highlight_noun_chunks(text):
    """Highlight noun chunks for Chinese"""
    nlp = spacy_load('zh')
    html = ''
    replacements = []
    for sent in split_sentences(text):
        # Sentiment score from SnowNLP is at [0, 1] range
        score = SnowNLP(sent).sentiments
        senti = 'neutral'
        if score > 0.6:
            senti = 'positive'
        elif score < 0.4:
            senti = 'negative'

        doc = nlp(sent)
        doc.noun_chunks_iterator = zh_noun_chunks_iterator
        chunks = sorted(doc.noun_chunks, key=lambda x: -len(x))

        # highlight the longest two noun chunks
        for chunk in chunks[:2]:
            # the chunk must has at least two words
            if len(chunk) > 1:
                # add highlight to replacements
                sent = _sub_highlight(sent, chunk.text, senti, replacements)
        html += '<span class="sentence">' + sent + '</span>'

    return _show_highlight(html, replacements)
