#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For Word Embedding Features
"""
import re

# -------- For Word Embeddings ---------
RE_EXCL = re.compile('！+')
RE_QUES = re.compile('？+')


def split_by(s, regexp, char):
    if char in s.strip(char):
        tmp = regexp.split(s)
        last = tmp.pop()
        ret = [x + char for x in tmp]
        ret.append(last)  # add last sentence back
        return ret


def article_to_sentences(articles, split_sentence=True):
    sentences, aids, slens = [], [], []
    for aid, article in enumerate(articles):
        if not split_sentence:
            tokens = article.split()
            sentences.append(tokens)
            aids.append(aid)
            slens.append(len(tokens))
            continue
        ss = article.split('。')
        while ss:
            s = ss.pop(0).strip()
            if not s:
                continue
            tmp = split_by(s, RE_EXCL, '！')
            if tmp:
                ss = tmp + ss
                continue
                
            tmp = split_by(s, RE_QUES, '？')
            if tmp:
                ss = tmp + ss
                continue
                
            tokens = s.split()
            sentences.append(tokens)
            # keep a record of article ids and sentence length
            # so that we know which sentence/word belongs to
            # which article
            aids.append(aid)
            slens.append(len(tokens))
    return sentences, aids, slens


def content_to_corpus(df, txt_path, print_sample=10):
    """Convert Review content to text corpus for word embedding training"""
    sentences = df['content']
    if print_sample:
        print('\n'.join(sentences[:print_sample]))
    all_content = '\n'.join(sentences)
    with open(txt_path, 'w') as f:
        f.write(all_content + '\n')
