#!/usr/bin/python3
# 
# utils.py
# Utils.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
#

import ud_dataloader
import config
from get_trans import get_transition_sequence, cross_check
import logging
import time
import numpy as np

def getWordPos(data):
    words = {}
    pos_tag = set()
    for sen in data:
        for token in sen.tokens:
            w = token.form
            t = token.pos_tag
            if not w in words:
                words[w] = 1
            else:
                words[w] = words[w] + 1
            if not t in pos_tag:
                pos_tag.add(t)
    pos_tag = list(pos_tag)
    return words, pos_tag

def getDeprelList(data):
    deprels = set()
    for sen in data:
        for token in sen.tokens:
            deprels.add(token.deprel)
    return list(deprels)

def mapTokenToId(sen: ud_dataloader.UDSentence, word_map:dict):
    ret = []
    for item in sen.tokens:
        if item.form in word_map:
            ret.append(word_map[item.form])
        else:
            ret.append(word_map[config.UNKNOW_TOKEN])
    return ret

def mapPosTagToId(sen: ud_dataloader.UDSentence, tag_map:dict):
    ret = []
    for item in sen.tokens:
        if item.pos_tag in tag_map:
            ret.append(tag_map[item.pos_tag])
    return ret

def mapDeprelTagToId(sen: ud_dataloader.UDSentence, tag_map:dict):
    ret = []
    for item in sen.tokens:
        if item.deprel in tag_map:
            ret.append(tag_map[item.deprel])
    return ret

def mapTransTagToId(sen: ud_dataloader.UDSentence):
    ret = []
    trans = get_transition_sequence(sen)
    for item in trans:
        ret.append(config.PARSER_TAGS_MAP[item])
    return ret

def init_logging(phase):
    current_time = time.localtime()
    fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(
            format=fmt,
            filename='{}_parser_{}{:02}{:02}_{:02}{:02}{:02}.log'.format(
                phase,
                current_time.tm_year,
                current_time.tm_mon,
                current_time.tm_mday,
                current_time.tm_hour,
                current_time.tm_min,
                current_time.tm_sec), level=logging.INFO)
    formatter = logging.Formatter(fmt)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return current_time

def reconstrut_tree_with_transition_labels(sen: ud_dataloader, trans):
    assert trans[0] == config.PARSER_TAGS_MAP['SHIFT']
    buf = sen.tokens
    stack = [0]
    buf_idx = 1
    heads = [-1] * len(buf)

    for item in trans[1:]:
        item = int(item)
        if config.PARSER_TAGS[item] == 'SHIFT':
            stack.append(buf_idx)
            buf_idx = buf_idx + 1
        elif config.PARSER_TAGS[item] == 'LEFT-ARC':
            if len(stack) > 0:
                s2 = stack.pop()
            else:
                s2 = len(heads) + 10
            if len(stack) > 0:
                s1 = stack.pop()
            else:
                s1 = len(heads) + 10
            if (s1 < len(heads)):
                heads[s1] = s2
            stack.append(s2)

        elif config.PARSER_TAGS[item] == 'RIGHT-ARC':
            if len(stack) > 0:
                s2 = stack.pop()
            else:
                s2 = len(heads) + 10
            if len(stack) > 0:
                s1 = stack.pop()
            else:
                s1 = len(heads) + 10

            if (s2 < len(heads)):
                heads[s2] = s1
            stack.append(s1)
        else:
            raise RuntimeError('Unrecongized label!')
    return heads

def getUAS(heads_pred, sen, punctuation_tag=None):
    """
    calculating UAS.

    Arguments:
        heads_pred: head label predicted with reconstrut_tree_with_transition_labels
        sen: sentence object
        punctuation_tag: punctuation POS tag. Do not ignore punctuation if it is None.
    """
    heads_gt = [t.head for t in sen.tokens]
    punc = np.array([t.pos_tag != punctuation_tag for t in sen.tokens])
    heads_gt = np.array(heads_gt) * punc
    heads_pred = np.array(heads_pred) * punc
    return (heads_pred == heads_gt).sum() - 1 # remove root

def getLAS(heads_pred, deprel_pred, sen, deprel_map, punctuation_tag=None):
    """
    calculating LAS.

    Arguments:
        heads_pred: head label predicted with reconstrut_tree_with_transition_labels
        deprel_pred: dependent relation label predicted
        sen: sentence object
        punctuation_tag: punctuation POS tag. Do not ignore punctuation if it is None.
    """
    heads_gt = [t.head for t in sen.tokens]
    deprel_gt = [deprel_map[t.deprel] for t in sen.tokens]
    punc = np.array([t.pos_tag != punctuation_tag for t in sen.tokens])
    heads_gt = np.array(heads_gt) * punc
    deprel_gt = np.array(deprel_gt) * punc
    heads_pred = np.array(heads_pred) * punc
    deprel_pred = np.array(deprel_pred) * punc
    label_correct = (heads_pred == heads_gt) * (deprel_pred == deprel_gt)
    return label_correct.sum() - 1 # remove root
