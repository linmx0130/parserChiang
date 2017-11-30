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
