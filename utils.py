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
            token.form = token.form.lower()
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
        ret.append(word_map[item.form])
    return ret

def mapTransTagToId(sen: ud_dataloader.UDSentence):
    ret = []
    trans = get_transition_sequence(sen)
    for item in trans:
        ret.append(config.PARSER_TAGS_MAP[item])
    return ret

def init_logging():
    current_time = time.localtime()
    fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(
            format=fmt,
            filename='train_parser_{}{:02}{:02}_{:02}{:02}{:02}.log'.format(
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
