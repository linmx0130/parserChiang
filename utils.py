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
import argparse

def getWordPos(data):
    """
    Get word list and POS tags list.

    Arguments:
        data: a list of sentence objects.
    Return:
        words: a dict of str->int. 
               The value is the appearance count of the word.
        pos_tag: POS tag list.
    """
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

def mapTokenToId(sen: ud_dataloader.UDSentence, word_map:dict, word_dropout_rate=0, words_count=None):
    """
    map tokens in a sentence into word Id with word_map
    
    Arguments:
        sen: sentence object.
        word_map: word string to ID mapping dict.
        word_dropout_rate: word dropout rate, set to 0 if dropout is not applied.
        words_count: word counts dict for word dropout.
    """
    ret = []
    if word_dropout_rate > 0:
        assert words_count is not None
        drop_rand = np.random.uniform(0, 1, len(sen.tokens))

    for i, item in enumerate(sen.tokens):
        not_unk_token = False
        if item.form in word_map:
            if word_dropout_rate > 0:
                # A higher words_count will lead to lower keep_thresh
                # So it will keep word with a low drop_rand[i] 
                keep_thresh = word_dropout_rate / (word_dropout_rate + words_count[item.form]) 
                if drop_rand[i] >= keep_thresh:
                    not_unk_token = True
            else:
                not_unk_token = True

        if not_unk_token:
            ret.append(word_map[item.form])
        else:
            ret.append(word_map[config.UNKNOW_TOKEN])
    return ret

def mapPosTagToId(sen: ud_dataloader.UDSentence, tag_map:dict):
    ret = []
    for item in sen.tokens:
        if item.pos_tag in tag_map:
            ret.append(tag_map[item.pos_tag])
        else:
            raise RuntimeError("POS tag {} not found!".format(item.pos_tag))
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

def getUAS(heads_pred, sen, punctuation_tag=[]):
    """
    calculating UAS.

    Arguments:
        heads_pred: head label predicted with reconstrut_tree_with_transition_labels
        sen: sentence object
        punctuation_tag: punctuation POS tag. Do not ignore punctuation if it is None.
    """
    if type(punctuation_tag) is str:
        punctuation_tag = [punctuation_tag, ]

    heads_gt = [t.head for t in sen.tokens]
    punc = np.array([not(t.pos_tag in punctuation_tag) for t in sen.tokens])
    heads_gt = np.array(heads_gt) * punc
    heads_pred = np.array(heads_pred) * punc
    return (heads_pred == heads_gt).sum() - 1 # remove root

def getLAS(heads_pred, deprel_pred, sen, deprel_map, punctuation_tag=[]):
    """
    calculating LAS.

    Arguments:
        heads_pred: head label predicted with reconstrut_tree_with_transition_labels
        deprel_pred: dependent relation label predicted
        sen: sentence object
        punctuation_tag: punctuation POS tag. Do not ignore punctuation if it is None.
    """
    if type(punctuation_tag) is str:
        punctuation_tag = [punctuation_tag, ]

    heads_gt = [t.head for t in sen.tokens]
    deprel_gt = [deprel_map[t.deprel] for t in sen.tokens]
    punc = np.array([not(t.pos_tag in punctuation_tag) for t in sen.tokens])
    heads_gt = np.array(heads_gt) * punc
    deprel_gt = np.array(deprel_gt) * punc
    heads_pred = np.array(heads_pred) * punc
    deprel_pred = np.array(deprel_pred) * punc
    label_correct = (heads_pred == heads_gt) * (deprel_pred == deprel_gt)
    return label_correct.sum() - 1 # remove root

def loadWordvec(wordvec_filename):
    """
    load word vector text file.
    
    Arguments:
        wordvec_filename: filename of the file to load
    """
    word2vec = {}
    with open(wordvec_filename) as f:
        for l in f:
            l = l.split()
            w = l[0]
            vec = np.array([float(t) for t in l[1:]], dtype=np.float32)
            word2vec[w] = vec
    return word2vec

def setEmbeddingWithWordvec(embed_layer, word_map, wordvec_filename):
    word2vec = loadWordvec(wordvec_filename)
    weight = embed_layer.weight.data()
    for w in word2vec:
        if w in word_map:
            weight[word_map[w]] = word2vec[w]
    embed_layer.weight.set_data(weight)

def trainerArgumentParser():
    """
    trainer default argument parser generator
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordvec', dest='wordvec', default=None, 
            help='Load word vector file as initial value of embedding layers.')
    parser.add_argument('--cpu', dest='use_cpu', default=False, 
            action='store_true', help='Train on CPUs.')
    parser.add_argument('--trainer', dest='trainer', default='adam', 
            help='Choose optimization algorithm. {adam, adagrad, sgd} are supported.')
    return parser

def testerArgumentParser():
    """
    tester default argument parser generator 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', 
                        help="The directory stored model files and word map files.")
    parser.add_argument('model_file', help="Model file name.")
    parser.add_argument('--cpu', help='Use CPU to run the model.', 
                        dest='use_cpu', default=False, action='store_true')
    return parser

def getDefaultTrainerHyperparams(trainer_name):
    """
    Get default trainer by trainer name
    """
    ret = {'adam': {'learning_rate':0.001, 'wd': 1e-4},
           'adagrad': {'learning_rate':0.04, 'wd': 1e-4},
           'sgd': {'learning_rate': 0.0001, 'wd': 1e-4, 'momentum': 0.9}
           }
    return ret[trainer_name]
