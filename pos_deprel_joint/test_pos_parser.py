#!/usr/bin/python3
# 
# test_parser.py
# A basic LSTM transition-based parser. Testing code.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
#

import config
import ud_dataloader
import mxnet as mx
from mxnet import nd, autograd, gluon
from config import dev_data_fn 
from get_trans import cross_check
import pickle
from trans_parser_pos_model import ParserModel
from utils import * 
import os
import argparse
from tqdm import tqdm

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('model_file')
    return parser.parse_args()

current_time = init_logging("test")
args = parseArgs()
model_dump_path = args.model_path

data = ud_dataloader.parseDocument(dev_data_fn)
data = [t for t in data if cross_check(t.tokens) and len(t) > 4]
# data lowerize
for sen in data:
    for token in sen.tokens:
        token.form = token.form.lower()

# load word map
with open(os.path.join(model_dump_path, 'word_map.pkl'), 'rb') as f:
    word_map = pickle.load(f)
    pos_map = pickle.load(f)
    deprel_map = pickle.load(f)

logging.info("Test data loaded: {}".format(dev_data_fn))
logging.info("Sentences count = {}".format(len(data)))
logging.info("Words count = {}".format(len(word_map)))
logging.info("POS Tag count = {}".format(len(pos_map)))
logging.info("Dependent Relation count = {}".format(len(deprel_map)))

ctx = mx.gpu(0)
parserModel = ParserModel(len(word_map), config.NUM_EMBED, config.NUM_HIDDEN, len(pos_map), config.TAG_EMBED, len(deprel_map))
model_file = os.path.join(model_dump_path, args.model_file)
parserModel.load_params(model_file, ctx=ctx)
logging.info("Model loaded: {}".format(model_file))

zero_const = mx.nd.zeros(shape=config.NUM_HIDDEN * 2, ctx=ctx)

# eval
print("Evaluating...")
acc = 0
total_tags = 0
model_acc = 0
model_total_tags = 0
uas = 0
las = 0
total_tokens = 0
pos_acc = 0

for seni in tqdm(range(len(data))):
    sen = data[seni]
    tokens_cpu = mapTokenToId(sen, word_map)
    tokens = mx.nd.array(tokens_cpu, ctx)
    pos_tag = mapPosTagToId(sen, pos_map)
    deprel_tag_cpu = mapDeprelTagToId(sen, deprel_map)
    deprel_tag = mx.nd.array(deprel_tag_cpu, ctx)
    tags = mapTransTagToId(sen)
    
    model_output = []
    model_gt = []
    model_pred = []
    buf_idx = 0
    stack = []
    pred = []
    current_idx = 0

    # parse by transition
    with autograd.predict_mode():
        f, pos_f = parserModel(tokens)
        while buf_idx < len(tokens_cpu) or len(stack) > 1:
            if buf_idx < len(tokens_cpu):
                if len(stack) < 2:
                    stack.append(buf_idx)
                    buf_idx = buf_idx + 1
                    pred.append(0)
                    current_idx += 1
                    continue
                fn = [f[stack[-1]], f[stack[-2]]]
                if len(stack) >= 3:
                    fn.append(f[stack[-3]])
                else:
                    fn.append(zero_const)
                if buf_idx < len(tokens_cpu):
                    fn.append(f[buf_idx])
                else:
                    fn.append(zero_const)
            else:
                fn = [f[stack[-1]], ]
                if len(stack) >= 2:
                    fn.append(f[stack[-2]])
                else:
                    fn.append(zero_const)
                if len(stack) >= 3:
                    fn.append(f[stack[-3]])
                else:
                    fn.append(zero_const)
                fn.append(zero_const)
            fn = mx.nd.concat(fn[0], fn[1], fn[2], fn[3], dim=0).reshape((1, -1))
            output = parserModel.trans_pred(fn)
            
            if buf_idx == len(tokens_cpu):
                pred_action = output[0][1:].argmax(axis=0).asscalar() + 1
            else:
                pred_action = output[0].argmax(axis=0).asscalar()
            pred_action = int(pred_action)
            pred.append(pred_action)
            model_gt.append(tags[current_idx])
            model_pred.append(pred_action)

            current_tag = pred_action
            # Work as parser
            if current_tag == 0: #SHIFT
                stack.append(buf_idx)
                buf_idx = buf_idx + 1
            elif current_tag == 1: # LEFT-ARC
                s2 = stack.pop()
                s1 = stack.pop()
                stack.append(s2)
            elif current_tag == 2: #RIGHT-ARC
                s2 = stack.pop()
                s1 = stack.pop()
                stack.append(s1)
            current_idx += 1
        assert current_idx == len(tags)
    heads_pred = reconstrut_tree_with_transition_labels(sen, pred)
    deprel_pred = [deprel_map[None], ] # [ROOT] does not have head
    # get relation label
    for i in range(1, len(tokens_cpu)):
        deprel_f = mx.nd.concat(f[i], f[heads_pred[i]], dim=0).reshape((1, -1))
        deprel_f = parserModel.deprel_pred(deprel_f)
        deprel_pred.append(int(deprel_f[0].argmax(axis=0).asscalar()))
    uas += getUAS(heads_pred, sen, config.PUNC_POS_TAG)
    las += getLAS(heads_pred, deprel_pred, sen, deprel_map, config.PUNC_POS_TAG)
    # TODO calculating LAS
    pos_pred = mx.nd.argmax(pos_f, axis=1)
    pos_acc += (pos_pred == mx.nd.array(pos_tag, ctx=ctx)).sum().asscalar() -1 # remove root
    

    # accumulating statistics
    acc += (mx.nd.array(pred) == mx.nd.array(tags)).sum().asscalar()
    model_acc += (mx.nd.array(model_gt) == mx.nd.array(model_pred)).sum().asscalar()
    total_tags += len(tags)
    model_total_tags += len(model_gt)
    total_tokens += len(heads_pred) -1 

logging.info("Evaling: Pred acc={:.6} Model acc={:.6} UAS={:.6} LAS={:.6} POS acc={:.6}".format(acc/total_tags, model_acc/model_total_tags, uas/total_tokens, las/total_tokens, pos_acc/total_tokens))
logging.shutdown()
