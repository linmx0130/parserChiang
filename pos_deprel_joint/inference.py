#!/usr/bin/python3
# 
# inference.py
# A basic lstm transition-based parser. Inference code.
# Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
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

args_parser = inferencerArgumentParser()
args = args_parser.parse_args()

current_time = init_logging("inference")
model_dump_path = args.model_path

if args.input_format == 'ud':
    data = ud_dataloader.parseDocument(args.input_file)
    data = [t for t in data if cross_check(t.tokens)]
elif args.input_format == 'raw':
    # Use NLTK to tokenize data
    data = ud_dataloader.nltkParseDocument(args.input_file)

writer = ud_dataloader.UDWriter(args.inference_to)

for sen in data:
    for token in sen.tokens:
        token.form = token.form.lower()

def inverse_map(input_map):
    ret = {}
    for k in input_map:
        ret[input_map[k]] = k
    return ret

# load word map
with open(os.path.join(model_dump_path, 'word_map.pkl'), 'rb') as f:
    word_map = pickle.load(f)
    pos_map = pickle.load(f)
    deprel_map = pickle.load(f)
inv_word_map = inverse_map(word_map)
inv_pos_map = inverse_map(pos_map)
inv_deprel_map = inverse_map(deprel_map)

logging.info("Input data loaded: {}".format(args.input_file))
logging.info("Sentences count = {}".format(len(data)))
logging.info("Words count = {}".format(len(word_map)))
logging.info("POS Tag count = {}".format(len(pos_map)))
logging.info("Dependent Relation count = {}".format(len(deprel_map)))
logging.info("Inference output = {}".format(args.inference_to))

if args.use_cpu:
    ctx = mx.cpu(0)
else:
    ctx = mx.gpu(0)

parserModel = ParserModel(len(word_map), config.NUM_EMBED, config.NUM_HIDDEN, len(pos_map), config.TAG_EMBED, len(deprel_map))
model_file = os.path.join(model_dump_path, args.model_file)
parserModel.load_params(model_file, ctx=ctx)
logging.info("Model loaded: {}".format(model_file))

zero_const = mx.nd.zeros(shape=config.NUM_HIDDEN * 2, ctx=ctx)

# eval
print("Evaluating...")

for seni in tqdm(range(len(data))):
    sen = data[seni]
    tokens_cpu = mapTokenToId(sen, word_map)
    tokens = mx.nd.array(tokens_cpu, ctx)
    
    model_output = []
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
    heads_pred = reconstrut_tree_with_transition_labels(sen, pred)
    deprel_pred = [deprel_map[None], ] # [ROOT] does not have head
    # get relation label
    for i in range(1, len(tokens_cpu)):
        deprel_f = mx.nd.concat(f[i], f[heads_pred[i]], dim=0).reshape((1, -1))
        deprel_f = parserModel.deprel_pred(deprel_f)
        deprel_pred.append(int(deprel_f[0].argmax(axis=0).asscalar()))
    pos_pred = mx.nd.argmax(pos_f, axis=1).astype('int')
    for i in range(len(sen)):
        sen.tokens[i].pos_tag = inv_pos_map[pos_pred[i].asscalar()]
        sen.tokens[i].x_pos_tag = inv_pos_map[pos_pred[i].asscalar()]
        sen.tokens[i].head = heads_pred[i]
        sen.tokens[i].deprel = inv_deprel_map[deprel_pred[i]]
    writer.write_sentence(sen)

writer.close()
logging.shutdown()
