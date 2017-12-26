#!/usr/bin/python3
# 
# train_parser.py
# A basic LSTM transition-based parser. Training code.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
#

import config
import ud_dataloader
import mxnet as mx
from mxnet import nd, autograd, gluon
from config import train_data_fn 
from get_trans import cross_check
import random
from trans_parser_pos_model import ParserModel
import os
from utils import * 
import pickle

argsparser = trainerArgumentParser()
args = argsparser.parse_args()

current_time = init_logging("train")
model_dump_path = 'model_dumps_{}{:02}{:02}_{:02}_{:02}_{:02}/'.format(
        current_time.tm_year,
        current_time.tm_mon,
        current_time.tm_mday,
        current_time.tm_hour,
        current_time.tm_min,
        current_time.tm_sec)

if not os.path.exists(model_dump_path):
    os.mkdir(model_dump_path)
    logging.info("Model dump path: {}".format(model_dump_path))    

data = ud_dataloader.parseDocument(train_data_fn)
data = [t for t in data if cross_check(t.tokens) and len(t) > 4]
# data lowerize
for sen in data:
    for token in sen.tokens:
        token.form = token.form.lower()
    ud_dataloader.mask_pos_with_x(sen) 

words_count, pos_list = getWordPos(data)
deprel_list = getDeprelList(data)
words_count[config.UNKNOW_TOKEN] = 0
word_list = sorted(list(words_count.keys()))
word_map = {}
for i, w in enumerate(word_list):
    word_map[w] = i
pos_map = {}
for i, w in enumerate(pos_list):
    pos_map[w] = i
deprel_map = {}
for i, w in enumerate(deprel_list):
    deprel_map[w] = i

with open(os.path.join(model_dump_path, 'word_map.pkl'),'wb') as f:
    pickle.dump(word_map, f)
    pickle.dump(pos_map, f)
    pickle.dump(deprel_map, f)

logging.info("Dumped word map to word_map.pkl")
logging.info("Train data loaded: {}".format(train_data_fn))
logging.info("Sentences count= {}".format(len(data)))
logging.info("Words count = {}".format(len(word_map)))
logging.info("POS Tag count = {}".format(len(pos_map)))
logging.info("Dependent Relation count = {}".format(len(deprel_map)))

if args.use_cpu:
    ctx = mx.cpu(0)
else:
    ctx = mx.gpu(0)
parserModel = ParserModel(len(word_list), config.NUM_EMBED, config.NUM_HIDDEN, len(pos_list), config.TAG_EMBED, len(deprel_map))
parser_params = parserModel.collect_params()
parser_params.initialize(mx.init.Xavier(), ctx=ctx)
if args.wordvec is not None:
    logging.info("Loading wordvec from {}".format(args.wordvec))
    setEmbeddingWithWordvec(parserModel.embed, word_map, args.wordvec)

zero_const = mx.nd.zeros(shape=config.NUM_HIDDEN*2, ctx=ctx)

trainer = gluon.Trainer(parser_params, args.trainer, 
                        getDefaultTrainerHyperparams(args.trainer))
loss = gluon.loss.SoftmaxCrossEntropyLoss()

for epoch in range(1, 1000+1):
    random.shuffle(data)
    avg_loss = 0.0
    avg_pos_loss = 0.0
    avg_deprel_loss = 0.0
    acc_accu = 0.0
    acc_total = 0
    L = mx.nd.zeros(1, ctx=ctx) 
    pos_loss = mx.nd.zeros(1, ctx=ctx)
    total_L = mx.nd.zeros(1, ctx=ctx)
    deprel_L = mx.nd.zeros(1, ctx=ctx)
    #training 
    for seni, sen in enumerate(data):
        tokens_cpu = mapTokenToId(sen, word_map, config.WORD_DROPOUT_RATE, words_count)
        tokens = mx.nd.array(tokens_cpu, ctx)
        pos_tag_cpu = mapPosTagToId(sen, pos_map)
        pos_tag = mx.nd.array(pos_tag_cpu, ctx)
        deprel_tag_cpu = mapDeprelTagToId(sen, deprel_map)
        deprel_tag = mx.nd.array(deprel_tag_cpu, ctx)
        head_of_tokens = [t.head for t in sen.tokens]
        tags = mapTransTagToId(sen)
        
        model_output = []
        model_gt = []
        model_pred = []
        buf_idx = 0
        stack = []
        pred = []
        current_idx = 0

        with autograd.record():
            f, pos_f = parserModel(tokens)
            # POS Tagger loss
            pos_loss = pos_loss + loss(pos_f, pos_tag).mean()

            # parse by transition
            while buf_idx < len(tokens_cpu) or len(stack) > 1:
                if buf_idx < len(tokens_cpu):
                    if len(stack) < 2:
                        stack.append(buf_idx)
                        buf_idx = buf_idx + 1
                        pred.append(0)
                        assert tags[current_idx] == 0
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
                pred.append(output[0].argmax(axis=0).asscalar())
                current_tag = tags[current_idx]
                model_output.append(output)
                model_gt.append(tags[current_idx])
                model_pred.append(output[0].argmax(axis=0).asscalar())

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
            # get loss
            model_gt = mx.nd.array(model_gt, ctx=ctx)
            for i in range(0, len(model_output)):
                out = model_output[i]
                gt = model_gt[i]
                L = L + loss(out, gt)
            # train deprel model: concatenating head word and the child.
            # ignore index 0: ROOT does not have head
            for i in range(1, len(tokens_cpu)):
                deprel_f = mx.nd.concat(f[i], f[head_of_tokens[i]], dim=0).reshape((1, -1))
                deprel_f = parserModel.deprel_pred(deprel_f)
                deprel_L = deprel_L + loss(deprel_f, deprel_tag[i])
            total_L = total_L + L + pos_loss + deprel_L

        if (seni + 1) % config.UPDATE_STEP == 0:
            total_L.backward()
            trainer.step(1)
            L = mx.nd.zeros(1, ctx=ctx) 
            pos_loss = mx.nd.zeros(1, ctx=ctx)
            deprel_L = mx.nd.zeros(1, ctx=ctx)
            total_L = mx.nd.zeros(1, ctx=ctx)
        
        acc_accu += (mx.nd.array(model_gt)==mx.nd.array(model_pred)).sum().asscalar()
        acc_total += len(model_pred)
        avg_loss += L.asscalar() / len(model_output)
        avg_pos_loss += pos_loss.asscalar() / len(pos_tag)
        avg_deprel_loss += deprel_L.asscalar() / (len(head_of_tokens) - 1)

        if seni % config.prompt_inteval == config.prompt_inteval - 1:
            avg_loss /= config.prompt_inteval * config.UPDATE_STEP
            avg_pos_loss /= config.prompt_inteval * config.UPDATE_STEP
            avg_deprel_loss /= config.prompt_inteval * config.UPDATE_STEP
            acc = acc_accu / acc_total
            logging.info("Epoch {} sen {} POS loss={:.6} Dep loss={:.6} loss={:.6} train acc={:.6}".format(epoch, seni, avg_pos_loss, avg_deprel_loss, avg_loss, acc))

            avg_loss = 0
            acc_accu = 0
            acc_total = 0
            avg_pos_loss = 0
            avg_deprel_loss = 0
    
    model_file = os.path.join(model_dump_path, "epoch-{}.gluonmodel".format(epoch))
    parserModel.save_params(model_file)
    logging.info("Model dumped to {}".format(model_file))
