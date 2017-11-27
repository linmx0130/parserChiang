#!/usr/bin/python3
# 
# train_parser.py
# A basic LSTM transition-based parser.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
#

import ud_dataloader
import mxnet as mx
from mxnet import nd, autograd, gluon
from config import train_data_fn 
import config
from get_trans import cross_check
import random
from trans_parser_model import ParserModel
from utils import * 

init_logging()
data = ud_dataloader.parseDocument(train_data_fn)
data = [t for t in data if cross_check(t.tokens) and len(t) > 4]
words, pos_tag = getWordPos(data)
word_list = sorted(list(words.keys()))
word_map = {}
for i, w in enumerate(word_list):
    word_map[w] = i

logging.info("Train data loaded: {}".format(train_data_fn))
logging.info("Sentences count = {}".format(len(data)))
logging.info("Words count = {}".format(len(word_map)))

ctx = mx.gpu(0)
parserModel = ParserModel(len(word_list), 50, 50)
parser_params = parserModel.collect_params()
parser_params.initialize(mx.init.Xavier(), ctx=ctx)
logging.info("Parameters initialized: {}".format(str(parser_params)))

zero_const = mx.nd.random_uniform(-0.01, 0.01, shape=(1, 100), ctx=ctx)

trainer = gluon.Trainer(parser_params, 'adam', {'learning_rate': 0.005, 'wd':1e-8})
loss = gluon.loss.SoftmaxCrossEntropyLoss()


for epoch in range(1, 1000+1):
    random.shuffle(data)
    avg_loss = 0.0
    acc_accu = 0.0
    acc_total = 0
    
    #training 
    for seni, sen in enumerate(data):
        tokens_cpu = mapTokenToId(sen, word_map)
        tokens = mx.nd.array(tokens_cpu, ctx)
        tags = mapTransTagToId(sen)
        
        model_output = []
        model_gt = []
        model_pred = []
        buf_idx = 0
        stack = []
        pred = []
        current_idx = 0

        with autograd.record():
            f = parserModel(tokens)
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
                    for i in range(3):
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
                    fn.append(zero_const)
                #fn = mx.nd.concat(fn[0], fn[1], fn[2], fn[0]*fn[1], fn[0]*fn[2], fn[1]*fn[2], dim=1)
                fn = mx.nd.concat(fn[0], fn[1], fn[2], fn[0]*fn[1], fn[0]*fn[2], fn[1]*fn[2], dim=0).reshape((1, -1))
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
            L = loss(model_output[0], model_gt[0])
            for i in range(1, len(model_output)):
                out = model_output[i]
                gt = model_gt[i]
                L = L + loss(out, gt)
        L.backward()
        trainer.step(1)
        
        acc_accu += (mx.nd.array(model_gt)==mx.nd.array(model_pred)).sum().asscalar()
        acc_total += len(model_pred)
        avg_loss += L.asscalar() / len(model_output)
        if seni % config.prompt_inteval == config.prompt_inteval - 1:
            avg_loss /= config.prompt_inteval
            acc = acc_accu / acc_total
            logging.info("Epoch {} sen {} loss={:.6} train acc={:.6}".format(epoch, seni, avg_loss, acc))
            avg_loss = 0
            acc_accu = 0
            acc_total = 0
    
    # eval
    acc = 0
    total_tags = 0
    model_acc = 0
    model_total_tags = 0
    for seni, sen in enumerate(data):
        tokens_cpu = mapTokenToId(sen, word_map)
        tokens = mx.nd.array(tokens_cpu, ctx)
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
            f = parserModel(tokens)
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
                   for i in range(3):
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
                    fn.append(zero_const)
                #fn = mx.nd.concat(fn[0], fn[1], fn[2], fn[0]*fn[1], fn[0]*fn[2], fn[1]*fn[2], dim=1)
                fn = mx.nd.concat(fn[0], fn[1], fn[2], fn[0]*fn[1], fn[0]*fn[2], fn[1]*fn[2], dim=0).reshape((1, -1))
                output = parserModel.trans_pred(fn)
                pred.append(output[0].argmax(axis=0).asscalar())
                model_gt.append(tags[current_idx])
                model_pred.append(output[0].argmax(axis=0).asscalar())

                current_tag = tags[current_idx]
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
        acc += (mx.nd.array(pred) == mx.nd.array(tags)).sum().asscalar()
        model_acc += (mx.nd.array(model_gt) == mx.nd.array(model_pred)).sum().asscalar()
        total_tags += len(tags)
        model_total_tags += len(model_gt)
    logging.info("Evaling: Total tag acc = {:.6}, prediction tag acc = {:.6}".format(acc/total_tags, model_acc/model_total_tags))
