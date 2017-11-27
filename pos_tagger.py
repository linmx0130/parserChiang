#!/usr/bin/python3
# 
# pos_tagger.py
# A basic LSTM POS Tagger.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
#

import ud_dataloader
import mxnet as mx
from mxnet import nd, autograd, gluon
from config import train_data_fn as train_data
import config
import random

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


class TaggerModel(gluon.Block):
    def __init__(self, vocab_size, num_embed, num_hidden, tag_count, **kwargs):
        super(TaggerModel, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = gluon.nn.Embedding(vocab_size, num_embed, weight_initializer=mx.init.Uniform(0.1))
            self.lstm = gluon.rnn.LSTM(num_hidden, 1, bidirectional=True, input_size=num_embed)
            self.tag_cls = gluon.nn.Dense(tag_count, in_units=num_hidden*2)
        self.num_hidden = num_embed
        self.tag_count = tag_count

    def forward(self, inputs):
        embed = self.embed(inputs)
        s1, s2 = embed.shape
        embed = embed.reshape((s1, 1, s2))
        hidden = self.lstm(embed)
        batch_size, __, hn_size = hidden.shape
        hidden.reshape((batch_size, hn_size))
        cls = self.tag_cls(hidden)
        return cls
    
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

def mapTokenToId(sen: ud_dataloader.UDSentence, word_map:dict):
    ret = []
    for item in sen.tokens:
        ret.append(word_map[item.form])
    return ret

def mapTagToId(sen: ud_dataloader.UDSentence, tag_map:dict):
    ret = []
    for item in sen.tokens:
        ret.append(tag_map[item.pos_tag])
    return ret


data = ud_dataloader.parseDocument(train_data)
words, pos_tag = getWordPos(data)
word_list = sorted(list(words.keys()))
word_map = {}

for i, w in enumerate(word_list):
    word_map[w] = i
pos_tag_map = {}
for i, t in enumerate(pos_tag):
    pos_tag_map[t] = i

ctx = mx.gpu(0)
tagger = TaggerModel(len(word_list), 50, 50, len(pos_tag))
tagger.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(tagger.collect_params(), 'adam', {'learning_rate': 0.01})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

for epoch in range(1, 10+1):
    random.shuffle(data)
    avg_loss = 0.0
    acc_accu = 0.0
    acc_total = 0
    for i, sen in enumerate(data):
        tokens = mapTokenToId(sen, word_map)
        tokens = mx.nd.array(tokens, ctx)
        tags = mapTagToId(sen, pos_tag_map)
        tags = mx.nd.array(tags, ctx)
        with autograd.record():
            outputs = tagger(tokens)
            pred = outputs.argmax(axis=1)
            acc_accu += (tags==pred).sum().asscalar()
            acc_total += outputs.shape[0]
            L = loss(outputs, tags)
            L = L.mean()
            L.backward()
        trainer.step(1)
        avg_loss += L.asscalar()
        if i % config.prompt_inteval == 0:
            avg_loss /= config.prompt_inteval
            acc = acc_accu / acc_total
            print("Epoch {} sen {} loss={} train acc={}".format(epoch, i, avg_loss, acc))
            avg_loss = 0
            acc_accu = 0
            acc_total = 0
