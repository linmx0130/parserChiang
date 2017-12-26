#!/usr/bin/python3
#
# trans_parser_model.py
# A basic LSTM transition-based parser.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
# 

import mxnet as mx
from mxnet import nd, autograd, gluon

class SelfAttentionBlock(gluon.Block):
    def __init__(self, inp_size, hidden_size, **kwargs):
        super(SelfAttentionBlock, self).__init__(*kwargs)
        with self.name_scope():
            self.lin1 = gluon.nn.Dense(hidden_size, in_units=inp_size)
            self.lin2 = gluon.nn.Dense(hidden_size, in_units=inp_size)
            self.nonl = gluon.nn.Dense(1, in_units=hidden_size * 2, activation='relu')
            #self.lin3 = gluon.nn.Dense(inp_size, in_units=inp_size, activation='tanh')

    def forward(self, inp):
        f1 = self.lin1(inp)
        f2 = self.lin2(inp)
        n, h = f1.shape
        f1 = f1.reshape((1,n,h)).broadcast_axes(axis=0, size=n).reshape((n*n, h))
        f2 = f2.reshape((n,1,h)).broadcast_axes(axis=1, size=n).reshape((n*n, h))
        f12 = self.nonl(mx.nd.concat(f1, f2, dim=1)).reshape((n,n)) / n
        f = (mx.nd.dot(f12, inp) + inp) * 0.707
        return f


class TransPredModel(gluon.Block):
    def __init__(self, tag_count, inp_size, hidden_size, f_count, **kwargs):
        super(TransPredModel, self).__init__(*kwargs)
        self.atten = SelfAttentionBlock(inp_size, int(inp_size / 2))
        self.fc1 = gluon.nn.Dense(hidden_size, in_units=inp_size * f_count, activation='tanh')
        self.fc2 = gluon.nn.Dense(tag_count, in_units=hidden_size)
        self.f_count = f_count
    
    def forward(self, inp):
        _, fsize = inp.shape
        f = inp.reshape((self.f_count, int(fsize/self.f_count)))
        f = self.atten(f)
        f = f.reshape((1, fsize))
        f = self.fc1(f)
        f = self.fc2(f)
        return f


class SimpleMLPModel(gluon.Block):
    def __init__(self, tag_count, inp_size, hidden_size, **kwargs):
        super(SimpleMLPModel, self).__init__(*kwargs)
        self.fc1 = gluon.nn.Dense(hidden_size, in_units=inp_size, activation='tanh')
        self.fc2 = gluon.nn.Dense(tag_count, in_units=hidden_size)
    
    def forward(self, inp):
        f = self.fc1(inp)
        f = self.fc2(f)
        return f


class ParserModel(gluon.Block):
    def __init__(self, vocab_size, num_embed, num_hidden, tag_count, tag_embed_size, num_deprel, **kwargs):
        super(ParserModel, self).__init__(**kwargs)
        with self.name_scope():
            # Word and POS tag embed
            self.embed = gluon.nn.Embedding(vocab_size, num_embed, weight_initializer=mx.init.Uniform(0.1))
            self.dropout = gluon.nn.Dropout(0.2)
            self.tag_embed = gluon.nn.Embedding(tag_count, tag_embed_size, weight_initializer=mx.init.Uniform(0.1))
            # POS tagger
            self.lstm_tag = gluon.rnn.LSTM(num_hidden, 2, bidirectional=True, input_size=num_embed)
            self.tag_cls = gluon.nn.Dense(tag_count, in_units=num_hidden*2)
            # Parser
            self.lstm_parse = gluon.rnn.LSTM(num_hidden, 2, bidirectional=True, input_size=num_embed+tag_embed_size)
            self.trans_pred= TransPredModel(3, num_hidden * 2, 100, 4)
            self.deprel_pred = SimpleMLPModel(num_deprel, num_hidden * 2 * 2, 100)
            self.fusion_transformer = SelfAttentionBlock(num_hidden * 2, num_hidden)

        self.num_hidden = num_embed

    def forward(self, inputs_word):
        embed = self.dropout(self.embed(inputs_word))
        s1, s2 = embed.shape
        embed = embed.reshape((s1, 1, s2))
        # Get pos tag embed
        tag_hidden = self.lstm_tag(embed)
        s1, __, s2 = tag_hidden.shape
        tag_hidden = tag_hidden.reshape((s1, s2))
        tag_f = self.tag_cls(tag_hidden)
        with autograd.pause():
            tag_pred = mx.nd.argmax(tag_f, axis=1)
        tag_embed = self.tag_embed(tag_pred)
        
        ts1, ts2 = tag_embed.shape
        tag_embed = tag_embed.reshape((ts1, 1, ts2))
        embed = mx.nd.concat(embed, tag_embed, dim=2)

        hidden = self.lstm_parse(embed)
        batch_size, __, hn_size = hidden.shape
        hidden = hidden.reshape((batch_size, hn_size))
        
        return hidden, tag_f
    
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


