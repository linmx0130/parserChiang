#!/usr/bin/python3
#
# attention_parser_model.py
# A basic LSTM + Attention transition-based parser.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
# 

import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np

class TransPredModel(gluon.Block):
    def __init__(self, tag_count, inp_size, hidden_size, **kwargs):
        super(TransPredModel, self).__init__(*kwargs)
        self.fc1 = gluon.nn.Dense(hidden_size, in_units=inp_size, activation='tanh')
        self.fc2 = gluon.nn.Dense(tag_count, in_units=hidden_size)
    
    def forward(self, inp):
        f = self.fc1(inp)
        f = self.fc2(f)
        return f

class SelfAttentionBlock(gluon.Block):
    def __init__(self, inp_size, hidden_size, **kwargs):
        super(SelfAttentionBlock, self).__init__(*kwargs)
        self.fc1 = gluon.nn.Dense(hidden_size, in_units=inp_size)
        self.fc2 = gluon.nn.Dense(hidden_size, in_units=inp_size)
        self.fc3 = gluon.nn.Dense(inp_size, in_units=inp_size)
        self.softmax_normalize = np.sqrt(inp_size)

    def forward(self, inp):
        f1 = self.fc1(inp)
        f2 = self.fc2(inp)
        att = mx.nd.softmax(mx.nd.dot(f1, f2.T) / self.softmax_normalize)
        f3 = self.fc3(inp)
        f = (mx.nd.dot(att, f3) + inp) * 0.707
        return f

class ParserModel(gluon.Block):
    def __init__(self, vocab_size, num_embed, num_hidden, **kwargs):
        super(ParserModel, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = gluon.nn.Embedding(vocab_size, num_embed, weight_initializer=mx.init.Uniform(0.01))
            self.dropout = gluon.nn.Dropout(0.2)
            self.lstm1 = gluon.rnn.LSTM(num_hidden, 1, bidirectional=True, input_size=num_embed)
            self.atten = SelfAttentionBlock(num_hidden * 2, num_hidden)
            self.lstm2 = gluon.rnn.LSTM(num_hidden, 1, bidirectional=True, input_size=num_hidden *2)
            self.trans_pred= TransPredModel(3, num_hidden * 2*6, 20)
        self.num_hidden = num_embed

    def forward(self, inputs):
        embed = self.dropout(self.embed(inputs))
        s1, s2 = embed.shape
        embed = embed.reshape((s1, 1, s2))
        hidden = self.lstm1(embed)
        batch_size, __, hn_size = hidden.shape
        hidden = hidden.reshape((batch_size, hn_size))

        hidden = self.atten(hidden)

        hidden = hidden.reshape((batch_size, 1, hn_size))
        hidden = self.lstm2(hidden)
        hidden = hidden.reshape((batch_size, hn_size))
        return hidden
    
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


