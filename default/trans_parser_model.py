#!/usr/bin/python3
#
# trans_parser_model.py
# A basic LSTM transition-based parser.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
# 

import mxnet as mx
from mxnet import nd, autograd, gluon

class TransPredModel(gluon.Block):
    def __init__(self, tag_count, inp_size, hidden_size, **kwargs):
        super(TransPredModel, self).__init__(*kwargs)
        self.fc1 = gluon.nn.Dense(hidden_size, in_units=inp_size, activation='tanh')
        self.fc2 = gluon.nn.Dense(tag_count, in_units=hidden_size)
    
    def forward(self, inp):
        f = self.fc1(inp)
        f = self.fc2(f)
        return f

class ParserModel(gluon.Block):
    def __init__(self, vocab_size, num_embed, num_hidden, **kwargs):
        super(ParserModel, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = gluon.nn.Embedding(vocab_size, num_embed, weight_initializer=mx.init.Uniform(0.01))
            self.dropout = gluon.nn.Dropout(0.2)
            self.lstm = gluon.rnn.LSTM(num_hidden, 2, bidirectional=True, input_size=num_embed)
            self.trans_pred= TransPredModel(3, num_hidden * 2 * 4, 100)
        self.num_hidden = num_embed

    def forward(self, inputs):
        embed = self.dropout(self.embed(inputs))
        s1, s2 = embed.shape
        embed = embed.reshape((s1, 1, s2))
        hidden = self.lstm(embed)
        batch_size, __, hn_size = hidden.shape
        hidden = hidden.reshape((batch_size, hn_size))
        return hidden
    
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


