#!/usr/bin/python3
# 
# marginloss.py
# Max margin loss implemented with CustomOp.
# NOT HINGE LOSS!!!
#
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
#
import mxnet as mx
import numpy as np
from mxnet.test_utils import get_mnist
from mxnet.io import NDArrayIter
import logging

class MarginLoss(mx.operator.CustomOp):
    def __init__(self, margin=1, l2reg=0):
        self.margin = float(margin)
        self.l2reg = float(l2reg)

    def forward(self, is_train, req, in_data, out_data, aux):
        # get margin loss
        x = in_data[0]
        label = in_data[1]
        ind = mx.nd.arange(0, x.shape[0])
        mlp_gt = x[ind, label]
        min_val = mx.nd.min(x)
        xx = x.copy()
        xx[ind, label] = min_val -10
        mlp_max = mx.nd.max(xx, axis=1)
        margin_loss = mlp_max - mlp_gt + self.margin
        margin_loss *= (margin_loss>0)
        self.assign(out_data[0], req[0], margin_loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0]
        label = in_data[1]
        ind = mx.nd.arange(0, x.shape[0])
        mlp_gt = x[ind, label]
        min_val = mx.nd.min(x)
        xx = x.copy()
        xx[ind, label] = min_val -10
        mlp_maxind = mx.nd.argmax(xx, axis=1)
        
        mask = out_data[0] > 0 
        mask = mask.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        
        grad = mx.nd.zeros_like(x)
        grad[ind, label] = -1
        grad[ind, mlp_maxind] = 1
        grad *= mask

        # derive reg loss
        grad += x * self.l2reg * 2 / x.shape[1]
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("marginloss")
class MarginLossProp(mx.operator.CustomOpProp):
    def __init__(self, margin=1, l2reg=0.0001):
        super(MarginLossProp, self).__init__(need_top_grad=False)
        self.margin = margin
        self.l2reg = l2reg
    def list_arguments(self):
        return ['data', 'label']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0], )
        output_shape = (in_shape[0][0], )
        return [data_shape, label_shape], [output_shape,], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype], [dtype], []
    def create_operator(self, ctx, shapes, types):
        return MarginLoss(self.margin, self.l2reg)

def max_margin_loss(data, label, margin=1, l2reg=0):
    return mx.nd.Custom(data, label, margin=margin, l2reg=l2reg, op_type="marginloss")

if __name__=="__main__":
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type='relu')
    fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type='relu')
    fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)
    mlp = mx.symbol.Custom(data=fc3, name='marginloss', op_type='marginloss')

    logging.basicConfig(level=logging.DEBUG)
    mnist_data = get_mnist()
    train_data = mnist_data['train_data']
    train_label = mnist_data['train_label']
    val_data = mnist_data['test_data']
    val_label = mnist_data['test_label']
    train = mx.io.NDArrayIter(train_data, train_label, batch_size=32, shuffle=True, last_batch_handle='discard', label_name='marginloss_label')
    val = mx.io.NDArrayIter(val_data, val_label, batch_size=50, shuffle=False, last_batch_handle='discard')
    ctx = mx.cpu()
    mod = mx.mod.Module([fc3, mlp], context=ctx, data_names=['data'], label_names=['marginloss_label'])

    metric = mx.metric.Accuracy(output_names=['fc3'], label_names='marginloss_label')
    mod.fit(train_data=train, eval_data=val, optimizer='sgd', eval_metric=metric,
            optimizer_params={'learning_rate':0.1, 'momentum':0.9, 'wd':0.0001}, 
            num_epoch=10, batch_end_callback=mx.callback.Speedometer(32, 50))
