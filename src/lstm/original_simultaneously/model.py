import chainer
from chainer import Variable
from chainer import cuda
import numpy as np
from chainer import Chain
from chainer import reporter
import chainer.links as L
import chainer.functions as F

import ipdb

def convert_seq(batch, device=None, with_label=True):
    def to_device_batch(batch):
        if device is None:
            return batch
        # else device < 0:
        else:
            return [chainer.dataset.to_device(device, x) for x in batch]
    if with_label:
        return {'xs': to_device_batch([x for x, _ in batch]),
                'ys': to_device_batch([y for _, y in batch])}
    else:
        return to_device_batch([x for x in batch])

class BiLSTMBase(Chain):
    def __init__(self, input_size, n_labels, n_layers=1, dropout=0.5):
        super(BiLSTMBase, self).__init__()
        with self.init_scope():
      # self.f_lstm = L.LSTM(None, feature_size, dropout)
      # self.b_lstm = L.LSTM(None, feature_size, dropout)
            self.nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=input_size, dropout=dropout)
            self.l1 = L.Linear(input_size*2, n_labels)

    def __call__(self, xs, ys):
        pred_ys = self.traverse(xs)
        
        loss = .0

        pred_ys = [pred_y.T for pred_y in pred_ys]
        ys = [y.argmax(axis=0) for y in ys]

        for pred_y, y in zip(pred_ys, ys):
            _loss = F.softmax_cross_entropy(pred_y, y)
            loss += _loss

        reporter.report({'loss': loss.data}, self)
        accuracy_ga = .0
        accuracy_o = .0
        accuracy_ni = .0
        accuracy_all = .0

        pred_ys = [pred_y.data.argmax(axis=1) for pred_y in pred_ys]
        for pred_y, y in zip(pred_ys, ys):
            if y[0] == pred_y[0]:
                accuracy_ga += 1/len(ys)
                accuracy_all += 1/len(ys)/3
            if y[1] == pred_y[1]:
                accuracy_o += 1/len(ys)
                accuracy_all += 1/len(ys)/3
            if y[2] == pred_y[2]:
                accuracy_ni += 1/len(ys)
                accuracy_all += 1/len(ys)/3

        reporter.report({'accuracy_ga': accuracy_ga}, self)
        reporter.report({'accuracy_o': accuracy_o}, self)
        reporter.report({'accuracy_ni': accuracy_ni}, self)
        reporter.report({'accuracy_all': accuracy_all}, self)

        return loss

    def traverse(self, xs):
        xs = [Variable(x) for x in xs]
        hx, cx = None, None
        hx, cx, ys = self.nstep_bilstm(xs=xs, hx=hx, cx=cx)
        return [self.l1(y) for y in ys]
