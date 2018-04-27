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
        else:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([x.shape[0] for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev
    if with_label:
        return {'xs': to_device_batch([x for x, _ in batch]),
                'ys': to_device_batch([y for _, y in batch])}
    else:
        return to_device_batch([x for x in batch])

class BiLSTMBase(Chain):
    def __init__(self, input_size, n_labels, n_layers=1, dropout=0.5):
        super(BiLSTMBase, self).__init__()
        with self.init_scope():
            self.nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=input_size, dropout=dropout)
            self.l1 = L.Linear(input_size*2, n_labels)

    def __call__(self, xs, ys):
        pred_ys = self.traverse(xs)
        
        loss = .0
        for pred_y, y in zip(pred_ys, ys):
            _loss = F.softmax_cross_entropy(pred_y, y)
            loss += _loss
        reporter.report({'loss': loss.data}, self)

        accuracy = .0
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=0)[1] for pred_y in pred_ys]
        ys = [y.argmax(axis=0) for y in ys]
        for pred_y, y in zip(pred_ys, ys):
            if y == pred_y:
                accuracy += 1/len(ys)
        reporter.report({'accuracy': accuracy}, self)
        return loss

    def traverse(self, xs):
        xs = [Variable(x) for x in xs]
        hx, cx = None, None
        hx, cx, ys = self.nstep_bilstm(xs=xs, hx=hx, cx=cx)
        return [self.l1(y) for y in ys]
