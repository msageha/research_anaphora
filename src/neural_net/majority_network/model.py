import chainer
from chainer import Variable
from chainer import cuda
import numpy as np
from chainer import Chain
from chainer import reporter
import chainer.links as L
import chainer.functions as F

class MajorityNetwork(Chain):
    def __init__(self, input_size, n_labels):
        super(MajorityNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(input_size, n_labels)
    
    def __call__(self, xs, ys):
        pred_ys = self.traverse(xs)

        loss = .0
        for pred_y, y in zip(pred_ys, ys):
            _loss = F.softmax_cross_entropy(pred_y, y)
            loss += _loss/len(ys)
        accuracy = .0
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=0)[1] for pred_y in pred_ys]
        ys = [y.argmax(axis=0) for y in ys]
        for pred_y, y in zip(pred_ys, ys):
            if y == pred_y:
                accuracy += 1/len(ys)
        return loss, accuracy

    def traverse(self, xs):
        ys = [self.l1(x) for x in xs]
        return ys