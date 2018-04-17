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

class BiGRU(Chain):
    def __init__(self, input_size, n_labels, n_layers=1, dropout=0.3):
        super(BiGRU, self).__init__()
        with self.init_scope():
      # self.f_lstm = L.LSTM(None, feature_size, dropout)
      # self.b_lstm = L.LSTM(None, feature_size, dropout)
            self.nstep_bigru = L.NStepBiGRU(n_layers=n_layers, in_size=input_size, out_size=input_size, dropout=dropout)
            self.l1 = L.Linear(input_size*2, n_labels)
        self.dropout = dropout

    def __call__(self, xs, ys):
        pred_ys = self.traverse(xs)

        loss = .0
        for pred_y, y in zip(pred_ys, ys):
            pred_y = F.relu(pred_y)
            _loss = F.sigmoid_cross_entropy(pred_y.reshape(1, -1), y.reshape(1,-1))
            loss += _loss
        reporter.report({'loss': loss.data}, self)

        # accuracy = {'ga':0., 'o':0., 'ni':0., 'all':0.}
        precision = {'ga':0., 'o':0., 'ni':0., 'all':0.}
        recall = {'ga':0., 'o':0., 'ni':0., 'all':0.}
        f1 = {'ga':0., 'o':0., 'ni':0., 'all':0.}

        all_correct_num = 0
        all_predict_num = 0
        all_hit_num = 0
        for index, case in {2:'ga', 3:'o', 4:'ni'}.items():
            correct_num = 0
            predict_num = 0
            hit_num = 0
            for pred_y, y in zip(pred_ys, ys):
                pred_y = pred_y.data.T
                y = y.T
                
                if y[index].max() != 0:
                    correct_num += 1
                if pred_y[index].max() != 0:
                    predict_num += 1
                if y[index].max() != 0 and pred_y[index].max() != 0 and y[index].argmax() == pred_y[index].argmax():
                    hit_num += 1
            precision[case] = hit_num/correct_num
            recall[case] = hit_num/predict_num
            f1[case] = (2*hit_num)/(2*hit_num+correct_num+predict_num)
            all_correct_num += correct_num
            all_predict_num += predict_num
            all_hit_num += hit_num
        precision['all'] = all_hit_num/all_correct_num
        recall['all'] = all_hit_num/all_predict_num
        f1['all'] = (2*all_hit_num)/(2*all_hit_num+all_correct_num+all_predict_num)

        # pred_y.shape ==  (9, 5)
        # y.shape == (9, 5)
        # pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]

        reporter.report({'accuracy': loss.data}, self)
        reporter.report({'precision': loss.data}, self)
        reporter.report({'recall': loss.data}, self)
        reporter.report({'f1': loss.data}, self)

        return loss

    def traverse(self, xs):
        xs = [Variable(x) for x in xs]
        hx = None
        hx, ys = self.nstep_bigru(xs=xs, hx=hx)
        return [self.l1(y) for y in ys]

