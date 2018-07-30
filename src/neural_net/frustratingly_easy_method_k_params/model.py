import chainer
from chainer import Variable
from chainer import cuda
import numpy as np
from chainer import Chain
from chainer import reporter
import chainer.links as L
import chainer.functions as F

def convert_seq(batch, device=None, with_label=True):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([x.shape[0] for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev
    if with_label:
        return {'xs': to_device_batch([x for x, _, _ in batch]),
                'ys': to_device_batch([y for _, y, _ in batch]),
                'zs': to_device_batch([z for _, _, z in batch])}
    else:
        return to_device_batch([x for x in batch])

class BiLSTMBase(Chain):
    def __init__(self, input_size, output_size, n_labels, n_layers=1, dropout=0.2, case='', device=0):
        super(BiLSTMBase, self).__init__()
        with self.init_scope():
            self.shared_nstep_bilstm = L.NStepBiLSTM(n_layers=n_layers, in_size=input_size, out_size=output_size, dropout=dropout)
            self.oc_nstep_bilstm = L.NStepBiLSTM(n_layers=n_layers, in_size=input_size, out_size=output_size, dropout=dropout)
            self.oy_nstep_bilstm = L.NStepBiLSTM(n_layers=n_layers, in_size=input_size, out_size=output_size, dropout=dropout)
            self.ow_nstep_bilstm = L.NStepBiLSTM(n_layers=n_layers, in_size=input_size, out_size=output_size, dropout=dropout)
            self.pb_nstep_bilstm = L.NStepBiLSTM(n_layers=n_layers, in_size=input_size, out_size=output_size, dropout=dropout)
            self.pm_nstep_bilstm = L.NStepBiLSTM(n_layers=n_layers, in_size=input_size, out_size=output_size, dropout=dropout)
            self.pn_nstep_bilstm = L.NStepBiLSTM(n_layers=n_layers, in_size=input_size, out_size=output_size, dropout=dropout)
            self.oc_l1 = L.Linear(input_size*4, n_labels)
            self.oy_l1 = L.Linear(input_size*4, n_labels)
            self.ow_l1 = L.Linear(input_size*4, n_labels)
            self.pb_l1 = L.Linear(input_size*4, n_labels)
            self.pm_l1 = L.Linear(input_size*4, n_labels)
            self.pn_l1 = L.Linear(input_size*4, n_labels)

    def __call__(self, xs, ys, zs):
        pred_ys = self.traverse(xs, zs)

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

    def traverse(self, xs, zs):
        hx, cx = None, None
        hx, cx, ys1 = self.shared_nstep_bilstm(xs=xs, hx=hx, cx=cx)
        hx, cx = None, None
        if zs[0] == 'OC':
            hx, cx, ys2 = self.oc_nstep_bilstm(xs=xs, hx=hx, cx=cx)
        elif zs[0] == 'OY':
            hx, cx, ys2 = self.oy_nstep_bilstm(xs=xs, hx=hx, cx=cx)
        elif zs[0] == 'OW':
            hx, cx, ys2 = self.ow_nstep_bilstm(xs=xs, hx=hx, cx=cx)
        elif zs[0] == 'PB':
            hx, cx, ys2 = self.pb_nstep_bilstm(xs=xs, hx=hx, cx=cx)
        elif zs[0] == 'PM':
            hx, cx, ys2 = self.pm_nstep_bilstm(xs=xs, hx=hx, cx=cx)
        elif zs[0] == 'PN':
            hx, cx, ys2 = self.pn_nstep_bilstm(xs=xs, hx=hx, cx=cx)
        ys = [F.concat((y1, y2)) for y1, y2 in zip(ys1, ys2)]
        if zs[0] == 'OC':
            ys = [self.oc_l1(y) for y in ys]
        elif zs[0] == 'OY':
            ys = [self.oy_l1(y) for y in ys]
        elif zs[0] == 'OW':
            ys = [self.ow_l1(y) for y in ys]
        elif zs[0] == 'PB':
            ys = [self.pb_l1(y) for y in ys]
        elif zs[0] == 'PM':
            ys = [self.pm_l1(y) for y in ys]
        elif zs[0] == 'PN':
            ys = [self.pn_l1(y) for y in ys]

        return ys
