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
    def __init__(self, input_size, output_size, n_labels, n_layers=1, dropout=0.3, case='', device=0):
        super(BiLSTMBase, self).__init__()
        with self.init_scope():
            self.shared_nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=output_size, dropout=dropout)
            self.oc_nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=output_size, dropout=dropout)
            self.oy_nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=output_size, dropout=dropout)
            self.ow_nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=output_size, dropout=dropout)
            self.pb_nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=output_size, dropout=dropout)
            self.pm_nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=output_size, dropout=dropout)
            self.pn_nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=output_size, dropout=dropout)
            self.l1 = L.Linear(input_size*4, n_labels)

        sentence_length = 2000

        domain_statistics_positive = {}

        tmp = np.full((sentence_length, ), type_statistics_dict['union'][-1])
        for i in range(4):
            tmp[i] = type_statistics_dict['union'][i]
        tmp = np.diag(tmp)
        union_I = np.matrix(tmp, dtype=np.float32).I

        for domain in type_statistics_dict.keys():
            if domain == 'union':
                continue
            statistics = type_statistics_dict[domain]
            tmp = np.full((sentence_length, ), statistics[-1])
            for i in range(0, 4):
                tmp[i] = statistics[i]
            tmp = np.diag(tmp)
            domain_statistics_positive[domain] = np.matrix(tmp, dtype=np.float32)*union_I
            domain_statistics_positive[domain] = chainer.dataset.to_device(device, domain_statistics_positive[domain])

        for domain in type_statistics_dict.keys():
            type_statistics_dict[domain] = [100-type_statistics_dict[domain][i] for i in range(5)]

        domain_statistics_negative = {}

        tmp = np.full((sentence_length, ), type_statistics_dict['union'][-1])
        for i in range(4):
            tmp[i] = type_statistics_dict['union'][i]
        tmp = np.diag(tmp)
        union_neg_I = np.matrix(tmp, dtype=np.float32).I

        for domain in type_statistics_dict.keys():
            if domain == 'union':
                continue
            statistics = type_statistics_dict[domain]
            tmp = np.full((sentence_length, ), statistics[-1])
            for i in range(0, 4):
                tmp[i] = statistics[i]
            tmp = np.diag(tmp)
            domain_statistics_negative[domain] = np.matrix(tmp, dtype=np.float32)*union_neg_I
            domain_statistics_negative[domain] = chainer.dataset.to_device(device, domain_statistics_negative[domain])

        self.domain_statistics_positive = domain_statistics_positive
        self.domain_statistics_negative = domain_statistics_negative

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
        for z in zs:
            if z != zs[0]:
                print('ERROR!!!!!', flush=True)
        ys = [F.concat((y1, y2)) for y1, y2 in zip(ys1, ys2)]
        ys = [self.l1(y) for y in ys]
        ys_neg = [F.matmul(self.domain_statistics_negative[z][:y.shape[0], :y.shape[0]], y[:, 0].reshape(-1, 1)) for y, z in zip(ys, zs)]
        ys_pos = [F.matmul(self.domain_statistics_positive[z][:y.shape[0], :y.shape[0]], y[:, 1].reshape(-1, 1)) for y, z in zip(ys, zs)]
        ys = [F.concat((y_neg, y_pos)) for y_neg, y_pos in zip(ys_neg, ys_pos)]
        return ys
