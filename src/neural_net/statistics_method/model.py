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
                'zs': [z for _, _, z in batch]}
    else:
        return to_device_batch([x for x in batch])

class BiLSTMBase(Chain):
    def __init__(self, input_size, output_size, n_labels, n_layers=1, dropout=0.5, case='', device=0):
        super(BiLSTMBase, self).__init__()
        with self.init_scope():
            self.nstep_bilstm = L.NStepBiLSTM(n_layers=n_labels, in_size=input_size, out_size=input_size, dropout=dropout)
            self.l1 = L.Linear(input_size*2, n_labels)
        sentence_length = 2000
        
        statistics_union_positive = {'ga':[39.555, 2.3408, 0.97449, 9.3984, 33.922], 'o':[71.605,  0.030492, 0.010713, 7.3097, 24.3518], 'ni':[80.339, 0.12609, 0.0684, 1.5258, 8.9406]}
        statistics_OC = {'ga':[32.595, 9.1177, 6.2506, 71.522, 30.2101], 'o':[71.522, 0.13032, 0.082054, 71.522, 19.9631], 'ni':[84.511, 4, 1.0957, 0.56473, 3.7697, 10.0585]}
        statistics_OY = {'ga':[34.117, 14.779, 1.9571, 4.176, 31.0858], 'o':[75.724, 0.17991, 0.0054517, 1.4665, 19.9145], 'ni':[89.822, 0.28894, 0.13629, 1.4665, 8.28562]}
        statistics_OW = {'ga':[45.003, 0.16492, 0.017259, 4.2207, 28.279], 'o':[66.956, 0, 0, 1.1736, 28.824], 'ni':[92.274, 0.0019176, 0, 1.1736, 6.5506]}
        statistics_PB = {'ga':[39.198, 0.48589, 0.21272, 3.7976, 37.817], 'o':[72.81, 0, 0.0067174, 2.219, 23.3854], 'ni':[86.852, 0.013435, 0.013435, 2.219, 10.9023]}
        statistics_PM = {'ga':[40.045, 1.1111, 0.87591, 3.0789, 33.5531], 'o':[73.766, 0.019198, 0.0071993, 0.86871, 23.1283], 'ni':[89.439, 0.028797, 0.0071993, 0.86871, 9.65662]}
        statistics_PN = {'ga':[38.869, 0.4835, 0.37145, 3.1389, 37.981], 'o':[71.986, 0.0092095, 0.0030698, 1.056, 24.8631], 'ni':[90.031, 0.010744, 0.023024, 1.056, 8.87954]}

        statistics_dict_positive = {'OC':statistics_OC, 'OY':statistics_OY, 'OW':statistics_OW, 'PB':statistics_PB, 'PM':statistics_PM, 'PN':statistics_PN}

        domain_statistics_positive = {}

        tmp = np.full((sentence_length, ), statistics_union_positive[case][-1])
        for i in range(0, 4):
            tmp[i] = statistics_union_positive[case][i]
        tmp = np.diag(tmp)
        domain_statistics_positive['union.I'] = np.matrix(tmp, dtype=np.float32).I

        for domain in ['OC', 'OY', 'OW', 'PB', 'PN', 'PM']:
            statistics = statistics_dict_positive[domain]
            tmp = np.full((sentence_length, ), statistics[case][-1])
            for i in range(0, 4):
                tmp[i] = statistics[case][i]
            tmp = np.diag(tmp)
            domain_statistics_positive[domain] = np.matrix(tmp, dtype=np.float32)*domain_statistics_positive['union.I']
            domain_statistics_positive[domain] = chainer.dataset.to_device(device, domain_statistics_positive[domain])

        statistics_dict_negative = {'OC':statistics_OC, 'OY':statistics_OY, 'OW':statistics_OW, 'PB':statistics_PB, 'PM':statistics_PM, 'PN':statistics_PN}
        for key in statistics_dict_negative:
            for case in ['ga', 'o', 'ni']:
                statistics_dict_negative[key][case] = [100-statistics_dict_negative[key][case][i] for i in range(0, 5)]
        statistics_union_negative = {'ga':[39.555, 2.3408, 0.97449, 9.3984, 33.922], 'o':[71.605,  0.030492, 0.010713, 7.3097, 24.3518], 'ni':[80.339, 0.12609, 0.0684, 1.5258, 8.9406]}
        for case in ['ga', 'o', 'ni']:
            statistics_union_negative[case] = [100-statistics_union_negative[case][i] for i in range(0, 5)]
        
        domain_statistics_negative = {}
        
        tmp = np.full((sentence_length, ), statistics_union_negative[case][-1])
        for i in range(0, 4):
            tmp[i] = statistics_union_negative[case][i]
        tmp = np.diag(tmp)
        domain_statistics_negative['union.I'] = np.matrix(tmp, dtype=np.float32).I

        for domain in ['OC', 'OY', 'OW', 'PB', 'PN', 'PM']:
            statistics = statistics_dict_negative[domain]
            tmp = np.full((sentence_length, ), statistics[case][-1])
            for i in range(0, 4):
                tmp[i] = statistics[case][i]
            tmp = np.diag(tmp)
            domain_statistics_negative[domain] = np.matrix(tmp, dtype=np.float32)*domain_statistics_negative['union.I']
            domain_statistics_negative[domain] = chainer.dataset.to_device(device, domain_statistics_negative[domain])

        self.domain_statistics_positive = domain_statistics_positive
        self.domain_statistics_negative = domain_statistics_negative

    def __call__(self, xs, ys, zs):
        pred_ys = self.traverse(xs, zs)

        loss = .0
        for pred_y, y in zip(pred_ys, ys):
            _loss = F.softmax_cross_entropy(pred_y, y)
            loss += _loss
        reporter.report({'loss': loss}, self)

        accuracy = .0
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=0)[1] for pred_y in pred_ys]
        ys = [y.argmax(axis=0) for y in ys]
        for pred_y, y in zip(pred_ys, ys):
            if y == pred_y:
                accuracy += 1/len(ys)
        reporter.report({'accuracy': accuracy}, self)
        return loss

    def traverse(self, xs, zs):
        xs = [Variable(x) for x in xs]
        hx, cx = None, None
        hx, cx, ys = self.nstep_bilstm(xs=xs, hx=hx, cx=cx)
        ys = [ self.l1(y) for y in ys]
        ipdb.set_trace()
        ys = [F.matmul(self.domain_statistics_positive[z][:y.shape[0], :y.shape[0]], y) for y, z in zip(ys, zs)]
        return ys
