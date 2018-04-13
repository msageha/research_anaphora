import argparse
import pickle
import math
import json

import numpy as np
import chainer
import chainer.functions as F
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer import Variable
from chainer import cuda

from model import BiLSTMBase
from train import load_dataset
import os

domain_dict = {'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def load_model_path(path, part_flag=False):
    for domain in list(domain_dict) + ['union']:
        for epoch in range(20, 0, -1):
            model_path = '{0}/domain-{1}_epoch-{2}.npz'.format(path, domain, epoch)
            if os.path.exists(model_path):
                yield model_path
                break
    if part_flag:
        for part in range(10000, 190001, 30000):
            for epoch in range(20, 0, -1):
                model_path = '{0}/domain-union_pert_{1}_epoch-{2}.npz'.format(path, part, epoch)
                if os.path.exists(model_path):
                    yield model_path
                    break


def predict(model_path, test_data, domain, args):

    feature_size = test_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout)
    serializers.load_npz(model_path, model)
    accuracy_ga = .0
    accuracy_o = .0
    accuracy_ni = .0
    accuracy_all = .0


    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    for xs, ys in test_data:
        xs = cuda.cupy.array(xs, dtype=cuda.cupy.float32)
        pred_ys = model.traverse([xs])
        pred_ys = [pred_y.T for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=1) for pred_y in pred_ys]
        ys = ys.argmax(axis=0)
        if pred_ys[0] == ys[0]:
            accuracy_ga += 1
            accuracy_all += 1
        if pred_ys[1] == ys[1]:
            accuracy_o += 1
            accuracy_all += 1
        if pred_ys[2] == ys[2]:
            accuracy_ni += 1
            accuracy_all += 1
    accuracy_ga /= len(test_data)
    accuracy_o /= len(test_data)
    accuracy_ni /= len(test_data)
    accuracy_all /= 3*len(test_data)
    dump_path = '{0}/domain-{1}.tsv'.format(args.out, domain)
    print('model_path:{0}_domain:{1}_accuracy_all:{2:.3f}'.format(model_path, domain, accuracy_all*100))
    if not os.path.exists(dump_path):
        with open(dump_path, 'w') as f:
            f.write('model_path\tdomain\taccuracy_ga\taccuracy_o\taccuracy_ni\taccuracy_all\ttest_data_size\n')
    with open(dump_path, 'a') as f:
        f.write('{0}\t{1}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t{5:.3f}\t{6}\n'.format(model_path, domain, accuracy_ga*100, accuracy_o*100, accuracy_ni*100, accuracy_all*100, len(test_data)))

def main(train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', default='predict', help='Directory to output the result')
    parser.add_argument('--model_dir', '-m', type=str, default='')
    parser.add_argument('--part_flag', action='store_true')
    args = parser.parse_args()

    dataset_dict = load_dataset()
    print('start data load domain-all')
    all_test_x = []
    all_test_y = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        all_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        all_test_y += dataset_dict['{0}_y'.format(domain)][size:]

    for model in load_model_path(args.model_dir, args.part_flag):
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_y)
        predict(model, test_data, 'union', args)
        for domain in domain_dict:
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict(model, test_data, domain, args)


if __name__ == '__main__':
    main()