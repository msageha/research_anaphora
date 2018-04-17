import argparse
import pickle
import math
import json
import datetime
import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import chainer
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.training import extensions

from model import BiGRU
from model import convert_seq

domain_dict = {'OC':'Yahoo!知恵袋'}#, 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def load_dataset():
    dataset_dict = {}
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        with open('./dataframe/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
            df_list = pickle.load(f)
        x_dataset = []
        y_dataset = []
        for df in df_list:
            y_ga = np.array(df['ga_case'], dtype=np.int32)
            y_o = np.array(df['o_case'], dtype=np.int32)
            y_ni = np.array(df['ni_case'], dtype=np.int32)
            y_pred = np.array(df['pred'], dtype=np.int32)
            y_none = np.zeros(y_ga.shape[0], dtype=np.int32)
            for i, data in enumerate(zip(y_ga, y_o, y_ni, y_pred)):
                if max(data) == 0:
                    y_none[i] = 1
            y = np.vstack((y_none, y_pred, y_ga, y_o, y_ni)).T
            df = df.drop('ga_case', axis=1).drop('o_case', axis=1).drop('ni_case', axis=1).drop('word', axis=1).drop('pred_prev', axis=1).drop('pred_next', axis=1).drop('file_path', axis=1)
            x = np.array(df, dtype=np.float32)
            x_dataset.append(x)
            y_dataset.append(y)
        dataset_dict['{0}_x'.format(domain)] = x_dataset
        dataset_dict['{0}_y'.format(domain)] = y_dataset
    return dataset_dict


def training(train_data, test_data, domain, dump_path):
    print('training start domain-{0}'.format(domain,))
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=2)
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', default='normal', help='Directory to output the result')
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    if not os.path.exists('{0}/{1}'.format(args.out, dump_path)):
        os.mkdir('{0}/{1}'.format(args.out, dump_path))
    output_path = args.out + '/' + dump_path
    if args.short:
        output_path = args.out + '_short/' + dump_path
    if not os.path.exists('{0}/{1}'.format(output_path, 'args')):
        os.mkdir('{0}/{1}'.format(output_path, 'args'))
        os.mkdir('{0}/{1}'.format(output_path, 'log'))
        os.mkdir('{0}/{1}'.format(output_path, 'model'))
        os.mkdir('{0}/{1}'.format(output_path, 'tmpmodel'))

    print(json.dumps(args.__dict__, indent=2))
    with open('{0}/args/domain-{1}.json'.format(output_path, domain), 'w') as f:
        args.__dict__['train_size'] = len(train_data)
        args.__dict__['test_size'] = len(test_data)
        json.dump(args.__dict__, f, indent=2)

    feature_size = train_data[0][0].shape[1]

    model = BiGRU(input_size=feature_size, n_labels=5, n_layers=args.n_layers, dropout=args.dropout)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=convert_seq)
    trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out=output_path)

    evaluator = chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu, converter=convert_seq)
    trigger = chainer.training.triggers.MaxValueTrigger(key='validation/main/f1', trigger=(1, 'epoch'))

    trainer.extend(evaluator, trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='log/domain-{0}.log'.format(domain)), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/precision', 'main/recall', 'main/f1',
    'validation/main/loss', 'validation/main/precision', 'validation/main/recall', 'validation/main/f1', 'elapsed_time']), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, savefun=serializers.save_npz ,filename='model/domain-{0}_epoch-{{.updater.epoch}}.npz'.format(domain)), trigger=trigger)

    trainer.run()

def main(train_test_ratio=0.8):
    today = str(datetime.datetime.today())[:-16]
    dataset_dict = load_dataset()
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        train_x = dataset_dict['{0}_x'.format(domain)][:size]
        test_x = dataset_dict['{0}_x'.format(domain)][size:]
        train_y = dataset_dict['{0}_y'.format(domain)][:size]
        test_y = dataset_dict['{0}_y'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, today)
    print('start data load domain-union')
    union_train_x = []
    union_test_x = []
    union_train_y = []
    union_test_y = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        union_train_x += dataset_dict['{0}_x'.format(domain)][:size]
        union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        union_train_y += dataset_dict['{0}_y'.format(domain)][:size]
        union_test_y += dataset_dict['{0}_y'.format(domain)][size:]
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_y)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_y)
    training(train_data, test_data, 'union', today)

    union_train_x = np.array(union_train_x)
    union_train_y = np.array(union_train_y)
    for N in range(10000, len(union_train_x), 30000):
        perm = np.random.permutation(N)
        train_data = tuple_dataset.TupleDataset(union_train_x[perm], union_train_y[perm])
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_y)
        training(train_data, test_data, 'union_pert_{0}'.format(N), today)

if __name__=='__main__':
    main()