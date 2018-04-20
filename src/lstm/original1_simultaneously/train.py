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

from model import BiLSTMBase
from model import convert_seq

domain_dict = {'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def load_dataset(is_short):
    dataset_dict = {}
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        if is_short:
            with open('../original/dataframe_short/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
                df_list = pickle.load(f)
        else:
            with open('../original/dataframe_long/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
                df_list = pickle.load(f)
        x_dataset = []
        y_dataset = []
        for df in df_list:
            y_ga = np.array(df['ga_case'], dtype=np.int32)
            y_o = np.array(df['o_case'], dtype=np.int32)
            y_ni = np.array(df['ni_case'], dtype=np.int32)
            y = np.vstack((y_ga, y_o, y_ni)).T
            df = df.drop('ga_case', axis=1).drop('o_case', axis=1).drop('ni_case', axis=1).drop('ga_dep_tag', axis=1).drop('o_dep_tag', axis=1).drop('ni_dep_tag', axis=1)
            x = np.array(df, dtype=np.float32)
            x_dataset.append(x)
            y_dataset.append(y)
        dataset_dict['{0}_x'.format(domain)] = x_dataset
        dataset_dict['{0}_y'.format(domain)] = y_dataset
    return dataset_dict

def training(train_data, test_data, domain, dump_path, args):
    print('training start domain-{0}'.format(domain,))

    output_path = args.out
    if args.is_short:
        output_path += '_short'
    else:
        output_path += '_long'
    if not os.path.exists('{0}/{1}'.format(output_path, dump_path)):
        os.mkdir('{0}/{1}'.format(output_path, dump_path))
    output_path += '/' + dump_path
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

    model = BiLSTMBase(input_size=feature_size, n_labels=3, n_layers=args.n_layers, dropout=args.dropout)

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
    trigger = chainer.training.triggers.MaxValueTrigger(key='validation/main/accuracy_all', trigger=(1, 'epoch'))

    trainer.extend(evaluator, trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='log/domain-{0}.log'.format(domain)), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy_ga', 'main/accuracy_o', 'main/accuracy_ni', 'main/accuracy_all',
    'validation/main/loss', 'validation/main/accuracy_ga', 'validation/main/accuracy_o', 'validation/main/accuracy_ni', 'validation/main/accuracy_all', 'elapsed_time']), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, savefun=serializers.save_npz ,filename='model/domain-{0}_epoch-{{.updater.epoch}}.npz'.format(domain)), trigger=trigger)

    trainer.run()

def main(train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=2)
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', default='normal', help='Directory to output the result')
    parser.add_argument('--is_short', action='store_true')
    args = parser.parse_args()

    today = str(datetime.datetime.today())[:-16]
    dataset_dict = load_dataset(args.is_short)
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        train_x = dataset_dict['{0}_x'.format(domain)][:size]
        test_x = dataset_dict['{0}_x'.format(domain)][size:]
        train_y = dataset_dict['{0}_y'.format(domain)][:size]
        test_y = dataset_dict['{0}_y'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, today, args)
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
    training(train_data, test_data, 'union', today, args)

    union_train_x = np.array(union_train_x)
    union_train_y = np.array(union_train_y)
    for N in range(10000, len(union_train_x), 30000):
        perm = np.random.permutation(N)
        train_data = tuple_dataset.TupleDataset(union_train_x[perm], union_train_y[perm])
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_y)
        training(train_data, test_data, 'union_pert_{0}'.format(N), today, args)

def out_domain(train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=2)
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', default='out_domain', help='Directory to output the result')
    parser.add_argument('--is_short', action='store_true')
    args = parser.parse_args()

    today = str(datetime.datetime.today())[:-16]
    dataset_dict = load_dataset(args.is_short)
    print('start data load out_domain')
    for out_domain in domain_dict:
        union_train_x = []
        union_test_x = []
        union_train_y = []
        union_test_y = []
        for domain in domain_dict:
            if out_domain == domain:
                continue
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
            union_train_x += dataset_dict['{0}_x'.format(domain)][:size]
            union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
            union_train_y += dataset_dict['{0}_y'.format(domain)][:size]
            union_test_y += dataset_dict['{0}_y'.format(domain)][size:]
        print('out domain {0}\tdata_size {1}'.format(out_domain, len(union_train_x)))
        train_data = tuple_dataset.TupleDataset(union_train_x, union_train_y)
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_y)
        training(train_data, test_data, 'out-{0}'.format(out_domain), today, args)

if __name__ == '__main__':
    '''
    パラメータ
    train_test_ratio
    weight_decay
    dropout
    batchsize
    epoch
    optimizer(adam)
    '''
    out_domain()