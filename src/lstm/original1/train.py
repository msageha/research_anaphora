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
            with open('./dataframe_short/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
                df_list = pickle.load(f)
        else:
            with open('./dataframe_long/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
                df_list = pickle.load(f)
        x_dataset = []
        y_dataset = []
        y_ga_dataset = []
        y_o_dataset = []
        y_ni_dataset = []
        for df in df_list:
            y_ga = np.array(df['ga_case'], dtype=np.int32)
            y_o = np.array(df['o_case'], dtype=np.int32)
            y_ni = np.array(df['ni_case'], dtype=np.int32)
            y = np.vstack((y_ga, y_o, y_ni)).T
            df = df.drop('ga_case', axis=1).drop('o_case', axis=1).drop('ni_case', axis=1).drop('ga_dep_tag', axis=1).drop('o_dep_tag', axis=1).drop('ni_dep_tag', axis=1)
            x = np.array(df, dtype=np.float32)
            x_dataset.append(x)
            y_dataset.append(y)
            y_ga_dataset.append(y_ga)
            y_o_dataset.append(y_o)
            y_ni_dataset.append(y_ni)
        dataset_dict['{0}_x'.format(domain)] = x_dataset
        dataset_dict['{0}_y_ga'.format(domain)] = y_ga_dataset
        dataset_dict['{0}_y'.format(domain)] = y_dataset
        dataset_dict['{0}_y_o'.format(domain)] = y_o_dataset
        dataset_dict['{0}_y_ni'.format(domain)] = y_ni_dataset
    return dataset_dict

def training(train_data, test_data, domain, case, dump_path, args):
    print('training start domain-{0}, case-{1}'.format(domain, case))
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
    with open('{0}/args/domain-{1}_case-{2}.json'.format(output_path, domain, case), 'w') as f:
        args.__dict__['train_size'] = len(train_data)
        args.__dict__['test_size'] = len(test_data)
        json.dump(args.__dict__, f, indent=2)

    feature_size = train_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)

    # updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=convert_seq)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out=output_path)

    # evaluator = chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu, converter=convert_seq)
    evaluator = chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu)
    trigger = chainer.training.triggers.MaxValueTrigger(key='validation/main/accuracy', trigger=(1, 'epoch'))

    trainer.extend(evaluator, trigger=(1, 'epoch'))
    # trainer.extend(extensions.dump_graph(out_name="./graph/domain-{0}_case-{1}.dot".format(domain, case)))
    trainer.extend(extensions.LogReport(log_name='log/domain-{0}_case-{1}.log'.format(domain, case)), trigger=(1, 'epoch'))
    # trainer.extend(extensions.snapshot(filename='snapshot/domain-{0}_case-{1}_epoch-{{.updater.epoch}}'.format(domain, case)), trigger=(1, 'epoch'))
    # trainer.extend(extensions.MicroAverage('main/correct', 'main/total', 'main/accuracy'))
    # trainer.extend(extensions.MicroAverage('validation/main/correct', 'validation/main/total', 'validation/main/accuracy'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, savefun=serializers.save_npz ,filename='model/domain-{0}_case-{1}_epoch-{{.updater.epoch}}.npz'.format(domain, case)), trigger=trigger)
    # trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy/domain-{0}_case-{1}.png'.format(domain, case)))
    # trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()

def main(train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
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
        train_y = dataset_dict['{0}_y_ga'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ga', today, args)
        train_y = dataset_dict['{0}_y_o'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'o', today, args)
        train_y = dataset_dict['{0}_y_ni'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ni', today, args)
    print('start data load domain-union')
    union_train_x = []
    union_test_x = []
    union_train_ga = []
    union_test_ga = []
    union_train_o = []
    union_test_o = []
    union_train_ni = []
    union_test_ni = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        union_train_x += dataset_dict['{0}_x'.format(domain)][:size]
        union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        union_train_ga += dataset_dict['{0}_y_ga'.format(domain)][:size]
        union_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
        union_train_o += dataset_dict['{0}_y_o'.format(domain)][:size]
        union_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        union_train_ni += dataset_dict['{0}_y_ni'.format(domain)][:size]
        union_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ga)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ga)
    training(train_data, test_data, 'union', 'ga', today, args)
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_o)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_o)
    training(train_data, test_data, 'union', 'o', today, args)
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ni)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ni)
    training(train_data, test_data, 'union', 'ni', today, args)
    union_train_x = np.array(union_train_x)
    union_train_ga = np.array(union_train_ga)
    union_train_o = np.array(union_train_o)
    union_train_ni = np.array(union_train_ni)
    for N in range(10000, len(union_train_x), 30000):
        perm = np.random.permutation(N)
        train_data = tuple_dataset.TupleDataset(union_train_x[perm], union_train_ga[perm])
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ga)
        training(train_data, test_data, 'union_pert_{0}'.format(N), 'ga', today, args)
        train_data = tuple_dataset.TupleDataset(union_train_x[perm], union_train_o[perm])
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_o)
        training(train_data, test_data, 'union_pert_{0}'.format(N), 'o', today, args)
        train_data = tuple_dataset.TupleDataset(union_train_x[perm], union_train_ni[perm])
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ni)
        training(train_data, test_data, 'union_pert_{0}'.format(N), 'ni', today, args)

def out_domain(train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
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
        union_train_ga = []
        union_test_ga = []
        union_train_o = []
        union_test_o = []
        union_train_ni = []
        union_test_ni = []
        for domain in domain_dict:
            if out_domain == domain:
                continue
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
            union_train_x += dataset_dict['{0}_x'.format(domain)][:size]
            union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
            union_train_ga += dataset_dict['{0}_y_ga'.format(domain)][:size]
            union_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
            union_train_o += dataset_dict['{0}_y_o'.format(domain)][:size]
            union_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
            union_train_ni += dataset_dict['{0}_y_ni'.format(domain)][:size]
            union_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
        print('out domain {0}\tdata_size {1}'.format(out_domain, len(union_train_x)))
        train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ga)
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ga)
        training(train_data, test_data, 'out-{0}'.format(out_domain), 'ga', today, args)
        train_data = tuple_dataset.TupleDataset(union_train_x, union_train_o)
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_o)
        training(train_data, test_data, 'out-{0}'.format(out_domain), 'o', today, args)
        train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ni)
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ni)
        training(train_data, test_data, 'out-{0}'.format(out_domain), 'ni', today, args)

def arrange_training(arrange_size=14672, train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
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
        train_x = dataset_dict['{0}_x'.format(domain)][:arrange_size]
        test_x = dataset_dict['{0}_x'.format(domain)][size:]
        train_y = dataset_dict['{0}_y_ga'.format(domain)][:arrange_size]
        test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ga', today, args)
        train_y = dataset_dict['{0}_y_o'.format(domain)][:arrange_size]
        test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'o', today, args)
        train_y = dataset_dict['{0}_y_ni'.format(domain)][:arrange_size]
        test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ni', today, args)
    print('start data load domain-union')
    union_train_x = []
    union_test_x = []
    union_train_ga = []
    union_test_ga = []
    union_train_o = []
    union_test_o = []
    union_train_ni = []
    union_test_ni = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        union_train_x += dataset_dict['{0}_x'.format(domain)][:arrange_size]
        union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        union_train_ga += dataset_dict['{0}_y_ga'.format(domain)][:arrange_size]
        union_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
        union_train_o += dataset_dict['{0}_y_o'.format(domain)][:arrange_size]
        union_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        union_train_ni += dataset_dict['{0}_y_ni'.format(domain)][:arrange_size]
        union_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ga)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ga)
    training(train_data, test_data, 'union', 'ga', today, args)
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_o)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_o)
    training(train_data, test_data, 'union', 'o', today, args)
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ni)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ni)
    training(train_data, test_data, 'union', 'ni', today, args)

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
    arrange_training()