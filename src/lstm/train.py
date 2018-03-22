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

def load_dataset():
    dataset_dict = {}
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        with open('./dataframe/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
            df_list = pickle.load(f)
        x_dataset = []
        y_ga_dataset = []
        y_o_dataset = []
        y_ni_dataset = []
        for df in df_list:
            y_ga = np.array(df['ga_case'], dtype=np.int32)
            y_o = np.array(df['o_case'], dtype=np.int32)
            y_ni = np.array(df['ni_case'], dtype=np.int32)
            df = df.drop('ga_case', axis=1).drop('o_case', axis=1).drop('ni_case', axis=1).drop('ga_dep_tag', axis=1).drop('o_dep_tag', axis=1).drop('ni_dep_tag', axis=1)
            x = np.array(df, dtype=np.float32)
            x_dataset.append(x)
            y_ga_dataset.append(y_ga)
            y_o_dataset.append(y_o)
            y_ni_dataset.append(y_ni)
        dataset_dict['{0}_x'.format(domain)] = x_dataset
        dataset_dict['{0}_y_ga'.format(domain)] = y_ga_dataset
        dataset_dict['{0}_y_o'.format(domain)] = y_o_dataset
        dataset_dict['{0}_y_ni'.format(domain)] = y_ni_dataset
    return dataset_dict

def training(train_data, test_data, domain, case):
    print('training start domain-{0}, case-{1}'.format(domain, case))
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='normal', help='Directory to output the result')
    args = parser.parse_args()

    today = str(datetime.datetime.today())[:-7]
    os.mkdir('{0}/{1}'.format(args.out, today))
    output_path = args.out + '/' + today
    os.mkdir('{0}/{1}'.format(output_path, 'args'))
    os.mkdir('{0}/{1}'.format(output_path, 'log'))
    os.mkdir('{0}/{1}'.format(output_path, 'model'))

    print(json.dumps(args.__dict__, indent=2))
    with open('{0}/args/domain-{1}_case-{2}.json'.format(output_path, domain, case), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    feature_size = train_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

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
    dataset_dict = load_dataset()
    print('start data load domain-all')
    all_train_x = []
    all_test_x = []
    all_train_ga = []
    all_test_ga = []
    all_train_o = []
    all_test_o = []
    all_train_ni = []
    all_test_ni = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        all_train_x += dataset_dict['{0}_x'.format(domain)][:size]
        all_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        all_train_ga += dataset_dict['{0}_y_ga'.format(domain)][:size]
        all_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
        all_train_o += dataset_dict['{0}_y_o'.format(domain)][:size]
        all_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        all_train_ni += dataset_dict['{0}_y_ni'.format(domain)][:size]
        all_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
    train_data = tuple_dataset.TupleDataset(all_train_x, all_train_ga)
    test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ga)
    training(train_data, test_data, 'all', 'ga')
    train_data = tuple_dataset.TupleDataset(all_train_x, all_train_o)
    test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_o)
    training(train_data, test_data, 'all', 'o')
    train_data = tuple_dataset.TupleDataset(all_train_x, all_train_ni)
    test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ni)
    training(train_data, test_data, 'all', 'ni')
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        train_x = dataset_dict['{0}_x'.format(domain)][:size]
        test_x = dataset_dict['{0}_x'.format(domain)][size:]
        train_y = dataset_dict['{0}_y_ga'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ga')
        train_y = dataset_dict['{0}_y_o'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'o')
        train_y = dataset_dict['{0}_y_ni'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ni')

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
    main()