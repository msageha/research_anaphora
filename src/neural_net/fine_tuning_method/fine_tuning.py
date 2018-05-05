import argparse
import pickle
import math
import json
import os
import datetime
import random
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')

import numpy as np
import chainer
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.training import extensions

import sys
sys.path.append('../baseline')
from model import BiLSTMBase
from model import convert_seq
from train import load_dataset
from train import set_random_seed

domain_dict = OrderedDict([('OC', 'Yahoo!知恵袋'), ('OY', 'Yahoo!ブログ'), ('OW', '白書'), ('PB', '書籍'), ('PM', '雑誌'), ('PN', '新聞')])

def load_union_model_path(path, case):
    for epoch in range(20, 0, -1):
        model_path = '{0}/model/domain-union_case-{1}_epoch-{2}.npz'.format(path, case, epoch)
        if os.path.exists(model_path):
            return model_path

def fine_tuning(model_path, train_data, test_data, domain, case, args):
    with open('{0}/args/domain-union_case-{1}.json'.format(args.dir, case)) as f:
        tmp = json.load(f)
    for key in tmp.keys():
        args.__dict__[key] = tmp[key]
    args.__dict__['train_size'] = len(train_data)
    args.__dict__['test_size'] = len(test_data)
    args.__dict__['epoch'] = 15
    print('fine_tuning start domain-{0}, case-{1}'.format(domain, case))

    # output_path = 'fine_tuning/dropout-{0}_batchsize-{1}'.format(args.dropout, args.batchsize)
    # if not os.path.exists('./{0}'.format(output_path)):
    #     os.mkdir('./{0}'.format(output_path))
    output_path = 'fine_tuning'
    dump_path = 'alpha-{0}_beta1-{1}_weightdecay-{2}'.format(args.alpha, args.beta1, args.weightdecay)
    if not os.path.exists('./{0}/{1}'.format(output_path, dump_path)):
        os.mkdir('./{0}/{1}'.format(output_path, dump_path))
    output_path += '/' + dump_path
    if not os.path.exists('{0}/{1}'.format(output_path, 'args')):
        os.mkdir('{0}/{1}'.format(output_path, 'args'))
        os.mkdir('{0}/{1}'.format(output_path, 'log'))
        os.mkdir('{0}/{1}'.format(output_path, 'model'))
        os.mkdir('{0}/{1}'.format(output_path, 'tmpmodel'))
        os.mkdir('{0}/{1}'.format(output_path, 'graph'))
    
    print(json.dumps(args.__dict__, indent=2))
    with open('{0}/args/domain-{1}_case-{2}.json'.format(output_path, domain, case), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    feature_size = train_data[0][0].shape[1]
    model = BiLSTMBase(input_size=feature_size, output_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout) #saveした時と同じ構成にすること．
    serializers.load_npz(model_path, model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    #optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.alpha, beta1=args.beta1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weightdecay))

    if args.disable_update_lstm:
        model.nstep_bilstm.disable_update()

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=convert_seq)
    trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out=output_path)

    evaluator = chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu, converter=convert_seq)
    trigger = chainer.training.triggers.MaxValueTrigger(key='validation/main/accuracy', trigger=(1, 'epoch'))

    trainer.extend(evaluator, trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='log/domain-{0}_case-{1}.log'.format(domain, case)), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(model, savefun=serializers.save_npz ,filename='model/domain-{0}_case-{1}_epoch-{{.updater.epoch}}.npz'.format(domain, case)), trigger=trigger)
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], file_name='./graph/accuracy_domain-{0}_case-{1}.png'.format(domain, case), x_key='epoch'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], file_name='./graph/loss_domain-{0}_case-{1}.png'.format(domain, case), x_key='epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--weightdecay', '-w', type=float, default=1e-4)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--df_path', default='../dataframe')
    parser.add_argument('--disable_update_lstm', action='store_true')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    args = parser.parse_args()
    dataset_dict = load_dataset(args.df_path)

    for case in ['ga', 'o', 'ni']:
        model_path = load_union_model_path(args.dir, case)
        for domain in domain_dict:
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
            train_x = dataset_dict['{0}_x'.format(domain)][:size]
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            train_y = dataset_dict['{0}_y_{1}'.format(domain, case)][:size]
            test_y = dataset_dict['{0}_y_{1}'.format(domain, case)][size:]
            train_data = tuple_dataset.TupleDataset(train_x, train_y)
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            fine_tuning(model_path, train_data, test_data, domain, case, args)

def params_search():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--weightdecay', '-w', type=float, default=1e-4)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--df_path', default='../dataframe')
    parser.add_argument('--disable_update_lstm', action='store_true')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    args = parser.parse_args()
    dataset_dict = load_dataset(args.df_path)

    case = 'ga'
    model_path = load_union_model_path(args.dir, case)
    for alpha in [0.01, 0.005, 0.001, 0.0005]:
        for beta1 in [0.8, 0.85, 0.9, 0.95]:
            for weightdecay in [0.001, 0.0005, 0.0001, 0.00005]:
                args.alpha = alpha
                args.beta1 = beta1
                args.weightdecay = weightdecay
                for domain in domain_dict:
                    size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
                    train_x = dataset_dict['{0}_x'.format(domain)][:size]
                    test_x = dataset_dict['{0}_x'.format(domain)][size:]
                    train_y = dataset_dict['{0}_y_{1}'.format(domain, case)][:size]
                    test_y = dataset_dict['{0}_y_{1}'.format(domain, case)][size:]
                    train_data = tuple_dataset.TupleDataset(train_x, train_y)
                    test_data  = tuple_dataset.TupleDataset(test_x, test_y)
                    fine_tuning(model_path, train_data, test_data, domain, case, args)

if __name__ == '__main__':
    params_search()