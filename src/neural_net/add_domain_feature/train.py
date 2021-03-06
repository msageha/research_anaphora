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
from chainer import cuda
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.training import extensions

import sys
sys.path.append('../baseline')
from model import BiLSTMBase
from model import convert_seq

domain_dict = OrderedDict([('OC', 'Yahoo!知恵袋'), ('OY', 'Yahoo!ブログ'), ('OW', '白書'), ('PB', '書籍'), ('PM', '雑誌'), ('PN', '新聞')])

def set_random_seed(seed):
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)
    # set Chainer(CuPy) random seed
    cuda.cupy.random.seed(seed)

def load_dataset(df_path):
    domain_index = 0
    dataset_dict = {}
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        with open('{0}/dataframe_list_{1}.pickle'.format(df_path, domain), 'rb') as f:
            df_list = pickle.load(f)
        x_dataset = []
        y_ga_dataset = []
        y_ga_dep_tag_dataset = []
        y_o_dataset = []
        y_o_dep_tag_dataset = []
        y_ni_dataset = []
        y_ni_dep_tag_dataset = []
        word_dataset = []
        is_verb_dataset = []
        for df in df_list:
            y_ga = np.array(df['ga_case'], dtype=np.int32)
            y_o = np.array(df['o_case'], dtype=np.int32)
            y_ni = np.array(df['ni_case'], dtype=np.int32)
            y_ga_dep_tag = np.array(df['ga_dep_tag'])
            y_o_dep_tag = np.array(df['o_dep_tag'])
            y_ni_dep_tag = np.array(df['ni_dep_tag'])
            word = np.array(df['word'])
            is_verb = np.array(df['is_verb']).argmax()
            for i in range(17):
                df = df.drop('feature:{}'.format(i), axis=1)
            df = df.drop('word', axis=1).drop('ga_case', axis=1).drop('o_case', axis=1).drop('ni_case', axis=1).drop('ga_dep_tag', axis=1).drop('o_dep_tag', axis=1).drop('ni_dep_tag', axis=1)
            x = np.array(df, dtype=np.float32)
            y_ga_dataset.append(y_ga)
            y_o_dataset.append(y_o)
            y_ni_dataset.append(y_ni)
            y_ga_dep_tag_dataset.append(y_ga_dep_tag)
            y_o_dep_tag_dataset.append(y_o_dep_tag)
            y_ni_dep_tag_dataset.append(y_ni_dep_tag)
            domain_vec = np.zeros((x.shape[0], 6))
            domain_vec[:, domain_index] += 1
            x = np.hstack((x, domain_vec))
            x = np.array(x, dtype=np.float32)
            x_dataset.append(x)
            word_dataset.append(word)
            is_verb_dataset.append(is_verb)
        dataset_dict['{0}_x'.format(domain)] = x_dataset
        dataset_dict['{0}_y_ga'.format(domain)] = y_ga_dataset
        dataset_dict['{0}_y_o'.format(domain)] = y_o_dataset
        dataset_dict['{0}_y_ni'.format(domain)] = y_ni_dataset
        dataset_dict['{0}_y_ga_dep_tag'.format(domain)] = y_ga_dep_tag_dataset
        dataset_dict['{0}_y_o_dep_tag'.format(domain)] = y_o_dep_tag_dataset
        dataset_dict['{0}_y_ni_dep_tag'.format(domain)] = y_ni_dep_tag_dataset
        dataset_dict['{0}_word'.format(domain)] = word_dataset
        dataset_dict['{0}_is_verb'.format(domain)] = is_verb_dataset
        domain_index += 1
    return dataset_dict

def training(train_data, test_data, domain, case, dump_path, args):
    print('training start domain-{0}, case-{1}'.format(domain, case))
    set_random_seed(args.seed)

    if not os.path.exists('{0}'.format(dump_path)):
        os.mkdir('{0}'.format(dump_path))
    if not os.path.exists('{0}/{1}'.format(dump_path, 'args')):
        os.mkdir('{0}/{1}'.format(dump_path, 'args'))
        os.mkdir('{0}/{1}'.format(dump_path, 'log'))
        os.mkdir('{0}/{1}'.format(dump_path, 'model'))
        os.mkdir('{0}/{1}'.format(dump_path, 'tmpmodel'))
        os.mkdir('{0}/{1}'.format(dump_path, 'graph'))

    with open('{0}/args/domain-{1}_case-{2}.json'.format(dump_path, domain, case), 'w') as f:
        args.__dict__['train_size'] = len(train_data)
        args.__dict__['test_size'] = len(test_data)
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))

    feature_size = train_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, output_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=convert_seq)
    trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out=dump_path)

    evaluator = chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu, converter=convert_seq)
    trigger = chainer.training.triggers.MaxValueTrigger(key='validation/main/accuracy', trigger=(1, 'epoch'))

    trainer.extend(evaluator, trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='log/domain-{0}_case-{1}.log'.format(domain, case)), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(model, savefun=serializers.save_npz ,filename='domain-{0}_case-{1}_epoch-{{.updater.epoch}}.npz'.format(domain, case)), trigger=trigger)
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], file_name='accuracy_domain-{0}_case-{1}.png'.format(domain, case), x_key='epoch'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], file_name='loss_domain-{0}_case-{1}.png'.format(domain, case), x_key='epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.run()

    os.system('mv {0}/*.npz {0}/model/'.format(dump_path))
    os.system('mv {0}/*.png {0}/graph/'.format(dump_path))

def union(dataset_dict, args, dump_path):
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
        train_size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.7)
        dev_size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
        union_train_x += dataset_dict['{0}_x'.format(domain)][:train_size]
        union_test_x += dataset_dict['{0}_x'.format(domain)][train_size:dev_size]
        union_train_ga += dataset_dict['{0}_y_ga'.format(domain)][:train_size]
        union_test_ga += dataset_dict['{0}_y_ga'.format(domain)][train_size:dev_size]
        union_train_o += dataset_dict['{0}_y_o'.format(domain)][:train_size]
        union_test_o += dataset_dict['{0}_y_o'.format(domain)][train_size:dev_size]
        union_train_ni += dataset_dict['{0}_y_ni'.format(domain)][:train_size]
        union_test_ni += dataset_dict['{0}_y_ni'.format(domain)][train_size:dev_size]
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ga)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ga)
    training(train_data, test_data, 'union', 'ga', dump_path, args)
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_o)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_o)
    training(train_data, test_data, 'union', 'o', dump_path, args)
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ni)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ni)
    training(train_data, test_data, 'union', 'ni', dump_path, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.2)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    parser.add_argument('--seed', default=1)
    parser.add_argument('--df_path', default='../dataframe')
    args = parser.parse_args()

    dataset_dict = load_dataset(args.df_path)
    union(dataset_dict, args, 'normal/dropout-{0}_batchsize-{1}'.format(args.dropout, args.batchsize))

if __name__ == '__main__':
    main()