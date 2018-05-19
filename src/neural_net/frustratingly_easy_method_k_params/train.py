import argparse
import pickle
import time
import math
import json
import os
import random
from collections import OrderedDict
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import chainer
from chainer import cuda, Variable
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.training import extensions

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
        z_dataset = []
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
            x_dataset.append(x)
            y_ga_dataset.append(y_ga)
            y_ga_dep_tag_dataset.append(y_ga_dep_tag)
            y_o_dataset.append(y_o)
            y_o_dep_tag_dataset.append(y_o_dep_tag)
            y_ni_dataset.append(y_ni)
            y_ni_dep_tag_dataset.append(y_ni_dep_tag)
            z_dataset.append(domain)
            word_dataset.append(word)
            is_verb_dataset.append(is_verb)
        dataset_dict['{0}_x'.format(domain)] = x_dataset
        dataset_dict['{0}_y_ga'.format(domain)] = y_ga_dataset
        dataset_dict['{0}_y_o'.format(domain)] = y_o_dataset
        dataset_dict['{0}_y_ni'.format(domain)] = y_ni_dataset
        dataset_dict['{0}_y_ga_dep_tag'.format(domain)] = y_ga_dep_tag_dataset
        dataset_dict['{0}_y_o_dep_tag'.format(domain)] = y_o_dep_tag_dataset
        dataset_dict['{0}_y_ni_dep_tag'.format(domain)] = y_ni_dep_tag_dataset
        dataset_dict['{0}_z'.format(domain)] = z_dataset
        dataset_dict['{0}_word'.format(domain)] = word_dataset
        dataset_dict['{0}_is_verb'.format(domain)] = is_verb_dataset
    return dataset_dict

def training(train_dataset_dict, test_dataset_dict, domain, case, dump_path, args):
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
    
    train_data_size = sum([len(train_dataset_dict['{0}_x'.format(domain)]) for domain in domain_dict])
    test_data_size = sum([len(test_dataset_dict['{0}_x'.format(domain)]) for domain in domain_dict])

    with open('{0}/args/domain-{1}_case-{2}.json'.format(dump_path, domain, case), 'w') as f:
        args.__dict__['train_size'] = train_data_size
        args.__dict__['test_size'] = test_data_size
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))

    feature_size = train_dataset_dict['OC_x'][0][0].shape[0]

    model = BiLSTMBase(input_size=feature_size, output_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout, case=case, device=args.gpu)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    print('epoch\tmain/loss\tmain/accuracy\tvalidation/main/loss\tvalidation/main/accuracy\telapsed_time')
    st = time.time()
    max_accuracy = 0
    logs = []
    for epoch in range(1, args.epoch+1):
        training_data = []
        for domain in domain_dict:
            N = len(train_dataset_dict['{0}_x'.format(domain)])
            perm = np.random.permutation(N)
            for i in range(0, N, args.batchsize):
                batch_x = train_dataset_dict['{0}_x'.format(domain)][perm[i:i+args.batchsize]]
                batch_y = train_dataset_dict['{0}_y_{1}'.format(domain, case)][perm[i:i+args.batchsize]]
                batch_z = train_dataset_dict['{0}_z'.format(domain)][perm[i:i+args.batchsize]]
                training_data.append((batch_x, batch_y, batch_z))
        train_total_loss = 0
        train_total_accuracy = 0
        random.shuffle(training_data)
        for xs, ys, zs in training_data:
            xs = [cuda.to_gpu(x) for x in xs]
            xs = [Variable(x) for x in xs]
            ys = [cuda.to_gpu(y) for y in ys]
            loss, accuracy = model(xs=xs, ys=ys, zs=zs)
            model.zerograds()
            loss.backward()
            optimizer.update()
            train_total_loss += loss.data
            train_total_accuracy += accuracy
        train_total_loss /= len(training_data)
        train_total_accuracy /= len(training_data)
        test_data = []
        for domain in domain_dict:
            N = len(test_dataset_dict['{0}_x'.format(domain)])
            perm = np.random.permutation(N)
            for i in range(0, N, args.batchsize):
                batch_x = test_dataset_dict['{0}_x'.format(domain)][perm[i:i+args.batchsize]]
                batch_y = test_dataset_dict['{0}_y_{1}'.format(domain, case)][perm[i:i+args.batchsize]]
                batch_z = test_dataset_dict['{0}_z'.format(domain)][perm[i:i+args.batchsize]]
                test_data.append((batch_x, batch_y, batch_z))
        test_total_loss = 0
        test_total_accuracy = 0
        random.shuffle(test_data)
        for xs, ys, zs in test_data:
            xs = [cuda.to_gpu(x) for x in xs]
            xs = [Variable(x) for x in xs]
            ys = [cuda.to_gpu(y) for y in ys]
            loss, accuracy = model(xs=xs, ys=ys, zs=zs)
            test_total_loss += loss.data
            test_total_accuracy += accuracy
        test_total_loss /= len(test_data)
        test_total_accuracy /= len(test_data)
        if test_total_accuracy > max_accuracy:
            model.to_cpu()
            chainer.serializers.save_npz("{0}/model/domain-{1}_case-{2}_epoch-{3}.npz".format(dump_path, 'union', case, epoch), model)
            model.to_gpu()
        max_accuracy = max(max_accuracy, test_total_accuracy)
        logs.append({
            "main/loss": float(train_total_loss),
            "main/accuracy": float(train_total_accuracy),
            "validation/main/loss": float(test_total_loss),
            "validation/main/accuracy": float(test_total_accuracy),
            "epoch": epoch,
            "elapsed_time": float(time.time() - st)
        })
        print('{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(epoch, train_total_loss, train_total_accuracy, test_total_loss, test_total_accuracy, time.time() - st))
    with open('{0}/log/domain-{1}_case-{2}.json'.format(dump_path, domain, case), 'w') as f:
        json.dump(logs, f, indent=4)

def union(dataset_dict, args, dump_path):
    print('start data load domain-union')
    train_dataset_dict = {}
    test_dataset_dict = {}
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
        train_dataset_dict['{0}_x'.format(domain)] = np.array(dataset_dict['{0}_x'.format(domain)][:size])
        test_dataset_dict['{0}_x'.format(domain)] = np.array(dataset_dict['{0}_x'.format(domain)][size:])
        # train_dataset_dict['{0}_y_ga'.format(domain)] = np.array(dataset_dict['{0}_y_ga'.format(domain)][:size])
        # test_dataset_dict['{0}_y_ga'.format(domain)] = np.array(dataset_dict['{0}_y_ga'.format(domain)][size:])
        train_dataset_dict['{0}_y_o'.format(domain)] = np.array(dataset_dict['{0}_y_o'.format(domain)][:size])
        test_dataset_dict['{0}_y_o'.format(domain)] = np.array(dataset_dict['{0}_y_o'.format(domain)][size:])
        train_dataset_dict['{0}_y_ni'.format(domain)] = np.array(dataset_dict['{0}_y_ni'.format(domain)][:size])
        test_dataset_dict['{0}_y_ni'.format(domain)] = np.array(dataset_dict['{0}_y_ni'.format(domain)][size:])
        train_dataset_dict['{0}_z'.format(domain)] = np.array(dataset_dict['{0}_z'.format(domain)][:size])
        test_dataset_dict['{0}_z'.format(domain)] = np.array(dataset_dict['{0}_z'.format(domain)][size:])
    # training(train_dataset_dict, test_dataset_dict, 'union', 'ga', dump_path, args)
    training(train_dataset_dict, test_dataset_dict, 'union', 'o', dump_path, args)
    training(train_dataset_dict, test_dataset_dict, 'union', 'ni', dump_path, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.2)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=15)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--df_path', default='../dataframe')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    parser.add_argument('--seed', default=1)
    args = parser.parse_args()

    dataset_dict = load_dataset(args.df_path)
    union(dataset_dict, args, 'normal/dropout-{0}_batchsize-{1}'.format(args.dropout, args.batchsize))


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