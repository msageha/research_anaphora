import argparse
import pickle
import math
import json
import os
import random
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import chainer
from chainer import cuda
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.training import extensions

from model import BiLSTMBase
from model import convert_seq
from train import load_dataset
from train import training

domain_dict = OrderedDict([('OC', 'Yahoo!知恵袋'), ('OY', 'Yahoo!ブログ'), ('OW', '白書'), ('PB', '書籍'), ('PM', '雑誌'), ('PN', '新聞')])

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
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
        union_train_x += dataset_dict['{0}_x'.format(domain)][:size]
        union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        union_train_ga += dataset_dict['{0}_y_ga'.format(domain)][:size]
        union_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
    train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ga)
    test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ga)
    training(train_data, test_data, 'union', 'ga', dump_path, args)

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
    for dropout in [0.1*i for i in range(4)]:
        args.dropout = dropout
        for batchsize in [2**i for i in range(5, 9)]:
            args.batchsize = batchsize
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