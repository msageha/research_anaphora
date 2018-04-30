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
from train import set_random_seed
from train import training

domain_dict = OrderedDict([('OC', 'Yahoo!知恵袋'), ('OY', 'Yahoo!ブログ'), ('OW', '白書'), ('PB', '書籍'), ('PM', '雑誌'), ('PN', '新聞')])

def load_dataset(df_path):
    dataset_dict = {}
    domain_index = 0
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        with open('{0}/dataframe_list_{1}.pickle'.format(df_path, domain), 'rb') as f:
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
            domain_vec = np.zeros((x.shape[0], 6))
            domain_vec[:, domain_index] += 1
            x = np.hstack((x, domain_vec))
        dataset_dict['{0}_x'.format(domain)] = x_dataset
        dataset_dict['{0}_y_ga'.format(domain)] = y_ga_dataset
        dataset_dict['{0}_y'.format(domain)] = y_dataset
        dataset_dict['{0}_y_o'.format(domain)] = y_o_dataset
        dataset_dict['{0}_y_ni'.format(domain)] = y_ni_dataset

        domain_index += 1
    return dataset_dict

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
        union_train_o += dataset_dict['{0}_y_o'.format(domain)][:size]
        union_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        union_train_ni += dataset_dict['{0}_y_ni'.format(domain)][:size]
        union_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
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
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--df_path', default='../dataframe')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    parser.add_argument('--seed', default=1)
    args = parser.parse_args()

    dataset_dict = load_dataset(args.df_path)
    union(dataset_dict, args, 'normal/dropout-{0}_batchsize-{1}'.format(args.dropout, args.batchsize))

if __name__ == '__main__':
    main()