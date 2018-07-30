import argparse
import pickle
import math
import json
import os
import time
import datetime
import random
from collections import OrderedDict
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

import numpy as np
import chainer
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.training import extensions


import chainer
import chainer.functions as F
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer import cuda, Variable

import chainer
import chainer.functions as F
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer import cuda

from model import MajorityNetwork

import sys
sys.path.append('../')
from baseline.model import BiLSTMBase as Fine_BiLSTMBase
from baseline.model import convert_seq as Fine_convert_seq
from frustratingly_easy_method_k_params.model import BiLSTMBase as Frust_BiLSTMBase
from frustratingly_easy_method_k_params.model import convert_seq as Frust_convert_seq
from statistics_method.model import BiLSTMBase as Statistic_BiLSTMBase
from statistics_method.model import convert_seq as Statistic_convert_seq

import ipdb

domain_dict = OrderedDict([('OC', 'Yahoo!知恵袋'), ('OY', 'Yahoo!ブログ'), ('OW', '白書'), ('PB', '書籍'), ('PM', '雑誌'), ('PN', '新聞')])

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

def load_fine_model_path(path, case, domain):
    for epoch in range(20, 0, -1):
        model_path = '{0}/model/domain-{1}_case-{2}_epoch-{3}.npz'.format(path, domain, case, epoch)
        if os.path.exists(model_path):
            return model_path

def load_statistics_model_path(path, case, domain='union'):
    for epoch in range(20, 0, -1):
        model_path = '{0}/model/domain-{1}_case-{2}_epoch-{3}.npz'.format(path, domain, case, epoch)
        if os.path.exists(model_path):
            return model_path

def load_frust_model_path(path, case, domain='union'):
    for epoch in range(20, 0, -1):
        model_path = '{0}/model/domain-{1}_case-{2}_epoch-{3}.npz'.format(path, domain, case, epoch)
        if os.path.exists(model_path):
            return model_path

def set_random_seed(seed):
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)
    # set Chainer(CuPy) random seed
    cuda.cupy.random.seed(seed)

def calculate_type_statistics(dataset_dict, case, out_domain=''):
    type_statistics_dict = {'union':[0., 0., 0., 0., 0.,]}
    union_y_length = 0
    for domain in domain_dict:
        if domain == out_domain:
            continue
        type_statistics = [0., 0., 0., 0., 0.,]
        y = dataset_dict['{0}_y_{1}'.format(domain, case)]
        for _y in y:
            index = _y.argmax() if _y.argmax() < 4 else 4
            type_statistics[index] += 1
        type_statistics_dict[domain] = type_statistics
        for i in range(5):
            type_statistics_dict['union'][i] += type_statistics_dict[domain][i]
            type_statistics_dict[domain][i] = type_statistics_dict[domain][i]/len(y)*100
        union_y_length += len(y)
    for i in range(5):
        type_statistics_dict['union'][i] = type_statistics_dict['union'][i]/union_y_length*100
    return type_statistics_dict

def training(frust_model_path, statistics_model_path, fine_model_path, train_data, test_data, type_statistics_dict, domain, case, dump_path, args):
    set_random_seed(args.seed)

    if not os.path.exists('{0}'.format(dump_path)):
        os.mkdir('{0}'.format(dump_path))
    if not os.path.exists('{0}/{1}'.format(dump_path, 'args')):
        os.mkdir('{0}/{1}'.format(dump_path, 'args'))
        os.mkdir('{0}/{1}'.format(dump_path, 'log'))
        os.mkdir('{0}/{1}'.format(dump_path, 'model'))
        os.mkdir('{0}/{1}'.format(dump_path, 'tmpmodel'))
        os.mkdir('{0}/{1}'.format(dump_path, 'graph'))
    
    train_data_size = len(train_data)
    test_data_size = len(test_data)

    with open('{0}/args/domain-{1}_case-{2}.json'.format(dump_path, domain, case), 'w') as f:
        args.__dict__['train_size'] = train_data_size
        args.__dict__['test_size'] = test_data_size
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))

    feature_size = test_data[0][0].shape[1]

    statistics_model = Statistic_BiLSTMBase(input_size=feature_size, output_size=feature_size, n_labels=2, n_layers=1, dropout=0.2, type_statistics_dict=type_statistics_dict)
    frust_model = Frust_BiLSTMBase(input_size=feature_size, output_size=feature_size, n_labels=2, n_layers=1, dropout=0.2)
    fine_model = Fine_BiLSTMBase(input_size=feature_size, output_size=feature_size, n_labels=2, n_layers=1, dropout=0.2)
    model = MajorityNetwork(input_size=6, n_labels=2)

    serializers.load_npz(statistics_model_path, statistics_model)
    serializers.load_npz(frust_model_path, frust_model)
    serializers.load_npz(fine_model_path, fine_model)

    cuda.get_device(0).use()
    statistics_model.to_gpu()
    frust_model.to_gpu()
    fine_model.to_gpu()
    model.to_gpu()

    # statistics_model.disable_update()
    # frust_model.disable_update()
    # fine_model.disable_update()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    print('epoch\tmain/loss\tmain/accuracy\tvalidation/main/loss\tvalidation/main/accuracy\telapsed_time')
    st = time.time()
    max_accuracy = 0
    logs = []

    for epoch in range(1, 10+1):
        train_total_loss = 0
        train_total_accuracy = 0
        N = len(train_data)
        perm = np.random.permutation(N)

        for i in range(0, N, args.batchsize):
            batch_xs = train_data[0][perm[i:i+args.batchsize]]
            batch_ys = train_data[1][perm[i:i+args.batchsize]]
            batch_zs = train_data[2][perm[i:i+args.batchsize]]

            xs1 = [cuda.cupy.array(x, dtype=cuda.cupy.float32) for x in batch_xs]
            xs2 = [cuda.to_gpu(x) for x in batch_xs]
            xs2 = [Variable(x) for x in xs2]
            ys = [cuda.to_gpu(y) for y in batch_ys]
            zs = batch_zs
            with chainer.using_config('train', False):
                statistics_pred_ys = statistics_model.traverse(xs1, zs)
                fine_pred_ys = fine_model.traverse(xs1)
                frust_pred_ys = frust_model.traverse(xs2, zs)
            statistics_pred_ys = [F.softmax(statistics_pred_y) for statistics_pred_y in statistics_pred_ys]
            fine_pred_ys = [F.softmax(fine_pred_y) for fine_pred_y in fine_pred_ys]
            frust_pred_ys = [F.softmax(frust_pred_y) for frust_pred_y in frust_pred_ys]
            xs = [F.concat((fine_pred_y.data, frust_pred_y.data, statistics_pred_y.data)) for fine_pred_y, frust_pred_y, statistics_pred_y in zip(fine_pred_ys, frust_pred_ys, statistics_pred_ys)]
            loss, accuracy = model(xs=xs, ys=ys)
            loss.backward()
            optimizer.update()
            train_total_loss += loss.data
            train_total_accuracy += accuracy
        train_total_loss /= len(train_data)
        train_total_accuracy /= len(train_data)

        test_total_loss = 0
        test_total_accuracy = 0
        for xs, ys, zs in test_data:
            xs1 = cuda.cupy.array(xs, dtype=cuda.cupy.float32)
            xs2 = cuda.to_gpu(xs)
            xs2 = Variable(xs2)
            ys = cuda.to_gpu(ys)
            with chainer.using_config('train', False):
                statistics_pred_ys = statistics_model.traverse([xs1], [zs])
                fine_pred_ys = fine_model.traverse([xs1])
                frust_pred_ys = frust_model.traverse([xs2], [zs])
                statistics_pred_ys = [F.softmax(statistics_pred_y) for statistics_pred_y in statistics_pred_ys]
                fine_pred_ys = [F.softmax(fine_pred_y) for fine_pred_y in fine_pred_ys]
                frust_pred_ys = [F.softmax(frust_pred_y) for frust_pred_y in frust_pred_ys]
                xs = [F.concat((fine_pred_y.data, frust_pred_y.data, statistics_pred_y.data)) for fine_pred_y, frust_pred_y, statistics_pred_y in zip(fine_pred_ys, frust_pred_ys, statistics_pred_ys)]
                loss, accuracy = model(xs=xs, ys=[ys])

            test_total_loss += loss.data
            test_total_accuracy += accuracy
        test_total_loss /= len(train_data)
        test_total_accuracy /= len(train_data)

        if test_total_accuracy > max_accuracy:
            model.to_cpu()
            chainer.serializers.save_npz("{0}/model/domain-{1}_case-{2}_epoch-{3}.npz".format(dump_path, domain, case, epoch), model)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--df_path', default='../dataframe')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dataset_dict = load_dataset(args.df_path)
    train_dataset_dict = {}
    for domain in domain_dict:
        train_size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.7)
        train_dataset_dict['{0}_y_ga'.format(domain)] = dataset_dict['{0}_y_ga'.format(domain)][:train_size]
        train_dataset_dict['{0}_y_o'.format(domain)] = dataset_dict['{0}_y_o'.format(domain)][:train_size]
        train_dataset_dict['{0}_y_ni'.format(domain)] = dataset_dict['{0}_y_ni'.format(domain)][:train_size]

    for case in ['ga', 'o', 'ni']:
        type_statistics_dict = calculate_type_statistics(train_dataset_dict, case)
        frust_model_path = load_frust_model_path('../frustratingly_easy_method_k_params/normal/dropout-0.2_batchsize-32', case)
        statistics_model_path = load_statistics_model_path('../statistics_method/normal/dropout-0.2_batchsize-32', case)
        for domain in domain_dict:
            fine_model_path = load_fine_model_path('../fine_tuning_method/fine_tuning/alpha-0.001_beta1-0.9_weightdecay-0.0001', case, domain)

            train_size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.7)
            dev_size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
            train_x = dataset_dict['{0}_x'.format(domain)][:train_size]
            train_z = dataset_dict['{0}_z'.format(domain)][:train_size]
            test_x = dataset_dict['{0}_x'.format(domain)][train_size:dev_size]
            test_z = dataset_dict['{0}_z'.format(domain)][train_size:dev_size]
            if case == 'ga':
                train_y = dataset_dict['{0}_y_ga'.format(domain)][:train_size]
                test_y = dataset_dict['{0}_y_ga'.format(domain)][train_size:dev_size]
            elif case == 'o':
                train_y = dataset_dict['{0}_y_o'.format(domain)][:train_size]
                test_y = dataset_dict['{0}_y_o'.format(domain)][train_size:dev_size]
            elif case == 'ni':
                train_y = dataset_dict['{0}_y_ni'.format(domain)][:train_size]
                test_y = dataset_dict['{0}_y_ni'.format(domain)][train_size:dev_size]

            train_data = np.array((train_x, train_y, train_z))
            test_data = chainer.datasets.TupleDataset(test_x, test_y, test_z)

            training(frust_model_path, statistics_model_path, fine_model_path, train_data, test_data, type_statistics_dict, domain, case, 'normal/dropout-{0}_batchsize-{1}'.format(args.dropout, args.batchsize), args)
    
if __name__ == '__main__':
    main()
