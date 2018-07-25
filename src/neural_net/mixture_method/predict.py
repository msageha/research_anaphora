import argparse
import pickle
import time
import math
import json
import os
import random
from collections import OrderedDict, defaultdict
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
import chainer.functions as F

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

def load_model_path(path, case, domain):
    for epoch in range(20, 0, -1):
        model_path = '{0}/fine_tuning/domain-{1}_case-{2}_epoch-{3}.npz'.format(path, domain, case, epoch)
        if os.path.exists(model_path):
            return model_path

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

def return_item_type(num, dep_tag):
    if num == 0: return '照応なし'
    elif num == 1: return '発信者'
    elif num == 2: return '受信者'
    elif num == 3: return '項不定'
    else:
        if 'dep' in list(dep_tag):
            return '文内(dep)'
        elif 'zero' in list(dep_tag):
            return '文内(zero)'
        return '文内'

def predict(model_path, test_data, type_statistics_dict, domain, case):
    confusion_matrix = defaultdict(dict)

    feature_size = test_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, output_size=feature_size, n_labels=2, n_layers=1, dropout=0.2, type_statistics_dict=type_statistics_dict)

    serializers.load_npz(model_path, model)

    correct_num = {'all':0., '照応なし':0., '文内':0., '文内(dep)':0., '文内(zero)':0., '発信者':0., '受信者':0., '項不定':0.}
    case_num = {'all':0., '照応なし':0., '文内':0., '文内(dep)':0., '文内(zero)':0., '発信者':0., '受信者':0., '項不定':0.}
    accuracy = {'all':0., '照応なし':0., '文内':0., '文内(dep)':0., '文内(zero)':0., '発信者':0., '受信者':0., '項不定':0.}

    for key1 in correct_num.keys():
        for key2 in correct_num.keys():
            confusion_matrix[key1][key2] = 0

    cuda.get_device(0).use()
    model.to_gpu()

    mistake_list = []

    for xs, ys, ys_dep_tag, zs, word, is_verb in test_data:
        xs = cuda.to_gpu(xs)
        xs = Variable(xs)
        with chainer.using_config('train', False):
            pred_ys = model.traverse([xs], [zs])

        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]

        pred_ys = pred_ys[0].data.argmax(axis=0)[1]

        ys = ys.argmax()
        item_type = return_item_type(ys, ys_dep_tag)
        case_num['all'] += 1
        case_num[item_type] += 1
        if pred_ys == ys:
            correct_num['all'] += 1
            correct_num[item_type] += 1

        item_type = return_item_type(ys, [])
        pred_item_type = return_item_type(pred_ys, [])
        confusion_matrix[item_type][pred_item_type] += 1

        if pred_ys != ys:
            if item_type == '文内':
                item_type = ys-4
            if pred_item_type == '文内':
                pred_item_type = pred_ys-4
            sentence = ''.join(word[4:is_verb]) + '"' + word[is_verb:is_verb+1] + '"' + ''.join(word[is_verb+1:])
            mistake_list.append([item_type, pred_item_type, is_verb-4, sentence])

    correct_num['文内'] = correct_num['文内(dep)'] + correct_num['文内(zero)']
    case_num['文内'] = case_num['文内(dep)'] + case_num['文内(zero)']

    for key in accuracy:
        if case_num[key]:
            accuracy[key] = correct_num[key]/case_num[key]*100
        else:
            accuracy[key] = 999

    output_path = './' + 'predict'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dump_path = '{0}/domain-{1}_caes-{2}.tsv'.format(output_path, domain, case)
    print('model_path:{0}_domain:{1}_accuracy:{2:.2f}'.format('majority', domain, accuracy['all']))
    if not os.path.exists(dump_path):
        with open(dump_path, 'a') as f:
            f.write('model_path\tdomain\taccuracy(全体)\taccuracy(照応なし)\taccuracy(発信者)\taccuracy(受信者)\taccuracy(項不定)\taccuracy(文内)\taccuracy(文内(dep))\taccuracy(文内(zep))\ttest_data_size\n')
    with open(dump_path, 'a') as f:
        f.write('{0}\t{1}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.2f}\t{10}\n'.format('majority', domain, accuracy['all'], accuracy['照応なし'], accuracy['発信者'], accuracy['受信者'], accuracy['項不定'], accuracy['文内'], accuracy['文内(dep)'], accuracy['文内(zero)'], len(test_data)))

    output_path = './' + 'confusion_matrix'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dump_path = '{0}/domain-{1}_case-{2}.tsv'.format(output_path, domain, case)
    with open(dump_path, 'w') as f:
        f.write('model_path\t'+'majority'+'\n')
        f.write(' \t \t予測結果\n')
        f.write(' \t \t照応なし\t発信者\t受信者\t項不定\t文内\tsum(全体)\n実際の分類結果')
        for case_type in ['照応なし', '発信者', '受信者', '項不定', '文内']:
            f.write(' \t{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(case_type, confusion_matrix[case_type]['照応なし'], confusion_matrix[case_type]['発信者'], confusion_matrix[case_type]['受信者'], confusion_matrix[case_type]['項不定'], confusion_matrix[case_type]['文内'], case_num[case_type]))
        f.write('\n')
    
    output_path = './' + 'mistake_sentence'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = '{0}/domain-{1}_case-{2}'.format(output_path, domain, case)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dump_path = '{0}/model-{1}.txt'.format(output_path, 'majority')
    with open(dump_path, 'a') as f:
        f.write('model_path\t'+'majority'+'\n')
        f.write('正解位置\t予測位置\t述語位置\t文\n')
        for mistake in mistake_list:
            mistake = [str(i) for i in mistake]
            f.write('\t'.join(mistake))
            f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir','-m', type=str, default='')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    args = parser.parse_args()
    dataset_dict = load_dataset('../dataframe')
    train_dataset_dict = {}
    for domain in domain_dict:
        train_size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.7)
        train_dataset_dict['{0}_y_ga'.format(domain)] = dataset_dict['{0}_y_ga'.format(domain)][:train_size]
        train_dataset_dict['{0}_y_o'.format(domain)] = dataset_dict['{0}_y_o'.format(domain)][:train_size]
        train_dataset_dict['{0}_y_ni'.format(domain)] = dataset_dict['{0}_y_ni'.format(domain)][:train_size]

    for case in ['ga', 'o', 'ni']:
        type_statistics_dict = calculate_type_statistics(train_dataset_dict, case)
        for domain in domain_dict:
            model_path = load_model_path(args.dir, case, domain)

            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_z = dataset_dict['{0}_z'.format(domain)][size:]
            if case == 'ga':
                test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
                test_y_dep_tag = dataset_dict['{0}_y_ga_dep_tag'.format(domain)][size:]
            elif case == 'o':
                test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
                test_y_dep_tag = dataset_dict['{0}_y_o_dep_tag'.format(domain)][size:]
            elif case == 'ni':
                test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
                test_y_dep_tag = dataset_dict['{0}_y_ni_dep_tag'.format(domain)][size:]

            test_word = dataset_dict['{0}_word'.format(domain)][size:]
            test_is_verb = dataset_dict['{0}_is_verb'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y, test_y_dep_tag, test_z, test_word, test_is_verb)

            predict(model_path, test_data, type_statistics_dict, domain, case)

if __name__ == '__main__':
    main()
