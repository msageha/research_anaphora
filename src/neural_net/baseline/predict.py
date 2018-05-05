import argparse
import pickle
import math
import json
from collections import OrderedDict

import chainer
import chainer.functions as F
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer import cuda

from model import BiLSTMBase
from train import load_dataset
import os

domain_dict = OrderedDict([('OC', 'Yahoo!知恵袋'), ('OY', 'Yahoo!ブログ'), ('OW', '白書'), ('PB', '書籍'), ('PM', '雑誌'), ('PN', '新聞')])

def load_model_path(path, case):
    for domain in list(domain_dict) + ['union']:
        for epoch in range(20, 0, -1):
            model_path = '{0}/model/domain-{1}_case-{2}_epoch-{3}.npz'.format(path, domain, case, epoch)
            if 'outdomain' in path:
                model_path = '{0}/model/domain-out-{1}_case-{2}_epoch-{3}.npz'.format(path, domain, case, epoch)
            if os.path.exists(model_path):
                yield model_path
                break

def return_item_type(num, dep_tag):
    if num == 0: return '照応なし'
    elif num == 1: return '発信者'
    elif num == 2: return '受信者'
    elif num == 3: return '項不定'
    else:
        if 'dep' == list(dep_tag)[num]:
            return '文内(dep)'
        elif 'zero' == list(dep_tag)[num]:
            return '文内(zero)'
        return '文内'

def predict(model_path, test_data, domain, case, args):
    with open('{0}/args/domain-{1}_case-{2}.json'.format(args.df_path, domain, case)) as f:
        tmp = json.load(f)
        for key in tmp.keys():
            args.__dict__[key] = tmp[key]
    
    feature_size = test_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, output_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout)
    serializers.load_npz(model_path, model)
    correct_num = {'all':0., '照応なし':0., '文内(dep)':0., '文内(zero)':0., '発信者':0., '受信者':0., '項不定':0.}
    case_num = {'all':0., '照応なし':0., '文内(dep)':0., '文内(zero)':0., '発信者':0., '受信者':0., '項不定':0.}
    accuracy = {'all':0., '照応なし':0., '文内(dep)':0., '文内(zero)':0., '発信者':0., '受信者':0., '項不定':0.}

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # #optimizer
    # optimizer = chainer.optimizers.Adam()
    # optimizer.setup(model)

    for xs, ys, ys_dep_tag in test_data:
        xs = cuda.cupy.array(xs, dtype=cuda.cupy.float32)
        pred_ys = model.traverse([xs])
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=0)[1] for pred_y in pred_ys]
        pred_ys = int(pred_ys[0])
        ys = ys.argmax()
        item_type = return_item_type(ys, ys_dep_tag)
        case_num['all'] += 1
        case_num[item_type] += 1
        if pred_ys == ys:
            correct_num['all'] += 1
            correct_num[item_type] += 1

    for key in accuracy:
        if case_num[key]:
            accuracy[key] = correct_num[key]/case_num[key]*100
        else:
            accuracy[key] = 999

    output_path = args.dir + '/' + 'predict'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dump_path = '{0}/domain-{1}_caes-{2}.tsv'.format(output_path, domain, case)
    print('model_path:{0}_domain:{1}_accuracy:{2:.2f}'.format(model_path, domain, accuracy['all']))
    if not os.path.exists(dump_path):
        with open(dump_path, 'w') as f:
            f.write('model_path\tdomain\taccuracy(全体)\taccuracy(照応なし)\taccuracy(発信者)\taccuracy(受信者)\taccuracy(項不定)\taccuracy(文内)\ttest_data_size\n')
    with open(dump_path, 'a') as f:
        f.write('{0}\t{1}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8}\n'.format(model_path, domain, accuracy['all'], accuracy['照応なし'], accuracy['発信者'], accuracy['受信者'], accuracy['項不定'], accuracy['文内'], len(test_data)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--dir', '-m', type=str, default='')
    parser.add_argument('--df_path', default='../dataframe')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    args = parser.parse_args()

    dataset_dict = load_dataset(args.df_path)
    print('start data load domain union')
    union_test_x = []
    union_test_ga = []
    union_test_o = []
    union_test_ni = []
    union_test_ga_dep_tag = []
    union_test_o_dep_tag = []
    union_test_ni_dep_tag = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
        union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        union_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
        union_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        union_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
        union_test_ga_dep_tag += dataset_dict['{0}_y_ga_dep_tag'.format(domain)][size:]
        union_test_o_dep_tag += dataset_dict['{0}_y_o_dep_tag'.format(domain)][size:]
        union_test_ni_dep_tag += dataset_dict['{0}_y_ni_dep_tag'.format(domain)][size:]
    for case in ['ga', 'o', 'ni']:
        for model_path in load_model_path(args.dir, case):
            if case == 'ga':
                test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ga, union_test_ga_dep_tag)
            elif case == 'o':
                test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_o, union_test_o_dep_tag)
            elif case == 'ni':
                test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ni, union_test_ni_dep_tag)
            predict(model_path, test_data, 'union', case, args)
            for domain in domain_dict:
                size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
                test_x = dataset_dict['{0}_x'.format(domain)][size:]
                if case == 'ga':
                    test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
                    test_y_dep_tag = dataset_dict['{0}_y_ga_dep_tag'.format(domain)][size:]
                elif case == 'o':
                    test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
                    test_y_dep_tag = dataset_dict['{0}_y_o_dep_tag'.format(domain)][size:]
                elif case == 'ni':
                    test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
                    test_y_dep_tag = dataset_dict['{0}_y_ni_dep_tag'.format(domain)][size:]
                test_data  = tuple_dataset.TupleDataset(test_x, test_y, test_y_dep_tag)
                predict(model_path, test_data, domain, case, args)

if __name__ == '__main__':
    main()