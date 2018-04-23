import argparse
import pickle
import math
import json

import numpy as np
import chainer
import chainer.functions as F
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer import Variable
from chainer import cuda

from model import BiLSTMBase
from train import load_dataset
import os

domain_dict = {'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def load_model_path(path, case, part_flag=False):
    for epoch in range(20, 0, -1):
        model_path = '{0}/domain-union_case-{1}_epoch-{2}.npz'.format(path, case, epoch)
        if os.path.exists(model_path):
            yield model_path
            break
    for domain in list(domain_dict):
        for epoch in range(20, 0, -1):
            model_path = '{0}/domain-out-{1}_case-{2}_epoch-{3}.npz'.format(path, domain, case, epoch)
            if os.path.exists(model_path):
                yield model_path
                break
    if part_flag:
        for part in range(10000, 190001, 30000):
            for epoch in range(20, 0, -1):
                model_path = '{0}/domain-union_pert_{1}_case-{2}_epoch-{3}.npz'.format(path, part, case, epoch)
                if os.path.exists(model_path):
                    yield model_path
                    break

def return_item_type(num):
    if num == 0: return '照応なし'
    elif num == 1: return '発信者'
    elif num == 2: return '受信者'
    elif num == 3: return '項不定'
    else: return '文内'

def predict(model_path, test_data, domain, case, args):

    feature_size = test_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout, case=case)
    serializers.load_npz(model_path, model)
    correct_num = {'all':0., '照応なし':0., '文内':0., '発信者':0., '受信者':0., '項不定':0.}
    case_num = {'all':0., '照応なし':0., '文内':0., '発信者':0., '受信者':0., '項不定':0.}
    accuracy = {'all':0., '照応なし':0., '文内':0., '発信者':0., '受信者':0., '項不定':0.}

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    for xs, ys, zs in test_data:
        xs = cuda.cupy.array(xs, dtype=cuda.cupy.float32)
        pred_ys = model.traverse([xs], [zs])
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=0)[1] for pred_y in pred_ys]
        pred_ys = int(pred_ys[0])
        ys = ys.argmax()
        item_type = return_item_type(ys)
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

    output_path = args.out
    if args.is_short:
        output_path += '_short'
    else:
        output_path += '_long'
    dump_path = '{0}/domain-{1}_caes-{2}.tsv'.format(output_path, domain, case)
    print('model_path:{0}_domain:{1}_accuracy:{2:.3f}'.format(model_path, domain, accuracy['all']))
    if not os.path.exists(dump_path):
        with open(dump_path, 'w') as f:
            f.write('model_path\tdomain\taccuracy(全体)\taccuracy(照応なし)\taccuracy(発信者)\taccuracy(受信者)\taccuracy(項不定)\taccuracy(文内)\ttest_data_size\n')
    with open(dump_path, 'a') as f:
        f.write('{0}\t{1}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t{5:.3f}\t{6:.3f}\t{7:.3f}\t{8}\n'.format(model_path, domain, accuracy['all'], accuracy['照応なし'], accuracy['発信者'], accuracy['受信者'], accuracy['項不定'], accuracy['文内'], len(test_data)))

def main(train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', default='predict', help='Directory to output the result')
    parser.add_argument('--model_dir', '-m', type=str, default='')
    parser.add_argument('--part_flag', action='store_true')
    parser.add_argument('--df_path', default='../original1/dataframe_long')
    args = parser.parse_args()

    dataset_dict = load_dataset(args.df_path)
    print('start data load domain-union')
    union_test_x = []
    union_test_ga = []
    union_test_o = []
    union_test_ni = []
    union_test_z = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        union_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
        union_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        union_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
        union_test_z += dataset_dict['{0}_z'.format(domain)][size:]

    for case in ['ga', 'o', 'ni']:
        for model in load_model_path(args.model_dir, case, args.part_flag):
            if case == 'ga':
                test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ga, union_test_z)
            elif case == 'o':
                test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_o, union_test_z)
            elif case == 'ni':
                test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ni, union_test_z)
            predict(model, test_data, 'union', case, args)
            
            for domain in domain_dict:
                size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
                test_x = dataset_dict['{0}_x'.format(domain)][size:]
                test_z = dataset_dict['{0}_z'.format(domain)][size:]

                if case == 'ga':
                    test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
                elif case == 'o':
                    test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
                elif case == 'ni':
                    test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
                
                test_data = tuple_dataset.TupleDataset(test_x, test_y, test_z)
                predict(model, test_data, domain, case, args)


if __name__ == '__main__':
    main()