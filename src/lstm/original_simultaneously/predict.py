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

from collections import defaultdict
import ipdb

domain_dict = {'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def load_model_path(path, part_flag=False):
    for domain in list(domain_dict) + ['union']:
        for epoch in range(20, 0, -1):
            model_path = '{0}/domain-{1}_epoch-{2}.npz'.format(path, domain, epoch)
            if os.path.exists(model_path):
                yield model_path
                break
    if part_flag:
        for part in range(10000, 190001, 30000):
            for epoch in range(20, 0, -1):
                model_path = '{0}/domain-union_pert_{1}_epoch-{2}.npz'.format(path, part, epoch)
                if os.path.exists(model_path):
                    yield model_path
                    break

def return_item_type(num):
    if num == 0: return '照応なし'
    elif num == 1: return '発信者'
    elif num == 2: return '受信者'
    elif num == 3: return '項不定'
    else: return '文内'

def predict(model_path, test_data, domain, args):
    feature_size = test_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, n_labels=3, n_layers=args.n_layers, dropout=args.dropout)
    serializers.load_npz(model_path, model)

    case_dict = {'ga':{'correct_num':defaultdict(float), 'item_num':defaultdict(float), 'accuracy':defaultdict(float)}, 'o':{'correct_num':defaultdict(float), 'item_num':defaultdict(float), 'accuracy':defaultdict(float)}, 'ni':{'correct_num':defaultdict(float), 'item_num':defaultdict(float), 'accuracy':defaultdict(float)}, 'all_case':{'correct_num':defaultdict(float), 'item_num':defaultdict(float), 'accuracy':defaultdict(float)}}

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    for xs, ys in test_data:
        xs = cuda.cupy.array(xs, dtype=cuda.cupy.float32)
        pred_ys = model.traverse([xs])[0]
        pred_ys = F.softmax(pred_ys)
        pred_ys = pred_ys.data.argmax(axis=0)
        ys = ys.argmax(axis=0)

        for i, case in enumerate(['ga', 'o', 'ni']):
            item_type = return_item_type(ys[i])
            case_dict[case]['item_num'][item_type] += 1
            case_dict[case]['item_num']['all'] += 1
            if pred_ys[i] == ys[i]:
                case_dict[case]['correct_num'][item_type] += 1
                case_dict[case]['correct_num']['all'] += 1

    for item_type in ['all', '照応なし', '文内', '発信者', '受信者', '項不定']:
        for case in ['ga', 'o', 'ni']:
            if case_dict[case]['item_num'][item_type]:
                case_dict[case]['accuracy'][item_type] = case_dict[case]['correct_num'][item_type]/case_dict[case]['item_num'][item_type]*100
            else:
                case_dict[case]['accuracy'][item_type] = None
            case_dict['all_case']['correct_num'][item_type] += case_dict[case]['correct_num'][item_type]
            case_dict['all_case']['item_num'][item_type] += case_dict[case]['item_num'][item_type]
        if case_dict['all_case']['item_num'][item_type]:
            case_dict['all_case']['accuracy'][item_type] = case_dict['all_case']['correct_num'][item_type]/case_dict['all_case']['item_num'][item_type]*100
        else:
            case_dict['all_case']['accuracy'][item_type] = None

    output_path = args.out
    if args.is_short:
        output_path += '_short'
    else:
        output_path += '_long'
    dump_path = '{0}/domain-{1}.tsv'.format(output_path, domain)
    print('model_path:{0}_domain:{1}_accuracy_all:{2:.3f}'.format(model_path, domain, case_dict['all_case']['accuracy']['all']))
    if not os.path.exists(dump_path):
        with open(dump_path, 'w') as f:
            f.write('model_path\tdomain\tcase\taccuracy(全体)\taccuracy(照応なし)\taccuracy(発信者)\taccuracy(受信者)\taccuracy(項不定)\taccuracy(文内)\ttest_data_size\n')
    with open(dump_path, 'a') as f:
        for case in ['ga', 'o', 'ni', 'all_case']:
            f.write('{0}\t{1}\t{2}\t{3:.3f}\t{4:.3f}\t{5:.3f}\t{6:.3f}\t{7:.3f}\t{8:.3f}\t{9}\n'.format(model_path, domain, case, case_dict[case]['accuracy']['all'], case_dict[case]['accuracy']['all'], case_dict[case]['accuracy']['照応なし'], case_dict[case]['accuracy']['発信者'], case_dict[case]['accuracy']['受信者'], case_dict[case]['accuracy']['項不定'], case_dict[case]['accuracy']['文内'], len(test_data)))

def main(train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', default='predict', help='Directory to output the result')
    parser.add_argument('--model_dir', '-m', type=str, default='')
    parser.add_argument('--part_flag', action='store_true')
    parser.add_argument('--is_short', action='store_true')
    args = parser.parse_args()

    dataset_dict = load_dataset(args.is_short)
    print('start data load domain-all')
    all_test_x = []
    all_test_y = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        all_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        all_test_y += dataset_dict['{0}_y'.format(domain)][size:]

    for model in load_model_path(args.model_dir, args.part_flag):
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_y)
        predict(model, test_data, 'union', args)
        for domain in domain_dict:
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict(model, test_data, domain, args)


if __name__ == '__main__':
    main()