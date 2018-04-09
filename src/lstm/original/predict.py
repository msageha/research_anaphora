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

# def load_dataset_without_dep():
#     dataset_dict = {}
#     for domain in domain_dict:
#         print('start data load domain-{0}'.format(domain))
#         with open('./dataframe/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
#             df_list = pickle.load(f)
#         x_ga_dataset = []
#         y_ga_dataset = []
#         x_o_dataset = []
#         y_o_dataset = []
#         x_ni_dataset = []
#         y_ni_dataset = []
#         size = math.ceil(len(df_list)*0.8)
#         for df in df_list[size:]:
#             y_ga = np.array(df['ga_case'], dtype=np.int32)
#             y_o = np.array(df['o_case'], dtype=np.int32)
#             y_ni = np.array(df['ni_case'], dtype=np.int32)
#             x = df.drop('ga_case', axis=1).drop('o_case', axis=1).drop('ni_case', axis=1).drop('ga_dep_tag', axis=1).drop('o_dep_tag', axis=1).drop('ni_dep_tag', axis=1)
#             x = np.array(x, dtype=np.float32)
#             if df['ga_dep_tag'].any() != 'dep':
#                 x_ga_dataset.append(x)
#                 y_ga_dataset.append(y_ga)
#             if df['o_dep_tag'].any() != 'dep':
#                 x_o_dataset.append(x)
#                 y_o_dataset.append(y_o)
#             if df['ni_dep_tag'].any() != 'dep':
#                 x_ni_dataset.append(x)
#                 y_ni_dataset.append(y_ni)
#         print('domain-{0}_case-ga_size-{1}'.format(domain, len(y_ga_dataset)))
#         print('domain-{0}_case-o_size-{1}'.format(domain, len(y_o_dataset)))
#         print('domain-{0}_case-ni_size-{1}'.format(domain, len(y_ni_dataset)))
#         dataset_dict['{0}_x_ga'.format(domain)] = x_ga_dataset
#         dataset_dict['{0}_y_ga'.format(domain)] = y_ga_dataset
#         dataset_dict['{0}_x_o'.format(domain)] = x_o_dataset
#         dataset_dict['{0}_y_o'.format(domain)] = y_o_dataset
#         dataset_dict['{0}_x_ni'.format(domain)] = x_ni_dataset
#         dataset_dict['{0}_y_ni'.format(domain)] = y_ni_dataset

def load_model_path(path, case, part_flag=False):
    for domain in list(domain_dict) + ['union']:
        for epoch in range(20, 0, -1):
            model_path = '{0}/domain-{1}_case-{2}_epoch-{3}.npz'.format(path, domain, case, epoch)
            if os.path.exists(model_path):
                yield model_path
                break
    if part_flag:
        for part in range(1000, 19001, 3000):
            for epoch in range(20, 0, -1):
                model_path = '{0}/domain-union_pert_{1}_case-{2}_epoch-{3}.npz'.format(path, part, case, epoch)
                if os.path.exists(model_path):
                    yield model_path
                    break


def predict(model_path, test_data, domain, case, args):

    feature_size = test_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout)
    serializers.load_npz(model_path, model)
    accuracy = 0.

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    for xs, ys in test_data:
        xs = cuda.cupy.array(xs, dtype=cuda.cupy.float32)
        pred_ys = model.traverse([xs])
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=0)[1] for pred_y in pred_ys]
        ys = ys.argmax()
        if pred_ys == ys:
            accuracy += 1
    accuracy /= len(test_data)
    print('model_path:{0}_domain:{1}_accuracy:{2:.3f}'.format(model_path, domain, accuracy*100))
    if not os.path.exists('{0}/domain-{1}_caes-{2}.tsv'.format(args.out, domain, case)):
        with open('{0}/domain-{1}_caes-{2}.txt'.format(args.out, domain, case), 'w') as f:
            f.write('model_path\tdomain\taccuracy\ttest_data_size\n')
    with open('{0}/domain-{1}_caes-{2}.txt'.format(args.out, domain, case), 'a') as f:
        f.write('{0}\t{1}\t{2:.3f}\t{3}'.format(model_path, domain, accuracy*100, len(test_data)))

def main(train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', default='predict', help='Directory to output the result')
    parser.add_argument('--model_dir', '-m', type=str, default='')
    parser.add_argument('--case', '-c', type=str, default='')
    parser.add_argument('--part_flag', action='store_true')
    args = parser.parse_args()

    dataset_dict = load_dataset()
    print('start data load domain-all')
    all_test_x = []
    all_test_ga = []
    all_test_o = []
    all_test_ni = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        all_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        all_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
        all_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        all_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]

    for model in load_model_path(args.model_dir, args.case, args.part_flag):
        if args.case == 'ga':
            test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ga)
        elif args.case == 'o':
            test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_o)
        elif args.case == 'ni':
            test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ni)
        predict(model, test_data, 'all', args.case, args)
        for domain in domain_dict:
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            if args.case == 'ga':
                test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
            elif args.case == 'o':
                test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
            elif args.case == 'ni':
                test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict(model, test_data, domain, args.case, args)


if __name__ == '__main__':
    main()