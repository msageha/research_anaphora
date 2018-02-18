import argparse
import pickle
import math
import json

import numpy as np
import chainer
import chainer.functions as F
from chainer.datasets import tuple_dataset
from chainer import serializers

from model import BiLSTMBase
from train import load_dataset

domain_dict = {'OC':'Yahoo!知恵袋', 'OW':'白書', 'OY':'Yahoo!ブログ', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def predict(model_path, test_data, domain):

    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.5)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='predict', help='Directory to output the result')
    args = parser.parse_args()

    feature_size = test_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout)
    serializers.load_npz(model_path, model)

    for xs, ys in test_data:
        pred_ys = model.traverse([xs])
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=0)[1] for pred_y in pred_ys]
        ys = ys.argmax()
        if pred_ys == ys:
            accuracy += 1
    accuracy /= len(test_data)
    print('model_path:{0}_domain:{1}_accuracy:{2}'.format(model_path, domain, accuracy))
    with open('{0}/domain-{1}_accuracy.txt'.format(args.out, domain), 'a') as f:
        f.write('model_path:{0}\tdomain:{1}\taccuracy:{2}\ttest_data_size:{3}'.format(model_path, domain, accuracy, len(test_data)))

def main():
    dataset_dict = load_dataset()
    model_dir = './result/model'
    model_list = ['domain-OC_case-ga_epoch-10.npz', 'domain-OW_case-ga_epoch-10.npz', 'domain-OY_case-ga_epoch-10.npz',
        'domain-PB_case-ga_epoch-10.npz', 'domain-PM_case-ga_epoch-10.npz', 'domain-PN_case-ga_epoch-10.npz', 'domain-all_case-ga_epoch-10.npz']
    print('start data load domain-all')
    all_test_x = []
    all_test_ga = []]
    all_test_o = []]
    all_test_ni = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
        all_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        all_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
        all_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        all_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
    for file in model_list
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ga)
        predict('{0}/{1}'.format(model_dir, file), test_data, 'all')
        for domain in domain_dict:
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict('{0}/{1}'.format(model_dir, file), test_data, domain)

    model_list = ['domain-OC_case-o_epoch-10.npz', 'domain-OW_case-o_epoch-10.npz', 'domain-OY_case-o_epoch-10.npz',
        'domain-PB_case-o_epoch-10.npz', 'domain-PM_case-o_epoch-10.npz', 'domain-PN_case-o_epoch-10.npz', 'domain-all_case-o_epoch-10.npz']
    for file in model_list:
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_o)
        predict('{0}/{1}'.format(model_dir, file), test_data, 'all')
        for domain in domain_dict:
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict('{0}/{1}'.format(model_dir, file), test_data, domain)

    model_list = ['domain-OC_case-ni_epoch-10.npz', 'domain-OW_case-ni_epoch-10.npz', 'domain-OY_case-ni_epoch-10.npz',
        'domain-PB_case-ni_epoch-10.npz', 'domain-PM_case-ni_epoch-10.npz', 'domain-PN_case-ni_epoch-10.npz', 'domain-all_case-ni_epoch-10.npz']
    for file in model_list:
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ni)
        predict('{0}/{1}'.format(model_dir, file), test_data, 'all')
        for domain in domain_dict:
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict('{0}/{1}'.format(model_dir, file), test_data, domain)