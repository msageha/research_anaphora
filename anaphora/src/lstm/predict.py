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

domain_dict = {'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def load_dataset_without_dep():
    dataset_dict = {}
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        with open('./dataframe/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
            df_list = pickle.load(f)
        x_ga_dataset = []
        y_ga_dataset = []
        x_o_dataset = []
        y_o_dataset = []
        x_ni_dataset = []
        y_ni_dataset = []
        size = math.ceil(len(df_list)*0.8)
        for df in df_list[size:]:
            y_ga = np.array(df['ga_case'], dtype=np.int32)
            y_o = np.array(df['o_case'], dtype=np.int32)
            y_ni = np.array(df['ni_case'], dtype=np.int32)
            x = df.drop('ga_case', axis=1).drop('o_case', axis=1).drop('ni_case', axis=1).drop('ga_dep_tag', axis=1).drop('o_dep_tag', axis=1).drop('ni_dep_tag', axis=1)
            x = np.array(x, dtype=np.float32)
            if df['ga_dep_tag'].any() != 'dep':
                x_ga_dataset.append(x)
                y_ga_dataset.append(y_ga)
            if df['o_dep_tag'].any() != 'dep':
                x_o_dataset.append(x)
                y_o_dataset.append(y_o)
            if df['ni_dep_tag'].any() != 'dep':
                x_ni_dataset.append(x)
                y_ni_dataset.append(y_ni)
        print('domain-{0}_case-ga_size-{1}'.format(domain, len(y_ga_dataset)))
        print('domain-{0}_case-o_size-{1}'.format(domain, len(y_o_dataset)))
        print('domain-{0}_case-ni_size-{1}'.format(domain, len(y_ni_dataset)))
        dataset_dict['{0}_x_ga'.format(domain)] = x_ga_dataset
        dataset_dict['{0}_y_ga'.format(domain)] = y_ga_dataset
        dataset_dict['{0}_x_o'.format(domain)] = x_o_dataset
        dataset_dict['{0}_y_o'.format(domain)] = y_o_dataset
        dataset_dict['{0}_x_ni'.format(domain)] = x_ni_dataset
        dataset_dict['{0}_y_ni'.format(domain)] = y_ni_dataset

def predict(model_path, test_data, domain, train_type):

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
    with open('{0}/domain-{1}_accuracy_{2}.txt'.format(args.out, domain, train_type), 'a') as f:
        f.write('model_path:{0}\tdomain:{1}\taccuracy:{2}\ttest_data_size:{3}'.format(model_path, domain, accuracy, len(test_data)))

def main():
    dataset_dict = load_dataset()
    model_dir = './result/model'
    model_list = ['domain-OC_case-ga_epoch-10.npz', 'domain-OW_case-ga_epoch-10.npz', 'domain-OY_case-ga_epoch-10.npz',
        'domain-PB_case-ga_epoch-10.npz', 'domain-PM_case-ga_epoch-10.npz', 'domain-PN_case-ga_epoch-10.npz', 'domain-all_case-ga_epoch-10.npz']
    print('start data load domain-all')
    all_test_x = []
    all_test_ga = []
    all_test_o = []
    all_test_ni = []
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
        all_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        all_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
        all_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        all_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
    for file in model_list:
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ga)
        predict('{0}/{1}'.format(model_dir, file), test_data, 'all', 'result')
        for domain in domain_dict:
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict('{0}/{1}'.format(model_dir, file), test_data, domain, 'result')

    model_list = ['domain-OC_case-o_epoch-10.npz', 'domain-OW_case-o_epoch-10.npz', 'domain-OY_case-o_epoch-10.npz',
        'domain-PB_case-o_epoch-10.npz', 'domain-PM_case-o_epoch-10.npz', 'domain-PN_case-o_epoch-10.npz', 'domain-all_case-o_epoch-10.npz']
    for file in model_list:
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_o)
        predict('{0}/{1}'.format(model_dir, file), test_data, 'all','result')
        for domain in domain_dict:
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict('{0}/{1}'.format(model_dir, file), test_data, domain,'result')

    model_list = ['domain-OC_case-ni_epoch-10.npz', 'domain-OW_case-ni_epoch-10.npz', 'domain-OY_case-ni_epoch-10.npz',
        'domain-PB_case-ni_epoch-10.npz', 'domain-PM_case-ni_epoch-10.npz', 'domain-PN_case-ni_epoch-10.npz', 'domain-all_case-ni_epoch-10.npz']
    for file in model_list:
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ni)
        predict('{0}/{1}'.format(model_dir, file), test_data, 'all', 'result')
        for domain in domain_dict:
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict('{0}/{1}'.format(model_dir, file), test_data, domain, 'result')

    model_dir = './fine_tuing/model'
    model_list = ['domain-OC_case-ga_epoch-10.npz', 'domain-OW_case-ga_epoch-10.npz', 'domain-OY_case-ga_epoch-10.npz',
        'domain-PB_case-ga_epoch-10.npz', 'domain-PM_case-ga_epoch-10.npz', 'domain-PN_case-ga_epoch-10.npz', 'domain-all_case-ga_epoch-10.npz']
    print('start data load domain-all')
    for file in model_list:
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ga)
        predict('{0}/{1}'.format(model_dir, file), test_data, 'all', 'fine_tuning')
        for domain in domain_dict:
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict('{0}/{1}'.format(model_dir, file), test_data, domain, 'fine_tuning')

    model_list = ['domain-OC_case-o_epoch-10.npz', 'domain-OW_case-o_epoch-10.npz', 'domain-OY_case-o_epoch-10.npz',
        'domain-PB_case-o_epoch-10.npz', 'domain-PM_case-o_epoch-10.npz', 'domain-PN_case-o_epoch-10.npz', 'domain-all_case-o_epoch-10.npz']
    for file in model_list:
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_o)
        predict('{0}/{1}'.format(model_dir, file), test_data, 'all','fine_tuning')
        for domain in domain_dict:
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict('{0}/{1}'.format(model_dir, file), test_data, domain,'fine_tuning')

    model_list = ['domain-OC_case-ni_epoch-10.npz', 'domain-OW_case-ni_epoch-10.npz', 'domain-OY_case-ni_epoch-10.npz',
        'domain-PB_case-ni_epoch-10.npz', 'domain-PM_case-ni_epoch-10.npz', 'domain-PN_case-ni_epoch-10.npz', 'domain-all_case-ni_epoch-10.npz']
    for file in model_list:
        test_data  = tuple_dataset.TupleDataset(all_test_x, all_test_ni)
        predict('{0}/{1}'.format(model_dir, file), test_data, 'all', 'fine_tuning')
        for domain in domain_dict:
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
            test_x = dataset_dict['{0}_x'.format(domain)][size:]
            test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
            test_data  = tuple_dataset.TupleDataset(test_x, test_y)
            predict('{0}/{1}'.format(model_dir, file), test_data, domain, 'fine_tuning')