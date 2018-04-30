import argparse
import pickle
import math
import json
import os
import random
from collections import OrderedDict

# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import chainer
from chainer import cuda
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.training import extensions

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

def load_dataset(dataset_path):
    dataset_dict = {}
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        dataset = np.load('{0}/{1}.npz'.format(dataset_path, domain))
        x_dataset = dataset['x']
        y_ga_dataset = dataset['y_ga']
        y_ga_dep_tag_dataset = dataset['y_ga_dep_tag']
        y_o_dataset = dataset['y_o']
        y_o_dep_tag_dataset = dataset['y_o_dep_tag']
        y_ni_dataset = dataset['y_ni']
        y_ni_dep_tag_dataset = dataset['y_ni_dep_tag']
        dataset_dict['{0}_x'.format(domain)] = x_dataset
        dataset_dict['{0}_y_ga'.format(domain)] = y_ga_dataset
        dataset_dict['{0}_y_o'.format(domain)] = y_o_dataset
        dataset_dict['{0}_y_ni'.format(domain)] = y_ni_dataset
        dataset_dict['{0}_y_ga_dep_tag'.format(domain)] = y_ga_dep_tag_dataset
        dataset_dict['{0}_y_o_dep_tag'.format(domain)] = y_o_dep_tag_dataset
        dataset_dict['{0}_y_ni_dep_tag'.format(domain)] = y_ni_dep_tag_dataset
    return dataset_dict

def training(train_data, test_data, domain, case, dump_path, args):
    print('training start domain-{0}, case-{1}'.format(domain, case))
    set_random_seed(args.seed)

    if not os.path.exists('{0}'.format(dump_path)):
        os.mkdir('{0}'.format(dump_path))
    if not os.path.exists('{0}/{1}'.format(dump_path, 'args')):
        os.mkdir('{0}/{1}'.format(dump_path, 'args'))
        os.mkdir('{0}/{1}'.format(dump_path, 'log'))
        os.mkdir('{0}/{1}'.format(dump_path, 'model'))
        os.mkdir('{0}/{1}'.format(dump_path, 'tmpmodel'))
        os.mkdir('{0}/{1}'.format(dump_path, 'graph'))

    with open('{0}/args/domain-{1}_case-{2}.json'.format(dump_path, domain, case), 'w') as f:
        args.__dict__['train_size'] = len(train_data)
        args.__dict__['test_size'] = len(test_data)
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))

    feature_size = train_data[0][0].shape[1]

    model = BiLSTMBase(input_size=feature_size, output_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=convert_seq)
    trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out=dump_path)

    evaluator = chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu, converter=convert_seq)
    trigger = chainer.training.triggers.MaxValueTrigger(key='validation/main/accuracy', trigger=(1, 'epoch'))

    trainer.extend(evaluator, trigger=(1, 'epoch'))
    # trainer.extend(extensions.dump_graph(root_name='main/loss', out_name="./graph/domain-{0}_case-{1}.dot".format(domain, case)))
    trainer.extend(extensions.LogReport(log_name='log/domain-{0}_case-{1}.log'.format(domain, case)), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(model, savefun=serializers.save_npz ,filename='model/domain-{0}_case-{1}_epoch-{{.updater.epoch}}.npz'.format(domain, case)), trigger=trigger)
    # trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], file_name='./graph/accuracy_domain-{0}_case-{1}.png'.format(domain, case), x_key='epoch'))
    # trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], file_name='./graph/loss_domain-{0}_case-{1}.png'.format(domain, case), x_key='epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.run()

def in_domain(dataset_dict, args, dump_path):
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
        train_x = dataset_dict['{0}_x'.format(domain)][:size]
        test_x = dataset_dict['{0}_x'.format(domain)][size:]
        train_y = dataset_dict['{0}_y_ga'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ga', dump_path, args)
        train_y = dataset_dict['{0}_y_o'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'o', dump_path, args)
        train_y = dataset_dict['{0}_y_ni'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ni', dump_path, args)
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
        print(domain)
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

def out_domain(dataset_dict, args, dump_path):
    print('start data load out_domain')
    for out_domain in domain_dict:
        union_train_x = []
        union_test_x = []
        union_train_ga = []
        union_test_ga = []
        union_train_o = []
        union_test_o = []
        union_train_ni = []
        union_test_ni = []
        for domain in domain_dict:
            if out_domain == domain:
                continue
            size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
            union_train_x += dataset_dict['{0}_x'.format(domain)][:size]
            union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
            union_train_ga += dataset_dict['{0}_y_ga'.format(domain)][:size]
            union_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
            union_train_o += dataset_dict['{0}_y_o'.format(domain)][:size]
            union_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
            union_train_ni += dataset_dict['{0}_y_ni'.format(domain)][:size]
            union_test_ni += dataset_dict['{0}_y_ni'.format(domain)][size:]
        print('out domain {0}\tdata_size {1}'.format(out_domain, len(union_train_x)))
        train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ga)
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ga)
        training(train_data, test_data, 'out-{0}'.format(out_domain), 'ga', dump_path, args)
        train_data = tuple_dataset.TupleDataset(union_train_x, union_train_o)
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_o)
        training(train_data, test_data, 'out-{0}'.format(out_domain), 'o', dump_path, args)
        train_data = tuple_dataset.TupleDataset(union_train_x, union_train_ni)
        test_data  = tuple_dataset.TupleDataset(union_test_x, union_test_ni)
        training(train_data, test_data, 'out-{0}'.format(out_domain), 'ni', dump_path, args)

def arrange(dataset_dict, args, dump_path):
    arrange_size= min([len(dataset_dict['{0}_x'.format(domain)]) for domain in domain_dict])
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*args.train_test_ratio)
        train_x = dataset_dict['{0}_x'.format(domain)][:arrange_size]
        test_x = dataset_dict['{0}_x'.format(domain)][size:]
        train_y = dataset_dict['{0}_y_ga'.format(domain)][:arrange_size]
        test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ga', dump_path, args)
        train_y = dataset_dict['{0}_y_o'.format(domain)][:arrange_size]
        test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'o', dump_path, args)
        train_y = dataset_dict['{0}_y_ni'.format(domain)][:arrange_size]
        test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(train_data, test_data, domain, 'ni', dump_path, args)
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
        union_train_x += dataset_dict['{0}_x'.format(domain)][:arrange_size]
        union_test_x += dataset_dict['{0}_x'.format(domain)][size:]
        union_train_ga += dataset_dict['{0}_y_ga'.format(domain)][:arrange_size]
        union_test_ga += dataset_dict['{0}_y_ga'.format(domain)][size:]
        union_train_o += dataset_dict['{0}_y_o'.format(domain)][:arrange_size]
        union_test_o += dataset_dict['{0}_y_o'.format(domain)][size:]
        union_train_ni += dataset_dict['{0}_y_ni'.format(domain)][:arrange_size]
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
    parser.add_argument('--dropout', '-d', type=float, default=0.2)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--dataset_path', default='./dataset')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    parser.add_argument('--seed', default=1)
    args = parser.parse_args()

    dataset_dict = load_dataset(args.dataset_path)
    in_domain(dataset_dict, args, 'normal/dropout-{0}_batchsize-{1}'.format(args.dropout, args.batchsize))
    out_domain(dataset_dict, args, 'outdomain/dropout-{0}_batchsize-{1}'.format(args.dropout, args.batchsize))
    arrange(dataset_dict, args, 'arranged/dropout-{0}_batchsize-{1}'.format(args.dropout, args.batchsize))

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