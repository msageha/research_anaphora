import argparse
import pickle
import math
import json

import numpy as np
import chainer
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.training import extensions

from model import BiLSTMBase
from model import convert_seq
from train import load_dataset

domain_dict = {'OC':'Yahoo!知恵袋', 'OW':'白書', 'OY':'Yahoo!ブログ', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def fine_tuning(model_path, train_data, test_data, domain, case):
    print('training start domain-{0}, case-{1}'.format(domain, case))
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.5)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--alpha', '-a', type=float, default=1e-3)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='fine_tuning', help='Directory to output the result')
    args = parser.parse_args()

    print(json.dumps(args.__dict__, indent=2))
    with open('args/domain-{0}_case-{1}.json'.format(domain, case), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    feature_size = train_data[0][0].shape[1]
    model = BiLSTMBase(input_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout) #saveした時と同じ構成にすること．
    serializers.load_npz(model_path, model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    #optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.alpha)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=convert_seq)
    trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out=args.out)

    evaluator = chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu, converter=convert_seq)

    trainer.extend(evaluator, trigger=(1000, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')), trigger=(100, 'iteration'))
    trainer.extend(extensions.MicroAverage('main/correct', 'main/total', 'main/accuracy'))
    trainer.extend(extensions.MicroAverage('validation/main/correct', 'validation/main/total', 'validation/main/accuracy'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(model,savefun=serializers.save_npz ,filename='model/domain-{0}_case-{1}_epoch-{{.updater.epoch}}.npz'), trigger=(1, 'epoch'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy/domain-{0}_case-{1}.png'.format(domain, case)))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()

def main(model_path):
    dataset_dict = load_dataset()
    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*0.8)
        train_x = dataset_dict['{0}_x'.format(domain)][:size]
        test_x = dataset_dict['{0}_x'.format(domain)][size:]
        train_y = dataset_dict['{0}_y_ga'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(model_path, train_data, test_data, domain, 'ga')
        train_y = dataset_dict['{0}_y_o'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(model_path, train_data, test_data, domain, 'o')
        train_y = dataset_dict['{0}_y_ni'.format(domain)][:size]
        test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        training(model_path, train_data, test_data, domain, 'ni')

if __name__ == '__main__':
    main(model_path)