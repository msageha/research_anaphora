import argparse
import pickle
import math
import json
import os
import datetime
import matplotlib
matplotlib.use('Agg')

import numpy as np
import chainer
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.training import extensions

from model import BiLSTMBase
from model import convert_seq
from train import load_dataset

domain_dict = {'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def fine_tuning(model_path, train_data, test_data, domain, case, dump_path, args):
    print('fine_tuning start domain-{0}, case-{1}'.format(domain, case))

    output_path = args.out
    if args.is_short:
        output_path += '_short'
    else:
        output_path += '_long'
    if not os.path.exists('./{0}/{1}'.format(output_path, dump_path)):
        os.mkdir('./{0}/{1}'.format(output_path, dump_path))
    output_path += '/' + dump_path
    if not os.path.exists('{0}/{1}'.format(output_path, 'args')):
        os.mkdir('{0}/{1}'.format(output_path, 'args'))
        os.mkdir('{0}/{1}'.format(output_path, 'log'))
        os.mkdir('{0}/{1}'.format(output_path, 'model'))
        os.mkdir('{0}/{1}'.format(output_path, 'tmpmodel'))
    
    print(json.dumps(args.__dict__, indent=2))
    with open('{0}/args/domain-{1}_case-{2}.json'.format(output_path, domain, case), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    feature_size = train_data[0][0].shape[1]
    model = BiLSTMBase(input_size=feature_size, n_labels=2, n_layers=args.n_layers, dropout=args.dropout) #saveした時と同じ構成にすること．
    serializers.load_npz(model_path, model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    #optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.alpha, beta1=args.beta1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weightdecay))

    if args.disable_update_lstm:
        model.nstep_bilstm.disable_update()

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=convert_seq)
    trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out=output_path)

    evaluator = chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu, converter=convert_seq)
    trigger = chainer.training.triggers.MaxValueTrigger(key='validation/main/accuracy', trigger=(1, 'epoch'))

    trainer.extend(evaluator, trigger=(1, 'epoch'))
    # trainer.extend(extensions.dump_graph(out_name="./graph/domain-{0}_case-{1}.dot".format(domain, case)))
    trainer.extend(extensions.LogReport(log_name='log/domain-{0}_case-{1}.log'.format(domain, case)), trigger=(1, 'epoch'))
    # trainer.extend(extensions.snapshot(filename='snapshot/domain-{0}_case-{1}_epoch-{{.updater.epoch}}'.format(domain, case)), trigger=(1, 'epoch'))
    # trainer.extend(extensions.MicroAverage('main/correct', 'main/total', 'main/accuracy'))
    # trainer.extend(extensions.MicroAverage('validation/ma in/correct', 'validation/main/total', 'validation/main/accuracy'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, savefun=serializers.save_npz ,filename='model/domain-{0}_case-{1}_epoch-{{.updater.epoch}}.npz'.format(domain, case)), trigger=trigger)

    trainer.run()

def main(train_test_ratio=0.8):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', '-n', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.3)
    parser.add_argument('--batchsize', '-b', type=int, default=30)
    parser.add_argument('--alpha', '-a', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--weightdecay', '-w', type=float, default=1e-4)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', type=str, default='fine_tuning', help='Directory to output the result')
    parser.add_argument('--model', '-m', type=str, default='')
    parser.add_argument('--case', '-c', type=str, default='')
    parser.add_argument('--disable_update_lstm', action='store_true')
    parser.add_argument('--is_short', action='store_true')
    args = parser.parse_args()
    dataset_dict = load_dataset(args.is_short)
    today = str(datetime.datetime.today())[:-16]
    if args.disable_update_lstm:
        today += '_disable_update_lstm'
    else:
        today += '_update_lstm'

    for domain in domain_dict:
        size = math.ceil(len(dataset_dict['{0}_x'.format(domain)])*train_test_ratio)
        train_x = dataset_dict['{0}_x'.format(domain)][:size]
        test_x = dataset_dict['{0}_x'.format(domain)][size:]
        if args.case == 'ga':
            train_y = dataset_dict['{0}_y_ga'.format(domain)][:size]
            test_y = dataset_dict['{0}_y_ga'.format(domain)][size:]
        elif args.case == 'o':
            train_y = dataset_dict['{0}_y_o'.format(domain)][:size]
            test_y = dataset_dict['{0}_y_o'.format(domain)][size:]
        elif args.case == 'ni':
            train_y = dataset_dict['{0}_y_ni'.format(domain)][:size]
            test_y = dataset_dict['{0}_y_ni'.format(domain)][size:]
        train_data = tuple_dataset.TupleDataset(train_x, train_y)
        test_data  = tuple_dataset.TupleDataset(test_x, test_y)
        fine_tuning('{0}'.format(args.model), train_data, test_data, domain, args.case, today, args)

if __name__ == '__main__':
    main()