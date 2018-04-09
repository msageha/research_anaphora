import argparse
import pickle
import math
import json
import datetime
import os

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


domain_dict = {'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def load_dataset(dataframe_path):
    zeros = np.zeros((30, 234))
    le = preprocessing.LabelEncoder()
    le.fit(list(domain_dict.keys()))
    x_dataset = []
    y_dataset = []
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        with open('{0}/dataframe_list_{1}.pickle'.format(dataframe_path, domain), 'rb') as f:
            df_list = pickle.load(f)
        for df in df_list:
            df = df.drop('ga_case', axis=1).drop('o_case', axis=1).drop('ni_case', axis=1).drop('ga_dep_tag', axis=1).drop('o_dep_tag', axis=1).drop('ni_dep_tag', axis=1)
            x = np.array(df, dtype=np.float32)
            x = np.vstack((x, zeros))[:30].reshape(-1)
            x_dataset.append(x)
            y_dataset.append(domain)

    y_dataset = le.transform(y_dataset)
    return x_dataset, y_dataset

def one_versus_the_rest(x_dataset, y_dataset, args):
    train_x, test_x, train_y, test_y = train_test_split(x_dataset, y_dataset)

    estimator = svm.SVC(C=args.c, kernel=args.kernel, gamma=args.gamma)
    classifier = OneVsRestClassifier(estimator)
    classifier.fit(train_x, train_y)
    pred_y = classifier.predict(test_x)
    print('One-versus-the-rest: {:.5f}'.format(accuracy_score(test_y, pred_y)))

def one_versus_the_one(x_dataset, y_dataset, args):
    train_x, test_x, train_y, test_y = train_test_split(x_dataset, y_dataset)

    classifier = svm.SVC(C=args.c, kernel=args.kernel, gamma=args.gamma)
    classifier.fit(train_x, train_y)
    pred_y = classifier.predict(test_x)
    print('One-versus-one: {:.5f}'.format(accuracy_score(test_y, pred_y)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', '-c', type=int, default=1)
    parser.add_argument('--kernel', '-k', type=str, default='rbf')
    parser.add_argument('--gamma', '-g', type=float, default=1e-2)
    parser.add_argument('--df_path', default='')
    args = parser.parse_args()

    x_dataset, y_dataset = load_dataset(args.df_path)
    one_versus_the_rest(x_dataset, y_dataset, args)
    one_versus_the_one(x_dataset, y_dataset, args)

if __name__ == "__main__":
    main()
