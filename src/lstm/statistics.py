from collections import defaultdict
import pickle
import math

import numpy as np
import pandas as pd

from train import load_dataset

domain_dict = {'OC':'Yahoo!知恵袋', 'OW':'白書', 'OY':'Yahoo!ブログ', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def main():
    all_ga_count = defaultdict(int)
    all_o_count = defaultdict(int)
    all_ni_count = defaultdict(int)
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        with open('./dataframe/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
            df_list = pickle.load(f)
        ga_count = defaultdict(int)
        o_count = defaultdict(int)
        ni_count = defaultdict(int)
        ga_dep_count = 0
        o_dep_count = 0
        ni_dep_count = 0
        for df in df_list:
            y_ga = np.array(df['ga_case'], dtype=np.int32)
            y_o = np.array(df['o_case'], dtype=np.int32)
            y_ni = np.array(df['ni_case'], dtype=np.int32)
            if df['ga_dep_tag'].any() == 'dep':
                ga_dep_count += 1
            if df['o_dep_tag'].any() == 'dep':
                o_dep_count += 1
            if df['ni_dep_tag'].any() == 'dep':
                ni_dep_count += 1
            if y_ga.argmax() > 3:
                ga_count[4] += 1
            else:
                ga_count[y_ga.argmax()] += 1
            if y_o.argmax() > 3:
                o_count[4] += 1
            else:
                o_count[y_o.argmax()] += 1
            if y_ni.argmax() > 3:
                ni_count[4] += 1
            else:
                ni_count[y_ni.argmax()] += 1
        
        print(f'{domain}|ga|' +'|'.join([str(key) for key in range(0, 5)]))
        print(f'{domain}|合計数|' +'|'.join([str(ga_count[key]) for key in range(0, 5)]))
        print(f'{domain}|o|' +'|'.join([str(key) for key in range(0, 5)]))
        print(f'{domain}|合計数|' +'|'.join([str(o_count[key]) for key in range(0, 5)]))
        print(f'{domain}|ni|' +'|'.join([str(key) for key in range(0, 5)]))
        print(f'{domain}|合計数|' +'|'.join([str(ni_count[key]) for key in range(0, 5)]))
        print(f'{domain}|ga_dep_tag|{ga_dep_count}|o_dep_tag|{o_dep_count}|ni_dep_tag|{ni_dep_count}')
        for i in range(0, 5):
            all_ga_count[i] += ga_count[i]
            all_o_count[i] += o_count[i]
            all_ni_count[i] += ni_count[i]

    print(f'all|ga|' + '|'.join([str(key) for key in range(0, 5)]))
    print(f'all|合計数|' + '|'.join([str(all_ga_count[key]) for key in range(0, 5)]))
    print(f'all|o|' + '|'.join([str(key) for key in range(0, 5)]))
    print(f'all|合計数|' + '|'.join([str(all_o_count[key]) for key in range(0, 5)]))
    print(f'all|ni|' + '|'.join([str(key) for key in range(0, 5)]))
    print(f'all|合計数|' + '|'.join([str(all_ni_count[key]) for key in range(0, 5)]))

def dataset_size():
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        with open('./dataframe/dataframe_list_{0}.pickle'.format(domain), 'rb') as f:
            df_list = pickle.load(f)
        size = math.ceil(len(df_list)*0.8)
        train_x_size = len(df_list[:size])
        test_x_size = len(df_list[size:])
        print('domain:{0}, train_size:{1}, test_size{2}'.format(domain, train_x_size, test_x_size))

if __name__=='__main__':
    # main()
    dataset_size()

