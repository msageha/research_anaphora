import pickle
import math
import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from collections import defaultdict

domain_dict = {'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}

def load_dataset(df_path, domain):
    print('start data load domain-{0}'.format(domain))
    with open('{0}/dataframe_list_{1}.pickle'.format(df_path, domain), 'rb') as f:
        df_list = pickle.load(f)
    return df_list

def calc_distance(df, case):
    calc = defaultdict(int)
    verb_index = np.array(df.is_verb).argmax()
    case_index = np.array(df['{}_case'.format(case)]).argmax()
    if case_index == 0:
        calc['None'] += 1
    elif case_index == 1:
        calc['exo1'] += 1
    elif case_index == 2:
        calc['exo2'] += 1
    elif case_index == 3:
        calc['exog'] += 1
    else:
        calc[verb_index - calc] += 1
    return calc

def sum_calc(df_list):
    sum_calc_dict = {'ga':defaultdict(int), 'o':defaultdict(int), 'ni':defaultdict(int)}
    for df in df_list:
        for case in ['ga', 'o', 'ni']:
            tmp_calc_dict  = calc_distance(df, case)
            for key in tmp_calc_dict:
                sum_calc_dict[key] += tmp_calc_dict[key]
    return sum_calc_dict

def main():
    df_list = '../lstm/original/dataframe'
    domain_dict = {}
    for domain in domain_dict:
        df_list = load_dataset(df_path, domain)
        domain_dict[domain] = sum_calc(df_list)
        del df_list


