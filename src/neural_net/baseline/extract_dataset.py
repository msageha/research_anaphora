import pandas as pd
import pickle
import numpy as np
from collections import OrderedDict

domain_dict = OrderedDict([('OC', 'Yahoo!知恵袋'), ('OY', 'Yahoo!ブログ'), ('OW', '白書'), ('PB', '書籍'), ('PM', '雑誌'), ('PN', '新聞')])

def make_dataset(df_path):
    dataset_dict = {}
    for domain in domain_dict:
        print('start data load domain-{0}'.format(domain))
        with open('{0}/dataframe_list_{1}.pickle'.format(df_path, domain), 'rb') as f:
            df_list = pickle.load(f)
        x_dataset = []
        y_ga_dataset = []
        y_ga_dep_tag_dataset = []
        y_o_dataset = []
        y_o_dep_tag_dataset = []
        y_ni_dataset = []
        y_ni_dep_tag_dataset = []
        for df in df_list:
            y_ga = np.array(df['ga_case'], dtype=np.int32)
            y_o = np.array(df['o_case'], dtype=np.int32)
            y_ni = np.array(df['ni_case'], dtype=np.int32)
            y_ga_dep_tag = np.array(df['ga_dep_tag'])
            y_o_dep_tag = np.array(df['o_dep_tag'])
            y_ni_dep_tag = np.array(df['ni_dep_tag'])
            df = df.drop('ga_case', axis=1).drop('o_case', axis=1).drop('ni_case', axis=1).drop('ga_dep_tag', axis=1).drop('o_dep_tag', axis=1).drop('ni_dep_tag', axis=1)
            x = np.array(df, dtype=np.float32)
            x_dataset.append(x)
            y_ga_dataset.append(y_ga)
            y_ga_dep_tag_dataset.append(y_ga_dep_tag)
            y_o_dataset.append(y_o)
            y_o_dep_tag_dataset.append(y_o_dep_tag)
            y_ni_dataset.append(y_ni)
            y_ni_dep_tag_dataset.append(y_ni_dep_tag)
        np.savez('dataset/{0}.npz'.format(domain), x=x_dataset, y_ga=y_ga_dataset, y_o=y_o_dataset, y_ni=y_ni_dataset, 
            y_ga_dep_tag=y_ga_dep_tag_dataset, y_o_dep_tag=y_o_dep_tag_dataset, y_ni_dep_tag=y_ni_dep_tag_dataset)
        # dataset_dict['{0}_x'.format(domain)] = x_dataset
        # dataset_dict['{0}_y_ga'.format(domain)] = y_ga_dataset
        # dataset_dict['{0}_y_o'.format(domain)] = y_o_dataset
        # dataset_dict['{0}_y_ni'.format(domain)] = y_ni_dataset
        # dataset_dict['{0}_y_ga_dep_tag'.format(domain)] = y_ga_dep_tag_dataset
        # dataset_dict['{0}_y_o_dep_tag'.format(domain)] = y_o_dep_tag_dataset
        # dataset_dict['{0}_y_ni_dep_tag'.format(domain)] = y_ni_dep_tag_dataset
    # return dataset_dict


if __name__ == '__main__':
    df_path = '../dataframe'
    make_dataset(df_path)