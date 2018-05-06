import pickle
import math
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

import numpy as np
import pandas as pd
from collections import defaultdict
import math

domain_dict = {'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'OW':'白書', 'PB':'書籍','PM':'雑誌','PN':'新聞'}
domain_color_dict = {'OC':'#FEA47F', 'OY':'#25CCF7', 'OW':'#EAB543', 'PB':'#55E6C1', 'PM':'#CAD3C8', 'PN':'#D6A2E8'}

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
        calc['文内'] += 1
    calc['述語'] += 1
        # if verb_index == case_index:
        #     calc[verb_index - case_index] += 1
    return calc

def sum_calc(df_list):
    sum_calc_dict = {'ga':defaultdict(int), 'o':defaultdict(int), 'ni':defaultdict(int)}
    for df in df_list:
        for case in ['ga', 'o', 'ni']:
            tmp_calc_dict  = calc_distance(df, case)
            for key in tmp_calc_dict:
                sum_calc_dict[case][key] += tmp_calc_dict[key]
    return sum_calc_dict

def main():
    df_path = '../lstm/original/dataframe'
    domain_calc_dict = {}
    for domain in domain_dict:
        df_list = load_dataset(df_path, domain)
        domain_calc_dict[domain] = sum_calc(df_list)
        print(len(df_list))
        del df_list
    return domain_calc_dict

def dump(dump_path, domain_calc_dict):
    min_dis = -100
    max_dis = 200
    with open(dump_path, 'w') as f:
        for domain in domain_dict:
            f.write(domain)
            f.write('\nindex\tga\to\tni\n')
            for i in range(min_dis-1, max_dis+1):
                f.write(str(i)+'\t')
                f.write(str(domain_calc_dict[domain]['ga'][i])+'\t')
                f.write(str(domain_calc_dict[domain]['o'][i])+'\t')
                f.write(str(domain_calc_dict[domain]['ni'][i])+'\n')
            f.write(f'\nNone\t{domain_calc_dict[domain]["ga"]["None"]}\t{domain_calc_dict[domain]["o"]["None"]}\t{domain_calc_dict[domain]["ni"]["None"]}\n')
            f.write(f'\nexo1\t{domain_calc_dict[domain]["ga"]["exo1"]}\t{domain_calc_dict[domain]["o"]["exo1"]}\t{domain_calc_dict[domain]["ni"]["exo1"]}\n')
            f.write(f'\nexo2\t{domain_calc_dict[domain]["ga"]["exo2"]}\t{domain_calc_dict[domain]["o"]["exo2"]}\t{domain_calc_dict[domain]["ni"]["exo2"]}\n')
            f.write(f'\nexog\t{domain_calc_dict[domain]["ga"]["exog"]}\t{domain_calc_dict[domain]["o"]["exog"]}\t{domain_calc_dict[domain]["ni"]["exog"]}\n')

def plot_graph(dump_path, domain_calc_dict):
    min_dis = -100
    max_dis = 200
    for case in ['ga', 'o', 'ni']:
        for domain in domain_dict:
            sum_count = 0
            x = []
            y = []
            for i in domain_calc_dict[domain][case]:
                if i == 'None':
                    continue
                sum_count += domain_calc_dict[domain][case][i]
            for i in range(min_dis-1, max_dis+1):
                if i>0:
                    x.append(math.sqrt(abs(i)))
                else:
                    x.append(math.sqrt(abs(i))*-1)
                y.append(domain_calc_dict[domain][case][i]/sum_count*100)

            x = np.array(x)
            y = np.array(y)
            plt.plot(x, y, '--o', markersize = 3, linewidth = 0.8,label=domain, color=domain_color_dict[domain], markeredgecolor=domain_color_dict[domain])
            x = []
            y = []

            # x.append(9)
            x.append(10)
            x.append(11)
            x.append(12)
            # y.append(domain_calc_dict[domain][case]["None"]/sum_count*100)
            y.append(domain_calc_dict[domain][case]["exo1"]/sum_count*100)
            y.append(domain_calc_dict[domain][case]["exo2"]/sum_count*100)
            y.append(domain_calc_dict[domain][case]["exog"]/sum_count*100)

            plt.plot(x, y, 'o', markersize = 3, color=domain_color_dict[domain], markeredgecolor=domain_color_dict[domain])

        domain = 'union'
        plt.axvline(x=0, color='black')
        plt.xlabel('距離(sqrt)')
        plt.ylabel('出現確率(%)')
        plt.tick_params(labelsize = 15)
        plt.grid(which='major',color='black',linestyle='--')
        plt.title(f"{domain}-{case}", loc='center')
        plt.axis([-7, 13, 0, 50])
        plt.legend()
        plt.savefig(f"{dump_path}/{domain}-{case}.png", dpi=500)
        plt.cla()
        plt.clf()

    plot_graph('graph', domain_calc_dict)

def circle_plot_graph(dump_path, domain_calc_dict):
    min_dis = -100
    max_dis = 200
    for case in ['ga', 'o', 'ni']:
        for domain in domain_dict:
            x = []
            y = []
            intra = 0
            for key in domain_calc_dict[domain][case].keys():
                if type(key) == np.int64:
                    intra += abs(domain_calc_dict[domain][case][key])
                else:
                    continue
            none = domain_calc_dict[domain][case]['None']
            exo1 = domain_calc_dict[domain][case]['exo1']
            exo2 = domain_calc_dict[domain][case]['exo2']
            exog = domain_calc_dict[domain][case]['exog']
            x = np.array([none, intra, exo1, exo2, exog])
            label = ['none', 'intra', '発信者', '受信者', '文間+外界']

            plt.pie(x, labels=label, startangle=90, colors=list(domain_color_dict.values()))

            plt.tick_params(labelsize = 15)
            plt.title(f"{domain}-{case}", loc='center')
            plt.savefig(f"{dump_path}/{domain}-{case}.png", dpi=500)
            plt.cla()
            plt.clf()
    circle_plot_graph('graph', domain_calc_dict)
