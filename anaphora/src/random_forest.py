import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import os
import re

directory = '/Users/sango.m.ab/Desktop/research/data/annotated/'
foldas = ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']

#正規表現
def is_num(text):
    if text == None:
        return None
    return re.match('\A[0-9]+\Z', text)
def get_num_from_id(text):
    m = re.search(r'id="([0-9]+)"', text)
    if m:
        return m.group(1)
    return None
def ga(text):
    m = re.search(r'ga="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def o(text):
    m = re.search(r'o="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def ni(text):
    m = re.search(r' ni="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def ga_dep(text):
    m = re.search(r'ga_dep="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def o_dep(text):
    m = re.search(r'o_dep="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def ni_dep(text):
    m = re.search(r' ni_dep="(.+?)"', text)
    if m:
        return m.group(1)
    return None

def load_data(file_path):
    X = []
    Y = []
    with open(file_path) as f:
        sentence = []
        for line in f:
            line = line.rstrip()
            if line == 'EOS':
                x, y = make_dataset(sentence)
                X += x
                Y += y
                sentence = []
            elif line[0] == '#' or line[0] == '*':
                pass
            else:
                sentence.append(line)
    return X, Y

def make_dataset(sentence):
    x = []
    y = []
    for one in sentence:
        word_one = one.split('\t')[0]
        feature_one = one.split('\t')[1].split(',')
        tag_one = one.split('\t')[2]
        if feature_one[0] != '名詞':
            continue
        noun_id = get_num_from_id(one)
        for two in sentence:
            word_two = two.split('\t')[0]
            feature_two = two.split('\t')[1].split(',')
            tag_two = two.split('\t')[2]
            if feature_two[0] != '名詞' and feature_two[0] != '動詞':
                continue
            if not ga_dep(two):
                continue
            x_tmp = [feature_one[1], feature_one[2], feature_one[3], feature_two[0], feature_two[1], feature_two[2], feature_two[3], feature_two[4], feature_two[5]]
            y_tmp = 0
            if ga_dep(two) == 'zero':
                key = ga(two)
                if is_num(noun_id) and is_num(key) and noun_id == key:
                    y_tmp = 1
            x.append(x_tmp)
            y.append(y_tmp)

    return x, y

def load_file():
    for folda in foldas:
        X = []
        Y = []
        print(folda)
        path = directory + folda + '/'
        for file in os.listdir(f'{path}'):
#             print(f'{path}{file}')
            x, y = load_data(path+file)
            X += x
            Y += y
        data, X, Y = make_df(X, Y)
        train(data, X, Y)
        print('-- -- -- -- -- -- -- -- --')

def make_df(X, Y):
    #アンダーサンプリング
    df = pd.DataFrame(X)
    df.columns = ['格素性1', '格素性2', '格素性3', '述語素性0', '述語素性1', '述語素性2', '述語素性3', '述語素性4', '述語素性5']
    df['label'] = Y
    sampling_size = len(df[df.label == 1])
    print ("sampling size : ", sampling_size)
    high_frequentry_data = df[df.label == 0].index
    low_frequentry_data = df[df.label == 1].index
    low_frequentry_data_sample = df.loc[low_frequentry_data]

    # 出現頻度の小さいクラスに、大きいクラスの個数を合わせてランダムにデータを抽出する
    random_indices = np.random.choice(high_frequentry_data, sampling_size, replace=False)
    high_frequentry_data_sample = df.loc[random_indices]
    pd.DataFrame(high_frequentry_data_sample)

    # データをマージする
    merged_df = pd.concat([high_frequentry_data_sample, low_frequentry_data_sample], ignore_index=True)

    data = merged_df.values
    le = LabelEncoder()
    for i in range(9):
        data[:, i] = le.fit_transform(data[:, i])
    X = df.values
    for i in range(9):
        X[:, i] = le.fit_transform(X[:, i])
    return data, X[:, :-1], Y

def train(data, X, Y):
    #学習
    data = np.asarray(data, dtype="int")
    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=0)

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    y_pred = clf_rf.predict(X)

    accu = accuracy_score(Y, y_pred)
    print('accuracy = {:>.4f}'.format(accu))

    # Feature Importance
    fti = clf_rf.feature_importances_

    print('Feature Importances:')
    for i, feat in enumerate(['格素性1', '格素性2', '格素性3', '述語素性0', '述語素性1', '述語素性2', '述語素性3', '述語素性4', '述語素性5']):
        print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))


if __name__ == '__main__':
  load_file()