from extract_feature import *
import re
import os
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

directory = '/Users/sango.m.ab/Desktop/research/data/annotated/'
foldas = ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']

#正規表現
def is_num(text):
    if text == None:
        return None
    return re.match('\A[0-9]+\Z', text)
def get_id_num(text):
    m = re.search(r'id="([0-9]+)"', text)
    if m:
        return m.group(1)
    return None
def get_ga_id(text):
    m = re.search(r'ga="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def get_o_id(text):
    m = re.search(r'o="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def get_ni_id(text):
    m = re.search(r' ni="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def get_ga_dep_tag(text):
    m = re.search(r'ga_dep="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def get_o_dep_tag(text):
    m = re.search(r'o_dep="(.+?)"', text)
    if m:
        return m.group(1)
    return None
def get_ni_dep_tag(text):
    m = re.search(r' ni_dep="(.+?)"', text)
    if m:
        return m.group(1)
    return None

def get_predicate_from_sentence(sentence):
    tag = ''
    predicate_word_position = 0
    for line in sentence.split('\n'):
        if line == '':
            continue
        if line[:2] == '* ':
            tag = line
            continue
        word, feature, annotate = line.split('\t')
        feature_list = feature.split(',')
        if feature_list[0] == '名詞' or feature_list[0] == '動詞':
            ga_dep_tag = get_ga_dep_tag(line)
            ga_id = get_ga_id(line)
            o_dep_tag = get_o_dep_tag(line)
            o_id = get_o_id(line)
            ni_dep_tag = get_ni_dep_tag(line)
            ni_id = get_ni_id(line)
            if ga_dep_tag:
                yield tag, line, predicate_word_position
        predicate_word_position += 1

def get_condidate_from_sentence(sentence):
    tag = ''
    condidate_word_position = 0
    for line in sentence.split('\n'):
        if line == '':
            continue
        if line[:2] == '* ':
            tag = line
            continue
        word, feature, annotate = line.split('\t')
        feature_list = feature.split(',')
        if feature_list[0] == '名詞':
            yield tag, line, condidate_word_position
        condidate_word_position += 1

def ga_case_make_vector_from_sentence(sentence):
    for predicate in get_predicate_from_sentence(sentence):
        predicate_tag, predicate_line, predicate_word_position = predicate
        
        for condidate in get_condidate_from_sentence(sentence):
            condidate_tag, condidate_line, condidate_word_position = condidate

            if predicate_word_position == condidate_word_position:
                continue
            vector_dict = {}
            vector_dict.update(get_predicate_lexical_infomation(predicate_line))
            vector_dict.update(get_condidate_lexical_infomation(condidate_line))
            vector_dict.update(get_condidate_is_subject_head(condidate_line, sentence))
            vector_dict.update(get_condidate_postpositional_particle_infomation(condidate_line, sentence))
            vector_dict.update(get_condidate_is_after_predicate(predicate_word_position, condidate_word_position))
            vector_dict.update(get_morpheme_distance(predicate_word_position, condidate_word_position))
            vector_dict.update(get_dependency_distance(sentence, predicate_word_position, condidate_word_position))
            # vector_dict.update(get_case_frame(predicate_line.split('\t')[0], True))
            # vector_dict.update(get_case_frame(condidate_line.split('\t')[0], False))
            vector_dict.update({'correct':check_correct_ga_dep(predicate_line, condidate_line)})
            
            yield vector_dict

def check_correct_ga_dep(predicate_line, condidate_line):
    if get_ga_dep_tag(predicate_line) == 'zero':
        ga_id = get_ga_id(predicate_line)
        condidate_id = get_num_from_id(condidate_line)
        if ga_id == condidate_id:
            return True
    return False

def load_file(file_path):
    global vector_dict_list
    sentence = ''
    with open(file_path) as f:
        for line in f:
            if line[0] == '#':
                continue
            if line.rstrip() == 'EOS':
                for vector_dict in ga_case_make_vector_from_sentence(sentence):
                    #初期化
                    if vector_dict_list == {}:
                        for key in vector_dict:
                            vector_dict_list[key] = []

                    for key in vector_dict:
                        vector_dict_list[key].append(vector_dict[key])
                sentence = ''
            else:
                sentence += line

def load_folda(folda_path):
    # r = Parallel(n_jobs=-1)( [delayed(load_file)(folda_path + file) for file in os.listdir(folda_path)] )
    for file in os.listdir(folda_path):
        file_path = folda_path + file
        print(file_path)
        load_file(file_path)

def load():
    global vector_dict_list
    data_set_dict = {}
    vector_dict_list = {}
    for folda in foldas:
        folda_path = directory + folda + '/'
        print(folda)
        load_folda(folda_path)
        print(len(vector_dict_list))
        data_set_dict[folda] = vector_dict_list
        print('-----')
        vector_dict_list = {}
    return data_set_dict

def make_dataframe(vector_dict_list, sampling_size=None):
    df = pd.DataFrame.from_dict(vector_dict_list)
    if sampling_size == None:
        sampling_size = len(df[df['correct']])
    # print ("sampling size : ", sampling_size)
    high_frequentry_data = df[df['correct'] == False].index
    low_frequentry_data = df[df['correct']].index
    low_frequentry_data_sample = df.loc[low_frequentry_data]

    # 出現頻度の小さいクラスに、大きいクラスの個数を合わせてランダムにデータを抽出する
    random_indices = np.random.choice(high_frequentry_data, sampling_size, replace=False)
    high_frequentry_data_sample = df.loc[random_indices]

    # データをマージする
    merged_df = pd.concat([high_frequentry_data_sample, low_frequentry_data_sample], ignore_index=True)

    Y = merged_df['correct'].values
    X = merged_df.drop("correct", axis=1).values
    feature_list = merged_df.drop("correct", axis=1).keys()

    le = LabelEncoder()
    for i in range(X.shape[1]):
        X[:, i] = le.fit_transform(X[:, i])
    Y =  le.fit_transform(Y)

    return X, Y, feature_list

def train(X_train, X_test, Y_train, Y_test, feature_list, is_feature_importance):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, Y_train)
    Y_pred = clf_rf.predict(X_test)

    accu = accuracy_score(Y_test, Y_pred)
    print('{:>.4f}'.format(accu), end='|')

def check_feature_importance(X, Y, feature_list):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X, Y)

    # Feature Importance
    fti = clf_rf.feature_importances_

    print('|', end='')
    for i, feat in enumerate(feature_list):
        print('\t{0}'.format(feat), end='|')
    print('\n|', end='')
    for i, feat in enumerate(feature_list):
        print('{0:>.6f}'.format(fti[i]), end='|')
    print()

if __name__ == '__main__':
    data_set_dict = load()
    print("feature_importances")
    for train_type in data_set_dict:
        print(train_type)
        X, Y, feature_list = make_dataframe(data_set_dict[train_type])
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        check_feature_importance(X, Y, feature_list)

    print("クロスドメイン データ量最大")
    print('||', end='')
    for train_type in data_set_dict:
        print(train_type, end='|')
    print('\n|---|---|---|---|---|---|---|')
    for train_type in data_set_dict:
        print('|'+train_type, end='|')
        X_train, Y_train, feature_list = make_dataframe(data_set_dict[train_type])
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
        for test_type in data_set_dict:
            if train_type == test_type:
                pass
            else:
                X_test, Y_test, feature_list = make_dataframe(data_set_dict[test_type])
                _, X_test, _, Y_test = train_test_split(X_test, Y_test, test_size=0.2, random_state=0)
            train(X_train, X_test, Y_train, Y_test, feature_list, False)
        print()

    print("クロスドメイン データ量揃えて(2078)")
    sampling_size = 1039
    print('||', end='')
    for train_type in data_set_dict:
        print(train_type, end='|')
    print('\n|---|---|---|---|---|---|---|')
    for train_type in data_set_dict:
        print('|'+train_type, end='|')
        X_train, Y_train, feature_list = make_dataframe(data_set_dict[train_type], sampling_size)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
        for test_type in data_set_dict:
            if train_type == test_type:
                pass
            else:
                X_test, Y_test, feature_list = make_dataframe(data_set_dict[test_type], sampling_size)
                _, X_test, _, Y_test = train_test_split(X_test, Y_test, test_size=0.2, random_state=0)
            train(X_train, X_test, Y_train, Y_test, feature_list, False)
        print()