import numpy as np
import pandas as pd
import gensim
import re
from joblib import Parallel, delayed
import os
from collections import defaultdict
import chainer.links as L
import pickle

research_path = '../../data/'
w2v_path = research_path + 'entity_vector/entity_vector.model.txt'
directory = research_path + 'annotated/'
domain_dict = {'OC':'Yahoo!知恵袋','OW':'白書','OY':'Yahoo!ブログ',
    'PB':'書籍','PM':'雑誌','PN':'新聞'}

#正規表現
def get_tag_id(text):
    m = re.search(r'id="([0-9]+)"', text)
    if m: return m.group(1)
    return ''
def get_ga_tag(text):
    m = re.search(r'ga="(.+?)"', text)
    if m: return m.group(1)
    return ''
def get_o_tag(text):
    m = re.search(r'o="(.+?)"', text)
    if m: return m.group(1)
    return ''
def get_ni_tag(text):
    m = re.search(r' ni="(.+?)"', text)
    if m: return m.group(1)
    return ''
def get_ga_dep_tag(text):
    m = re.search(r'ga_dep="(.+?)"', text)
    if m: return m.group(1)
    return None
def get_o_dep_tag(text):
    m = re.search(r'o_dep="(.+?)"', text)
    if m: return m.group(1)
    return None
def get_ni_dep_tag(text):
    m = re.search(r' ni_dep="(.+?)"', text)
    if m: return m.group(1)
    return None
def is_num(text):
    m = re.match('\A[0-9]+\Z', text)
    if m: return True
    else: return False

class Word2Vec:
    def __init__(self, model_file_path):
        model = gensim.models.KeyedVectors.load_word2vec_format(model_file_path)
        self.model = model

    def word_to_vector(self, word):
        if word and word in self.model.vocab:
            return self.model[word]
        else:
            return np.zeros(200, dtype=np.float32)

    def word_to_dataframe(self, word):
        vector = self.word_to_vector(word)
        df = pd.DataFrame([vector])
        df.columns = [f'word2vec:{i}' for i in range(200)]
        df['word'] = word
        return df

def file_to_dataframe_list(file_path):
    df_list = []
    print(file_path)
    for sentence in load_file(file_path):
        for df in sentence_find_verb(sentence):
            df['file_path'] = file_path
            df_list.append(df)
    return df_list

def load_file(file_path):
    sentence = ''
    with open(file_path) as f:
        for line in f:
            if line[0] == '#':
                continue
            if line.strip() == 'EOS':
                yield sentence.strip()
                sentence = ''
            else:
                sentence += line

def check_id_in_sentence(sentence, case_id):
    for line in sentence.split('\n'):
        tag_id = get_tag_id(line)
        if tag_id == case_id:
            return True
    return False

def sentence_find_verb(sentence):
    """
    動詞，形容詞，サ変名詞を対象
    """
    pred_word_number = 0 #何単語目に出現したか
    for i, line in enumerate(sentence.split('\n')):
        if line[0] != '*':
            pred_word_number += 1
        if '\t動詞' in line or '\t形容詞' in line or 'サ変可能' in line:
            ga_case_id = get_ga_tag(line)
            o_case_id = get_o_tag(line)
            ni_case_id = get_ni_tag(line)
            if is_num(ga_case_id) and check_id_in_sentence(sentence, ga_case_id):
                pass
            else:
                ga_case_id = None
            if is_num(o_case_id) and check_id_in_sentence(sentence, o_case_id):
                pass
            else:
                o_case_id = None
            if is_num(ni_case_id) and check_id_in_sentence(sentence, ni_case_id):
                pass
            else:
                ni_case_id = None
            yield sentence_to_vector(sentence, pred_word_number, ga_case_id, o_case_id, ni_case_id)
        if line[0] == '*':
            continue

def df_drop(df):
    for i in range(17):
        df = df.drop(f"feature:{i}", axis=1)
    df = df.drop('word', axis=1)
    return df

def make_pred_context_vector(sentence, pred_number, max_pred_context_size=24):
    word_number = 0
    pred_context = []
    for i, line in enumerate(sentence.split('\n')):
        if line[0] == '*':
            if pred_number > word_number:
                pred_context = []
            else:
                break
            continue
        word_number += 1
        word, feature, tag = line.split('\t')
        pred_context.append(word)
    pred_context_vector = np.array([])
    for word in pred_context:
        pred_context_vector = np.hstack((pred_context_vector, word2vec.word_to_vector(word)))
    for i in range(len(pred_context), max_pred_context_size):
        pred_context_vector = np.hstack((pred_context_vector, np.zeros(200)))
    if len(pred_context_vector)/200 > max_pred_context_size:
        print(f'error!!! {len(pred_context_vector)}')
        print(pred_context)
    return ','.join(pred_context), pred_context_vector

def df_pred_vector(sentence, pred_number):
    word_number = 0
    pred = ''
    for i, line in enumerate(sentence.split('\n')):
        if line[0] == '*':
            continue
        else:
            word_number += 1
            word, feature, tag = line.split('\t')
            if word_number == pred_number:
                pred = word
    pred_vector = word2vec.word_to_vector(pred)
    pred_context, pred_context_vector = make_pred_context_vector(sentence, pred_number)

    df_pred_vector = pd.DataFrame([pred_vector])
    df_pred_vector.columns = [f'pred_vec:{i}' for i in range(200)]
    df_pred_vector['pred'] = pred
    df_pred_context_vector = pd.DataFrame([pred_context_vector])
    df_pred_context_vector.columns = [f'pred_context_vec:{i}' for i in range(len(pred_context_vector))]
    df_pred_context_vector['pred_context'] = pred_context
    df_pred = pd.merge(df_pred_vector, df_pred_context_vector, left_index=True, right_index=True, how='outer')
    return df_pred

def sentence_to_vector(sentence, pred_number, ga_case_id, o_case_id, ni_case_id):
    print(type(ga_case_id))
    word_number = 0 #何単語目に出現したか
    df = pd.DataFrame()
    df_pred = df_pred_vector(sentence, pred_number)
    for i, line in enumerate(sentence.split('\n')):
        if line[0] == '*':
            continue
        word_number += 1
        word, feature, tag = line.split('\t')
        df_word = word2vec.word_to_dataframe(word)
        df_ = pd.merge(df_word, df_pred, left_index=True, right_index=True, how='outer')

        #正解ラベルか確認して，正解を入れる．
        tag_id = get_tag_id(tag)
        print(word, tag_id)
        print(type(tag_id))
        if ga_case_id == tag_id:
            df_['ga_case'] = 1
        else:
            df_['ga_case'] = 0
        if o_case_id == tag_id:
            df_['o_case'] = 1
        else:
            df_['o_case'] = 0
        if ni_case_id == tag_id:
            df_['ni_case'] = 1
        else:
            df_['ni_case'] = 0

        df = pd.concat([df, df_], ignore_index=True)
    df = df.fillna(0)
    return df

def reduction_dataframe(df_list):
    reduction_df_list = []
    for df in df_list:
      df = df.fillna(0)
      if df.iloc[0]['ga_case'] == 1 and df.iloc[0]['o_case'] == 1 and df.iloc[0]['ni_case'] == 1:
        "ガ格，ヲ格，ニ格がいずれもないものは，対象としない"
        continue
      reduction_df_list.append(df)
    return reduction_df_list

def main():
    for domain in domain_dict:
        print(f'start {domain}')
        r = Parallel(n_jobs=-1)([delayed(file_to_dataframe_list)(f'{directory}{domain}/{file}') for file in os.listdir(f'{directory}{domain}/')])
        dataset = []
        for df_list in r:
            dataset += reduction_dataframe(df_list)
        with open(f'./dataframe/dataframe_list_{domain}.pickle', 'wb') as f:
            pickle.dump(dataset, f)
        del r
        del dataset

if __name__=='__main__':
    word2vec = Word2Vec(w2v_path)
    main()
