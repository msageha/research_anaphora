import numpy as np
import pandas as pd
import re
from joblib import Parallel, delayed
import os
from collections import defaultdict
import chainer.links as L
import pickle
import random

research_path = '../../../data/'
w2v_path = research_path + 'entity_vector/entity_vector.model.txt'
directory = research_path + 'annotated/'
domain_dict = {'OW':'白書', 'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'PB':'書籍'}

tsubame = False
if tsubame == True:
    w2v_path = research_path + 'entity_vector/entity_vector.model.pickle'
else:
    import gensim
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
        if tsubame:
            with open(model_file_path, 'rb') as f:
                model = pickle.load(f)
            words = model.keys()
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(model_file_path)
            words = model.vocab.keys()
        self.model = model
        self.words = words

    def word_to_vector(self, word):
        if word and word in self.words:
            return self.model[word]
        else:
            return np.zeros(200, dtype=np.float32)

    def word_to_dataframe(self, word):
        vector = self.word_to_vector(word)
        df = pd.DataFrame([vector])
        df.columns = ['word2vec:{0}'.format(i) for i in range(200)]
        df['word'] = word
        return df

def file_to_dataframe_list(file_path):
    df_list = []
    print(file_path, flush=True)
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
        df = df.drop('feature:{0}'.format(i), axis=1)
    df = df.drop('word', axis=1)
    return df

def df_pred_vector(sentence, pred_number):
    word_number = 0
    pred_prev = ''
    pred = ''
    pred_next = ''
    for i, line in enumerate(sentence.split('\n')):
        if line[0] == '*':
            continue
        else:
            word_number += 1
            word, feature, tag = line.split('\t')
            if word_number == pred_number - 1:
                pred_prev = word
            elif word_number == pred_number:
                pred = word
            elif word_number == pred_number + 1:
                pred_next = word
    pred_prev_vector = word2vec.word_to_vector(pred_prev)
    pred_vector = word2vec.word_to_vector(pred)
    pred_next_vector = word2vec.word_to_vector(pred_next)
    pred_context_vector = np.hstack((pred_prev_vector, pred_vector, pred_next_vector))
    df_pred_context_vector = pd.DataFrame([pred_context_vector])
    df_pred_context_vector.columns = ['pred_context_vec:{0}'.format(i) for i in range(600)]
    df_pred_context_vector['pred_prev'] = pred_prev
    df_pred_context_vector['pred'] = pred
    df_pred_context_vector['pred_next'] = pred_next
    df_pred_vector = pd.DataFrame([pred_vector])
    df_pred_vector.columns = ['pred_vec:{0}'.format(i) for i in range(200)]
    df_pred = pd.merge(df_pred_vector, df_pred_context_vector, left_index=True, right_index=True, how='outer')
    return df_pred

def sentence_to_vector(sentence, pred_number, ga_case_id, o_case_id, ni_case_id):
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

        if pred_number-1 <= word_number and word_number <= pred_number+1:
            df_['marked'] = 1
        else:
            df_['marked'] = 0

        if pred_number == word_number:
            df_['pred'] = 1
        else:
            df_['pred'] = 0
        #正解ラベルか確認して，正解を入れる．
        tag_id = get_tag_id(tag)
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
        if df['ga_case'].max() == 0 and df['o_case'].max() == 0 and df['ni_case'].max() == 0:
            "ガ格，ヲ格，ニ格がいずれもないものは，対象としない.ただし，1/10だけデータに入れる．"
            if random.randint(0, 9) == 0:
                reduction_df_list.append(df)
        else:
            reduction_df_list.append(df)
    return reduction_df_list

def main():
    for domain in domain_dict:
        print('start {}'.format(domain))
        r = Parallel(n_jobs=-1)([delayed(file_to_dataframe_list)('{0}{1}/{2}'.format(directory, domain, file)) for file in os.listdir('{0}{1}/'.format(directory, domain))])
        dataset = []
        for df_list in r:
            
            dataset += df_list
        del r
        with open('./dataframe_short/dataframe_list_{}.pickle'.format(domain), 'wb') as f:
            pickle.dump(dataset, f)
        del dataset

if __name__=='__main__':
    word2vec = Word2Vec(w2v_path)
    main()
