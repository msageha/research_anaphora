import numpy as np
import pandas as pd
import gensim
import re
from joblib import Parallel, delayed
import os
from collections import defaultdict
import chainer.links as L
import pickle

w2v_path = '/gs/hs0/tga-cl/sango-m-ab/research/data/entity_vector/entity_vector.model.txt'
directory = '/gs/hs0/tga-cl/sango-m-ab/research/data/annotated/'
domain_dict = {'OC':'Yahoo!知恵袋','OW':'白書','OY':'Yahoo!ブログ',
    'PB':'書籍','PM':'雑誌','PN':'新聞'}

#正規表現
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
        if word == 'exo1':
            vector = self.word_to_vector('僕')
        elif word == 'exo2':
            vector = self.word_to_vector('おまえ')
        elif word == 'exog':
            vector = self.word_to_vector('これ')
        df = pd.DataFrame([vector])
        df.columns = [f'word2vec:{i}' for i in range(200)]
        df['word'] = word
        return df

class FeatureToEmbedID:
    def __init__(self):
        feature_size_dict = {"feature:0":24, "feature:1":25, "feature:2":11, "feature:3":5, "feature:4":93,
          "feature:5":31, "feature:6":30119, "feature:7":35418, "feature:8":1,
          "feature:9":1, "feature:10":5545, "feature:11":1, "feature:12":7,
          "feature:13":1, "feature:14":5, "feature:15":1, "feature:16":1 }

        self.feature0 = {'*': 0, "助詞":1, "未知語":2, "URL":3, "言いよどみ":4, "連体詞":5, "ローマ字文":6, "web誤脱":7,
          "英単語":8, "接頭辞":9, "助動詞":10, "接尾辞":11, "記号":12, "動詞":13, "漢文":14, "副詞":15, "形容詞":16,
          "接続詞":17, "補助記号":18, "代名詞":19, "名詞":20, "形状詞":21, "空白":22, "感動詞":23}

        self.feature1 = {"*":0, "ＡＡ":1, "形状詞的":2, "一般":3, "括弧閉":4, "終助詞":5, "フィラー":6, "係助詞":7, "句点":8,
          "普通名詞":9, "数詞":10, "固有名詞":11, "準体助詞":12, "タリ":13, "括弧開":14, "読点":15, "形容詞的":16,
          "動詞的":17, "名詞的":18, "格助詞":19, "接続助詞":20, "助動詞語幹":21, "非自立可能":22, "文字":23, "副助詞":24}

        self.feature2 = {"*":0, "助数詞可能":1, "一般":2, "副詞可能":3, "人名":4, "サ変形状詞可能":5, "顔文字":6,
          "助数詞":7, "地名":8, "サ変可能":9, "形状詞可能":10}

        self.feature3 = {"*":0, "国":1, "名":2, "姓":3, "一般":4}

        self.feature4 = {"*":0, "サ行変格":1, "文語助動詞-ヌ":2, "文語下二段-サ行":3, "文語下二段-ラ行":4, "下一段-バ行":5,
          "下一段-サ行":6, "文語四段-タ行":7, "助動詞-ヌ":8, "文語サ行変格":9, "下一段-ザ行":10, "文語助動詞-タリ-完了":11,
          "文語助動詞-ゴトシ":12, "下一段-カ行":13, "助動詞-レル":14, "文語助動詞-ナリ-断定":15, "文語ラ行変格":16,
          "文語四段-ハ行":17, "下一段-ガ行":18, "形容詞":19, "五段-バ行":20, "下一段-ナ行":21, "助動詞-ラシイ":22,
          "文語助動詞-ズ":23, "助動詞-ナイ":24, "五段-サ行":25, "五段-タ行":26, "文語助動詞-ケリ":27, "助動詞-ダ":28,
          "文語上一段-ナ行":29, "文語四段-マ行":30, "上一段-マ行":31, "文語下二段-ダ行":32, "文語助動詞-キ":33,
          "文語上一段-マ行":34, "文語助動詞-ベシ":35, "文語助動詞-ナリ-伝聞":36, "助動詞-ナンダ":37, "上一段-バ行":38,
          "助動詞-ジャ":39, "文語形容詞-ク":40, "文語上二段-ダ行":41, "文語下二段-タ行":42, "文語助動詞-タリ-断定":43,
          "文語下二段-ハ行":44, "文語四段-ガ行":45, "文語下二段-マ行":46, "文語助動詞-リ":47, "無変化型":48, "助動詞-ヘン":49,
          "文語下二段-ナ行":50, "上一段-ア行":51, "上一段-ガ行":52, "助動詞-デス":53, "五段-カ行":54, "助動詞-タ":55,
          "上一段-ザ行":56, "助動詞-タイ":57, "カ行変格":58, "五段-ガ行":59, "五段-ナ行":60, "文語上二段-バ行":61,
          "助動詞-ヤス":62, "五段-ワア行":63, "上一段-ラ行":64, "文語助動詞-ム":65, "上一段-ナ行":66, "五段-マ行":67,
          "文語形容詞-シク":68, "五段-ラ行":69, "文語四段-ラ行":70, "下一段-ラ行":71, "文語四段-サ行":72, "文語四段-カ行":73,
          "文語助動詞-ラシ":74, "助動詞-ヤ":75, "文語下一段-カ行":76, "助動詞-マイ":77, "文語下二段-ガ行":78, "助動詞-マス":79,
          "文語助動詞-マジ":80, "文語カ行変格":81, "下一段-タ行":82, "下一段-ダ行":83, "上一段-カ行":84, "文語上二段-ハ行":85,
          "下一段-ハ行":86, "文語助動詞-ジ":87, "上一段-タ行":88, "下一段-マ行":89, "文語下二段-カ行":90, "文語下二段-ア行":91,
          "下一段-ア行":92}

        self.feature5 = {"*":0, "連用形-イ音便":1, "連体形-撥音便":2, "連用形-一般":3, "語幹-一般":4, "ク語法":5, "終止形-融合":6,
          "未然形-サ":7, "終止形-一般":8, "語幹-サ":9, "已然形-一般":10, "未然形-撥音便":11, "仮定形-一般":12, "連体形-一般":13,
          "連体形-省略":14, "未然形-補助":15, "連用形-ニ":16, "仮定形-融合":17, "終止形-促音便":18, "終止形-ウ音便":19,
          "未然形-一般":20, "連用形-促音便":21, "終止形-撥音便":22, "未然形-セ":23, "意志推量形":24, "命令形":25, "連用形-省略":26,
          "連用形-撥音便":27, "連用形-ウ音便":28, "連体形-補助":29, "連用形-融合":30}

        self.em_fe0 = L.EmbedID(24, 5)
        self.em_fe1 = L.EmbedID(25, 5)
        self.em_fe2 = L.EmbedID(11, 5)
        self.em_fe3 = L.EmbedID(5, 5)
        self.em_fe4 = L.EmbedID(93, 5)
        self.em_fe5 = L.EmbedID(31, 5)

    def feature_to_df_by_embed(self, feature):
        feature0_id = self.feature0[feature[0]]
        feature1_id = self.feature1[feature[1]]
        feature2_id = self.feature2[feature[2]]
        feature3_id = self.feature3[feature[3]]
        feature4_id = self.feature4[feature[4]]
        feature5_id = self.feature5[feature[5]]
        feature_vec0 = self.em_fe0(np.array([feature0_id], dtype=np.int32)).data[0]
        feature_vec1 = self.em_fe1(np.array([feature1_id], dtype=np.int32)).data[0]
        feature_vec2 = self.em_fe2(np.array([feature2_id], dtype=np.int32)).data[0]
        feature_vec3 = self.em_fe3(np.array([feature3_id], dtype=np.int32)).data[0]
        feature_vec4 = self.em_fe4(np.array([feature4_id], dtype=np.int32)).data[0]
        feature_vec5 = self.em_fe5(np.array([feature5_id], dtype=np.int32)).data[0]

        feature_vec = np.concatenate((feature_vec0, feature_vec1, feature_vec2, feature_vec3, feature_vec4, feature_vec5))
        df = pd.DataFrame([feature_vec])
        df.columns = [f'feature_vec:::{i}' for i in range(feature_vec.size)]

        return df

def file_to_dataframe_list(file_path):
    df_list = []
    for sentence in load_file(file_path):
        for df in sentence_find_verb(sentence):
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
    word_number = 0 #何単語目に出現したか
    for i, line in enumerate(sentence.split('\n')):
        if line[0] != '*':
            word_number += 1
        if '動詞' in line or '形容詞' in line or 'サ変可能' in line:
            ga_case_id = get_ga_tag(line)
            o_case_id = get_o_tag(line)
            ni_case_id = get_ni_tag(line)
            if is_num(ga_case_id):
                if not check_id_in_sentence(sentence, ga_case_id):
                    ga_case_id = 'exog'
            if is_num(o_case_id):
                if not check_id_in_sentence(sentence, o_case_id):
                    o_case_id = 'exog'
            if is_num(ni_case_id):
                if not check_id_in_sentence(sentence, ni_case_id):
                    ni_case_id = 'exog'
            yield sentence_to_vector(sentence, word_number, ga_case_id, o_case_id, ni_case_id)
        if line[0] == '*':
            continue

def feature_to_dataframe(feature):
    feature = feature.split(',')
    df = pd.DataFrame([feature])
    df.columns = [f'feature:{i}' for i in range(17)]
    df_ = feature_to_embed.feature_to_df_by_embed(feature)
    df = pd.merge(df, df_, left_index=True, right_index=True, how='outer')
    return df

def make_df_none():
    df_word_vector = word2vec.word_to_dataframe('')
    df_feature = feature_to_dataframe('*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*')
    df_none = pd.merge(df_word_vector, df_feature, left_index=True, right_index=True, how='outer')
    return df_none
def make_df_exo1():
    df_word_vector = word2vec.word_to_dataframe('exo1')
    df_feature = feature_to_dataframe('代名詞,*,*,*,*,*,ボク,僕,*,*,*,*,漢,*,*,*,*')
    df_exo1 = pd.merge(df_word_vector, df_feature, left_index=True, right_index=True, how='outer')
    return df_exo1
def make_df_exo2():
    df_word_vector = word2vec.word_to_dataframe('exo2')
    df_feature = feature_to_dataframe('代名詞,*,*,*,*,*,オマエ,御前,*,*,*,*,和,*,*,*,*')
    df_exo2 = pd.merge(df_word_vector, df_feature, left_index=True, right_index=True, how='outer')
    return df_exo2
def make_df_exog():
    df_word_vector = word2vec.word_to_dataframe('exog')
    df_feature = feature_to_dataframe('代名詞,*,*,*,*,*,コレ,此れ,*,*,*,*,和,*,*,*,*')
    df_exog = pd.merge(df_word_vector, df_feature, left_index=True, right_index=True, how='outer')
    return df_exog

def df_drop(df):
    for i in range(17):
        df = df.drop(f"feature:{i}", axis=1)
    df = df.drop('word', axis=1)
    return df

def sentence_to_vector(sentence, verb_number, ga_case_id, o_case_id, ni_case_id):
    df_none = make_df_none()
    df_exo1 = make_df_exo1()
    df_exo2 = make_df_exo2()
    df_exog = make_df_exog()
    df = pd.concat([df_none, df_exo1, df_exo2, df_exog])
    
    word_number = 0 #何単語目に出現したか
    phrase_count = 0 #文節のカウント
    count_head_word_number = 0 #文節ごとの単語のカウント（主辞判定のため）
    for i, line in enumerate(sentence.split('\n')):
        if line[0] == '*':
            head_word_number = int(line.split()[3].split('/')[0])
            count_head_word_number = 0
            phrase_count += 1
            continue
        word_number += 1
        word, feature, tag = line.split('\t')
        df_word_vector = word2vec.word_to_dataframe(word)
        df_feature = feature_to_dataframe(feature)
        df_ = pd.merge(df_word_vector, df_feature, left_index=True, right_index=True, how='outer')
    #その他の素性
        #形態素距離
        df_['形態素距離'] = abs(word_number-verb_number)
        #主辞
        if count_head_word_number == head_word_number:
            df_['主辞'] = 1
        count_head_word_number += 1
        #文節
        df_['文節'] = phrase_count
        #述語の場合はdepのタグ付けもいれる
        if word_number == verb_number:
            df_['is_verb'] = 1
            ga_dep_tag = get_ga_dep_tag(tag)
            df_['ga_dep_tag'] = ga_dep_tag
            o_dep_tag = get_o_dep_tag(tag)
            df_['o_dep_tag'] = o_dep_tag
            ni_dep_tag = get_ni_dep_tag(tag)
            df_['ni_dep_tag'] = ni_dep_tag
        
        #正解ラベルか確認して，正解を入れる．
        tag_id = get_tag_id(tag)
        if is_num(tag_id):
            if ga_case_id == tag_id:
                df_['ga_case'] = 1
            if o_case_id == tag_id:
                df_['o_case'] = 1
            if ni_case_id == tag_id:
                df_['ni_case'] = 1
        df = pd.concat([df, df_], ignore_index=True)

    if 'ga_case' not in df.keys():
        if ga_case_id == 'exo1':
            df.at[1, 'ga_case'] = 1
        elif ga_case_id == 'exo2':
            df.at[2, 'ga_case'] = 1
        elif ga_case_id == 'exog':
            df.at[3, 'ga_case'] = 1
        else:
            df.at[0, 'ga_case'] = 1
    if 'o_case' not in df.keys():
        if o_case_id == 'exo1':
            df.at[1, 'o_case'] = 1
        elif o_case_id == 'exo2':
            df.at[2, 'o_case'] = 1
        elif o_case_id == 'exog':
            df.at[3, 'o_case'] = 1
        else:
            df.at[0, 'o_case'] = 1
    if 'ni_case' not in df.keys():
        if ni_case_id == 'exo1':
            df.at[1, 'ni_case'] = 1
        elif ni_case_id == 'exo2':
            df.at[2, 'ni_case'] = 1
        elif ni_case_id == 'exog':
            df.at[3, 'ni_case'] = 1
        else:
            df.at[0, 'ni_case'] = 1
    df = df_drop(df)
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
            dataset += df_list
        dataset = reduction_dataframe(dataset)
        with open(f'./dataframe_list_{domain}.pickle', 'wb') as f:
            pickle.dump(dataset, f)
        del r
        del dataset

if __name__=='__main__':
    word2vec = Word2Vec(w2v_path)
    feature_to_embed = FeatureToEmbedID()
    main()
