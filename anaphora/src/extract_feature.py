import re
import sqlite3

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
def get_num_from_tag(text):
    m = re.search(r'([0-9]+)[A-Z]', text)
    if m:
        return int(m.group(1))
    else:
        return None

#述語特徴
def get_predicate_lexical_infomation(line):
    word, feature, annotate = line.split('\t')
    feature_list = feature.split(',')
    vector_dict = {'述語素性0':'', '述語素性1':'', '述語素性2':'', '述語素性3':'', '述語素性4':'', '述語素性5':''}
    vector_dict['述語素性0'] = feature_list[0]
    vector_dict['述語素性1'] = feature_list[1]
    vector_dict['述語素性2'] = feature_list[2]
    vector_dict['述語素性3'] = feature_list[3]
    vector_dict['述語素性4'] = feature_list[4]
    vector_dict['述語素性5'] = feature_list[5]
    return vector_dict

# def predicate_sa_conjugate(text):
#     pass

#項候補
def get_condidate_lexical_infomation(line):
    word, feature, annotate = line.split('\t')
    feature_list = feature.split(',')
    vector_dict = {'格素性1':'', '格素性2':'', '格素性3':''}
    vector_dict['格素性1'] = feature_list[1]
    vector_dict['格素性2'] = feature_list[2]
    vector_dict['格素性3'] = feature_list[3]
    return vector_dict

def get_condidate_is_subject_head(condidate_line, sentence):
    subject_word_num = 0
    vector_dict = {'主辞':False}
    for line in sentence.split('\n'):
        if line == '':
            continue
        if line[0] == '*':
            tag = line
            subject_word_num = 0
        else:
            if condidate_line == line:
                subject_head_num = int(tag.split()[3].split('/')[0])
                if subject_head_num == subject_word_num:
                    vector_dict['主辞'] = True
                    break
            subject_word_num += 1
    return vector_dict

def get_condidate_postpositional_particle_infomation(condidate_line, sentence):
    for i, line in enumerate(sentence.split('\n')):
        if line == condidate_line:
            break
    condidate_postpositional_particle_line = sentence.split('\n')[i+1]
    vector_dict = {'項候補助詞':'', '項候補助詞素性0':'', '項候補助詞素性1':''}
    if condidate_postpositional_particle_line == '' or condidate_postpositional_particle_line[0] == '*':
        pass
    else:
        word, feature, annotate = condidate_postpositional_particle_line.split('\t')
        feature_list = feature.split(',')
        if feature_list[0] == '助詞' or feature_list[0] == '助動詞':
            vector_dict['項候補助詞'] = word
            vector_dict['項候補助詞素性0'] = feature_list[0]
            vector_dict['項候補助詞素性1'] = feature_list[1]
    return vector_dict

#述語と項候補間の統語構造
def get_condidate_is_after_predicate(predicate_word_position, condidate_word_position):
    vector_dict = {'前後関係':0}
    if predicate_word_position - condidate_word_position < 0:
        vector_dict['前後関係'] = -1
    else:
        vector_dict['前後関係'] = 1
    return vector_dict

def get_morpheme_distance(predicate_word_position, condidate_word_position):
    vector_dict = {'形態素距離':0}
    vector_dict['形態素距離'] = abs(int(predicate_word_position - condidate_word_position))
    return vector_dict

def get_dependency_distance(text, predicate_word_position, condidate_word_position):
    vector_dict = {'文節係り受け距離': 0}
    word_position = 0
    for i, line in enumerate(text.split('\n')):
        if line == '':
            continue
        if line[0] == '*':
            tag = line
        else:
            if word_position == predicate_word_position:
                predicate_tag = tag
            if word_position == condidate_word_position:
                condidate_tag = tag
            word_position += 1
    if condidate_word_position < predicate_word_position:
        first_tag = condidate_tag
        second_tag = predicate_tag
    else:
        first_tag = predicate_tag
        second_tag = condidate_tag
    dependency_distance = 0
    depending_position = get_num_from_tag(first_tag.split()[2])
    depending_goal = int(second_tag.split()[1])
    for i, line in enumerate(text.split('\n')):
        if line == '':
            continue
        if line[0] == '*':
            tag = line
            if int(tag.split()[1]) == depending_position:
                depending_position = get_num_from_tag(tag.split()[2])
                dependency_distance += 1
                if depending_position == depending_goal:
                    dependency_distance += 1
                    break
    vector_dict['文節係り受け距離'] = dependency_distance
    return vector_dict


#大規模データ素性
import sqlite3
from contextlib import closing
dbname = '/Users/sango.m.ab/Desktop/case frame/cf_J_w2v_mini_alt0.db'

connection = sqlite3.connect(dbname)
cursor = connection.cursor()

def get_case_frame(word, is_predicate):
    c = cursor.execute(f"SELECT headword, id, component_size, ガ格_size, ヲ格_size, ニ格_size FROM headword_id WHERE headword LIKE '{word}/%';")
    component_size_total = 0
    ga_size = 0
    o_size = 0
    ni_size = 0
    for row in c:
        component_size_total += row[2]
        ga_size += row[3]
        o_size += row[4]
        ni_size += row[5]

    if component_size_total != 0:
        ga_size /= component_size_total
        o_size /= component_size_total
        ni_size /= component_size_total
    if is_predicate:
        vector_dict = {'述語_case_frame_ガ格':ga_size, '述語_case_frame_ヲ格':o_size, '述語_case_frame_ニ格':ni_size}
    else:
        vector_dict = {'項候補_case_frame_ガ格':ga_size, '項候補_case_frame_ヲ格':o_size, '項候補_case_frame_ニ格':ni_size}
    return vector_dict
