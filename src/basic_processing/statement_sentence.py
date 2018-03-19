# 文内か，文間か
import os
import re

def get_num_from_id(text):
  m = re.search(r'id="([0-9]+)"', text)
  if m:
    return m.group(1)
  return None
def get_num_from_ga(text):
  m = re.search(r'ga="([0-9]+)"', text)
  if m:
    return m.group(1)
  return None
def get_num_from_o(text):
  m = re.search(r'o="([0-9]+)"', text)
  if m:
    return m.group(1)
  return None
def get_num_from_ni(text):
  m = re.search(r' ni="([0-9]+)"', text)
  if m:
    return m.group(1)
  return None



ga_dep = ['ga_dep="dep"', 'ga_dep="zero"']
o_dep = ['o_dep="dep"', 'o_dep="zero"']
ni_dep = ['ni_dep="dep"', 'ni_dep="zero"']
# 文内 - in
# 文間 - out
count_dic = {}
for key in ga_dep:
  count_dic[key] = {}
  count_dic[key]['in'] = 0
  count_dic[key]['out'] = 0
for key in o_dep:
  count_dic[key] = {}
  count_dic[key]['in'] = 0
  count_dic[key]['out'] = 0
for key in ni_dep:
  count_dic[key] = {}
  count_dic[key]['in'] = 0
  count_dic[key]['out'] = 0

output_dict = {}

directory = '/Users/sango.m.ab/Desktop/research/data'
for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
  print(folda)
  for file in os.listdir(f'{directory}/annotated/{folda}'):
    with open(f'{directory}/annotated/{folda}/{file}') as f:
      sentence = ''
      sentences = []
      for line in f:
        first = line.replace('\n', '').split('\t')[0].split(' ')[0]
        if first == '#' or first == '#!' or first =='*':
          continue
        elif first == 'EOS':
          sentences.append(sentence)
          sentence = ''
          continue
        sentence += line + '\n'
    for sentence in sentences:
      for word in sentence.split('\n'):
        for key in ga_dep:
          if key in word:
            id_num = get_num_from_ga(word)
            if id_num:
              if f'id="{id_num}"' in sentence:
                count_dic[key]['in'] += 1
              else:
                count_dic[key]['out'] += 1
        for key in o_dep:
          if key in word:
            id_num = get_num_from_o(word)
            if id_num:
              if f'id="{id_num}"' in sentence:
                count_dic[key]['in'] += 1
              else:
                count_dic[key]['out'] += 1
        for key in ni_dep:
          if key in word:
            id_num = get_num_from_ni(word)
            if id_num:
              if f'id="{id_num}"' in sentence:
                count_dic[key]['in'] += 1
              else:
                count_dic[key]['out'] += 1
                if key == 'ni_dep="dep"':
                  print(f'{file}, {id_num}')
  output_dict[folda] = count_dic
  count_dic = {}
  for key in ga_dep:
    count_dic[key] = {}
    count_dic[key]['in'] = 0
    count_dic[key]['out'] = 0
  for key in o_dep:
    count_dic[key] = {}
    count_dic[key]['in'] = 0
    count_dic[key]['out'] = 0
  for key in ni_dep:
    count_dic[key] = {}
    count_dic[key]['in'] = 0
    count_dic[key]['out'] = 0

word_num = [98996, 208108, 105792, 208749, 213386, 330314]
print('|       | OC   | OW   | OY   | PB    | PM    | PN    |')
for key in count_dic:
  print(f'''| 文内 {key} | {output_dict["OC"][key]['in']} | {output_dict["OW"][key]['in']} | {output_dict["OY"][key]['in']} | {output_dict["PB"][key]['in']} | {output_dict["PM"][key]['in']} | {output_dict["PN"][key]['in']} | ''')
for key in count_dic:
  print(f'''| 文間 {key} | {output_dict["OC"][key]['out']} | {output_dict["OW"][key]['out']} | {output_dict["OY"][key]['out']} | {output_dict["PB"][key]['out']} | {output_dict["PM"][key]['out']} | {output_dict["PN"][key]['out']} | ''')


print('割合')
print('|       | OC   | OW   | OY   | PB    | PM    | PN    |')
for key in count_dic:
  print(f'''| 文内 {key} | {round(output_dict["OC"][key]['in']/word_num[0]*100, 4)} | {round(output_dict["OW"][key]['in']/word_num[1]*100, 4)} | {round(output_dict["OY"][key]['in']/word_num[2]*100, 4)} | {round(output_dict["PB"][key]['in']/word_num[3]*100, 4)} | {round(output_dict["PM"][key]['in']/word_num[4]*100, 4)} | {round(output_dict["PN"][key]['in']/word_num[5]*100, 4)} | ''')
for key in count_dic:
  print(f'''| 文間 {key} | {round(output_dict["OC"][key]['out']/word_num[0]*100, 4)} | {round(output_dict["OW"][key]['out']/word_num[1]*100, 4)} | {round(output_dict["OY"][key]['out']/word_num[2]*100, 4)} | {round(output_dict["PB"][key]['out']/word_num[3]*100, 4)} | {round(output_dict["PM"][key]['out']/word_num[4]*100, 4)} | {round(output_dict["PN"][key]['out']/word_num[5]*100, 4)} | ''')
