# 文間，何文まで遠くにあるか
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

zero = ['ga_dep="zero"', 'o_dep="zero"', 'ni_dep="zero"']
count_dic = {}
for key in zero:
  count_dic[key] = {}
  for i in range(0, 31):
    count_dic[key][i] = 0

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
    for i, sentence in enumerate(sentences):
      for word in sentence.split('\n'):
        for key in zero:
          if key in word:
            if 'ga_' in key:
              id_num = get_num_from_ga(word)
            elif 'o_' in key:
              id_num = get_num_from_o(word)
            elif 'ni_' in key:
              id_num = get_num_from_ni(word)
            if id_num:
              if f'id="{id_num}"' in sentence:
                count_dic[key][0] += 1
              else:
                for j, _ in enumerate(sentences):
                  if f'id="{id_num}"' in sentences[j]:
                    if abs(j-i) < 30:
                      count_dic[key][abs(j-i)] += 1
                    else:
                      count_dic[key][30] += 1

  output_dict[folda] = count_dic
  count_dic = {}
  for key in zero:
    count_dic[key] = {}
    for i in range(0, 31):
      count_dic[key][i] = 0

bunkan_sum_num = [4791, 3811, 2811, 7990, 7852, 10790]
print('|       | OC   | OW   | OY   | PB    | PM    | PN    |')
for i in range(31):
  oc = 0
  ow = 0
  oy = 0
  pb = 0
  pm = 0
  pn = 0
  for key in count_dic:
    oc += output_dict["OC"][key][i]
    ow += output_dict["OW"][key][i]
    oy += output_dict["OY"][key][i]
    pb += output_dict["PB"][key][i]
    pm += output_dict["PM"][key][i]
    pn += output_dict["PN"][key][i]
  print(f'''| 文間 {i}文離れている | {oc} | {ow} | {oy} | {pb} | {pm} | {pn} | ''')

print('割合')
print('|       | OC   | OW   | OY   | PB    | PM    | PN    |')
for i in range(31):
  oc = 0
  ow = 0
  oy = 0
  pb = 0
  pm = 0
  pn = 0
  for key in count_dic:
    oc += output_dict["OC"][key][i]
    ow += output_dict["OW"][key][i]
    oy += output_dict["OY"][key][i]
    pb += output_dict["PB"][key][i]
    pm += output_dict["PM"][key][i]
    pn += output_dict["PN"][key][i]
  print(f'''| 文間 {i}文離れている | {round(oc/bunkan_sum_num[0]*100, 4)} | {round(ow/bunkan_sum_num[1]*100, 4)} | {round(oy/bunkan_sum_num[2]*100, 4)} | {round(pb/bunkan_sum_num[3]*100, 4)} | {round(pm/bunkan_sum_num[4]*100, 4)} | {round(pn/bunkan_sum_num[5]*100, 4)} | ''')

