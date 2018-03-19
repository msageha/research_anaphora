import os

dep_type = ['ga_dep="dep"', 'o_dep="dep"', 'ni_dep="dep"', 'ha_dep="dep"',
  'ga_dep="zero"', 'o_dep="zero"', 'ni_dep="zero"', 'ha_dep="zero"']

exo_type = ['ga="exo1"', 'ga="exo2"', 'ga="exog"', 'o="exo1"', 'o="exo2"', 'o="exog"',
  ' ni="exo1"',' ni="exo2"',' ni="exog"','ga/ni="exo1"','ga/ni="exo2"','ga/ni="exog"',
  'ha="exo1"','ha="exo2"','ha="exog"']

count_dic = {}
for key in exo_type:
  count_dic[key] = 0

output_dict = {}
directory = '/Users/sango.m.ab/Desktop/research/data'
for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
  print(folda)
  for file in os.listdir(f'{directory}/annotated/{folda}'):
    with open(f'{directory}/annotated/{folda}/{file}') as f:
      for line in f:
        first = line.replace('\n', '').split('\t')[0].split(' ')[0]
        if first == '#' or first == '#!' or first =='*':
          continue
        elif first == 'EOS':
          continue
        for key in count_dic:
          if key in line:
            count_dic[key] += 1
  for key in count_dic:
    print(f'{key}\t{count_dic[key]}')
  output_dict[folda] = count_dic
  count_dic = {}
  for key in exo_type:
    count_dic[key] = 0

word_num = [98996, 208108, 105792, 208749, 213386, 330314]
print('|       | OC   | OW   | OY   | PB    | PM    | PN    |')
for key in exo_type:
  print(f'| {key} | {output_dict["OC"][key]} | {output_dict["OW"][key]} | {output_dict["OY"][key]} | {output_dict["PB"][key]} | {output_dict["PM"][key]} | {output_dict["PN"][key]} | ')

print('割合')
print('|       | OC   | OW   | OY   | PB    | PM    | PN    |')
for key in exo_type:
  print(f'| {key} | {round(output_dict["OC"][key]/word_num[0]*100, 4)} | {round(output_dict["OW"][key]/word_num[1]*100, 4)} | {round(output_dict["OY"][key]/word_num[2]*100, 4)} | {round(output_dict["PB"][key]/word_num[3]*100, 4)} | {round(output_dict["PM"][key]/word_num[4]*100, 4)} | {round(output_dict["PN"][key]/word_num[5]*100, 4)} | ')
