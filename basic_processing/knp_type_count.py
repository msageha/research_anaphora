import os

#C 直接係り受けをもつ格要素 (格は明示されている)
C_type = ['ガ/C/', 'ヲ/C/', 'ニ/C/']
C2_type = ['ガ２/C/', 'ヲ２/C/', 'ニ２/C/']
#N 直接係り受けをもつ格要素 (格は明示されていない:未格,被連体修飾詞)
N_type = ['ガ/N/', 'ヲ/N/', 'ニ/N/']
N２_type = ['ガ２/N/', 'ヲ２/N/', 'ニ２/N/']

count_dic = {}
for key in C_type:
  count_dic[key] = 0
for key in C2_type:
  count_dic[key] = 0
for key in N_type:
  count_dic[key] = 0
for key in N2_type:
  count_dic[key] = 0

directory = '/Users/sango.m.ab/Desktop/research/data'
for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
  print(folda)
  for file in os.listdir(f'{directory}/knp/{folda}'):
    with open(f'{directory}/knp/{folda}/{file}') as f:
      text = ''
      for line in f:
        first = line.split(' ')[0].split('\t')[0]
        if first == '+':
          for key in count_dic:
            if key in line:
              count_dic[key] += 1
  for key in count_dic:
    print(f'{key}\t{count_dic[key]}')
  count_dic = {}
  for key in C_type:
    count_dic[key] = 0
  for key in C2_type:
    count_dic[key] = 0
  for key in N_type:
    count_dic[key] = 0
  for key in N2_type:
    count_dic[key] = 0

# ガ/C/