import os

directory = '/Users/sango.m.ab/Desktop/research/data'
output = {}
for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
  sentence_count = 0
  id_count = 0
  print(folda)
  for file in os.listdir(f'{directory}/annotated/{folda}'):
    with open(f'{directory}/annotated/{folda}/{file}') as f:
      for line in f:
        first = line.replace('\n', '').split('\t')[0].split(' ')[0]
        if first == 'EOS':
          sentence_count += 1
          id_count = 0
  print(sentence_count)
  output[folda] = sentence_count
print(f'''
  |  | OC | OW | OY | PB | PM | PN |
  |---|---|---|---|---|---|---|
  | 文数 | {output['OC']} | {output['OW']} | {output['OY']} | {output['PB']} | {output['PM']} | {output['PN']} |''')