import os

directory = '/Users/sango.m.ab/Desktop/research/data'
sentence_dict = {}
word_dict = {}
for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
  sentence_count = 0
  id_count = 0
  word_count = 0
  print(folda)
  for file in os.listdir(f'{directory}/annotated/{folda}'):
    with open(f'{directory}/annotated/{folda}/{file}') as f:
      for line in f:
        first = line.replace('\n', '').split('\t')[0].split(' ')[0]
        if first == 'EOS':
          sentence_count += 1
          id_count = 0
        elif first[0] != '*' and first[0] != '#':
          word_count += 1
  print(sentence_count)
  sentence_dict[folda] = sentence_count
  word_dict[folda] = word_count
print(f'''
  |  | OC | OW | OY | PB | PM | PN |
  |---|---|---|---|---|---|---|
  | 文数 | {sentence_dict['OC']} | {sentence_dict['OW']} | {sentence_dict['OY']} | {sentence_dict['PB']} | {sentence_dict['PM']} | {sentence_dict['PN']} |
  | 単語数 | {word_dict['OC']} | {word_dict['OW']} | {word_dict['OY']} | {word_dict['PB']} | {word_dict['PM']} | {word_dict['PN']} |
  | 1文あたりの単語数 | {word_dict['OC']/sentence_dict['OC']} | {word_dict['OW']/sentence_dict['OW']} | {word_dict['OY']/sentence_dict['OY']} | {word_dict['PB']/sentence_dict['PB']} | {word_dict['PM']/sentence_dict['PM']} | {word_dict['PN']/sentence_dict['PN']} |
  ''')