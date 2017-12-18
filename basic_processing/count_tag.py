import os
import re

def get_num_from_id(text):
  m = re.search(r'id="([0-9]+)"', text)
  if m:
    return m.group(1)
  return None
def ga(text):
  m = re.search(r'ga="(.+)"', text)
  if m:
    return m.group(1)
  return None
def o(text):
  m = re.search(r'o="(.+)"', text)
  if m:
    return m.group(1)
  return None
def ni(text):
  m = re.search(r' ni="(.+)"', text)
  if m:
    return m.group(1)
  return None

directory = '/Users/sango.m.ab/Desktop/research/data'

output = {}
noun_output = {}
verb_output = {}
for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
  count = 0
  noun_count = 0
  verb_count = 0
  print(folda)
  for file in os.listdir(f'{directory}/annotated/{folda}'):
    with open(f'{directory}/annotated/{folda}/{file}') as f:
      for line in f:
        if get_num_from_id(line):
          count += 1
          if f'\t名詞,' in line:
            noun_count += 1
          if f'\t動詞,' in line:
            verb_count += 1
  output[folda] = count
  noun_output[folda] = noun_count
  verb_output[folda] = verb_count
print(f'''
  |  | OC | OW | OY | PB | PM | PN |
  |---|---|---|---|---|---|---|
  | id=? のタグ付け | {output['OC']} | {output['OW']} | {output['OY']} | {output['PB']} | {output['PM']} | {output['PN']} |
  | 名詞でid=? のタグ付け | {noun_output['OC']} | {noun_output['OW']} | {noun_output['OY']} | {noun_output['PB']} | {noun_output['PM']} | {noun_output['PN']} |
  | 名詞でid=? のタグ付け(割合) | {noun_output['OC']/output['OC']} | {noun_output['OW']/output['OW']} | {noun_output['OY']/output['OY']} | {noun_output['PB']/output['PB']} | {noun_output['PM']/output['PM']} | {noun_output['PN']/output['PN']} |
  | 動詞でid=? のタグ付け | {verb_output['OC']} | {verb_output['OW']} | {verb_output['OY']} | {verb_output['PB']} | {verb_output['PM']} | {verb_output['PN']} |
  | 動詞でid=? のタグ付け(割合) | {verb_output['OC']/output['OC']} | {verb_output['OW']/output['OW']} | {verb_output['OY']/output['OY']} | {verb_output['PB']/output['PB']} | {verb_output['PM']/output['PM']} | {verb_output['PN']/output['PN']} |''')
