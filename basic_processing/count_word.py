import os

directory = '/Users/sango.m.ab/Desktop/research/data'
output = {}
for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
  print(folda)
  count = 0
  for file in os.listdir(f'{directory}/knp/{folda}'):
    with open(f'{directory}/knp/{folda}/{file}') as f:
      for line in f:
        first = line.replace('\n', '').split('\t')[0].split(' ')[0]
        if first == '#' or first == '*' or first == '+':
          continue
        elif first == 'EOS':
          continue
        else:
          count += 1
  output[folda] = count
  print(count)
print(f'''
  |  | OC | OW | OY | PB | PM | PN |
  |---|---|---|---|---|---|---|
  | 単語数 | {output['OC']} | {output['OW']} | {output['OY']} | {output['PB']} | {output['PM']} | {output['PN']} |''')