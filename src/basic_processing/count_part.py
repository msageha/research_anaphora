import os

directory = '/Users/sango.m.ab/Desktop/research/data'
parts = ['名詞', '動詞', '形容詞']
for part in parts:
  output = {}
  for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
    count = 0
    print(folda)
    for file in os.listdir(f'{directory}/annotated/{folda}'):
      with open(f'{directory}/annotated/{folda}/{file}') as f:
        for line in f:
          if f'\t{part},' in line:
            count += 1
    output[folda] = count
    print(count)
  print(f'''
    |  | OC | OW | OY | PB | PM | PN |
    |---|---|---|---|---|---|---|
    | {part} | {output['OC']} | {output['OW']} | {output['OY']} | {output['PB']} | {output['PM']} | {output['PN']} |''')