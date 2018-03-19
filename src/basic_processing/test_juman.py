import os

directory = '/Users/sango.m.ab/Desktop/research/data'
for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
  for file in os.listdir(f'{directory}/jumanpp/{folda}'):
    print(file)
    with open(f'{directory}/jumanpp/{folda}/{file}') as f:
      text = ''
      for line in f:
        first = line.replace('\n', '').split('\t')[0].split(' ')[0]
        if first == '@':
          continue
        elif first == 'EOS':
          text += '\n'
        else:
          text += first
    file = file.replace('.jmn', '.txt')
    with open(f'{directory}/test_jumanpp/{folda}/{file}', 'w') as f:
      f.write(text)