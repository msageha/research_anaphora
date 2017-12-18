import os
import subprocess
import re

def annotate_tab(folda, file):
    # cmd = "cabocha -f1 -n1 < "+text_name+" > "+text_name[:-4]+".cabocha"
    filename = file.replace('.txt', '')
    # cmd = f"jumanpp < ./extract_text/{folda}/{file}  > ./jumanpp/{folda}/{filename}.jmn"
    # output = subprocess.run(cmd, shell=True)
    #cmd = "knp -tab -anaphora -detail < juman_" + text_name[:-4] + ".jmn > knp_" + text_name[:-4] + ".knp"
    # cmd = f"knp -tab <./jumanpp/{folda}/{filename}.jmn > ./knp/{folda}/{filename}.knp"
    # output = subprocess.run(cmd, shell=True)
    cmd = f"knp -tab -anaphora <./jumanpp/{folda}/{filename}.jmn > ./knp_anaphora/{folda}/{filename}.knp"
    output = subprocess.run(cmd, shell=True)

from joblib import Parallel, delayed

for folda in ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']:
    print(f'start:::{folda}')
    r = Parallel(n_jobs=-1)(delayed(annotate_tab)(folda, file) for file in os.listdir(f'extract_text/{folda}/'))
    print(f'finish:::{folda}')
