#run to get esm representation for S. cerevisiae test set - needed for evaluate.py
import pandas as pd
import numpy as np
import os
import subprocess
from extract import create_parser, run

db_genes = pd.read_csv('../data/reference/db_genes.csv', index_col=0)
print('Make fasta for ESM')
with open('../data/reference/db_genes.fasta', 'w') as f:
    for g in db_genes.index:
        f.write('>'+g+'\n')
        f.write(db_genes.sacCer3[g]+'\n')
        
os.mkdir('../data/scer_esm_rep')

args_list = [
    'esm2_t33_650M_UR50D',
    '../data/reference/db_genes.fasta',
    '../data/scer_esm_rep/',
    '--include', 'per_tok'
]

parser = create_parser()
args = parser.parse_args(args_list)
print('Run extract.py')
run(args)
print('Done')