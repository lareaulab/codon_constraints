import os, glob, sys
import requests
import pandas as pd
import numpy as np
import subprocess

#Requires bowtie
illumina_dir = ... #specify directory of illumina fastq files from 'ESM1_T0T5_june24' -> excluded because of large size


#NOTE: sequenced reads include reads from a different library
#reads will be aligned to both libraries as well as empty backbone pZS165, but only reads aligning to library 'ylib2' 
#will be considered for this experiment
lib = pd.read_csv('../data/pooled_comp/combined_library.csv', index_col=0)
#empty backbone
with open('../data/pooled_comp/pzs165.txt', 'r') as f:
    pzs165 = f.readline()
#amplicons from both libraries
with open('../data/pooled_comp/combined_amplicons.fasta', 'w') as f:
    f.write('>pzs165'+'\n')
    f.write(pzs165.upper()+'\n')
    for i in lib.index:
        seq = lib.insert_seq[i]
        assert(len(seq) == 194)
        f.write('>'+i+'\n')
        f.write(seq.upper()+'\n')
        
subprocess.run('bowtie-build -f ../data/pooled_comp/combined_amplicons.fasta ../data/pooled_comp/bowtie_index/combined'.split(' '))

for name in ['T0A_S1', 'T0B_S2', 'T0C_S3', 'T0D_S4', 'T5A_S5', 'T5B_S6', 'T5C_S7', 'T5D_S8']:
    print(name[0:3])
    r1_name = illumina_dir + 'ESM1_'+name+'_L008_R1_001.fastq.gz' #make sure naming convention is correct
    results_name = '../data/pooled_comp/bowtie_results/'+name[0:3]+'.sam'
    command = "bowtie --threads 20 -n 2 -l 58 --best --trim5 28 --sam --sam-nosq --sam-nohead -q -x ../data/pooled_comp/bowtie_index/combined"
    command = command.split(' ') + [r1_name, results_name]
    subprocess.run(command)
    
    
def read_and_count_sam(samfile, maxlines=None, expected_start = '1', expected_cigar='151M', allow_mismatches=True, empty_name='pzs165', verbose=False):
    mismatches_count = {}
    with open(samfile, 'r') as f:
        counts = {}
        c = 0
        for line in f:
            if maxlines is not None and c >= maxlines:
                print(f'exceeded {maxlines} lines')
                break
            if line[0] == '@':
                continue
            c+=1
            vals = line.split("\t")
            pos = vals[3]
            flag = vals[1]
            cigar = vals[5]
            nm = [x for x in vals if x[0:2] == 'NM']
            if len(nm) == 0:
                nm = None
            else:
                nm = nm[0]
            #counting number of reads with mismatches:
            if nm in mismatches_count:
                mismatches_count[nm] += 1
            else:
                mismatches_count[nm] = 1
            if len(bin(int(flag))) < 7:
                rev_complement = '0'
            else:
                rev_complement = (bin(int(flag)))[-5] 
            #filter for reads that align to the start, align fully, are not a reverse complement (optional:have no mismatches)
            #don't have to follow these for pzs165 - the empty plasmid - if anything aligns to it in any way, want to count it
            if allow_mismatches:
                conditions = ((pos==expected_start) and (cigar==expected_cigar) and (rev_complement=='0'))
            else:
                conditions = ((pos==expected_start) and (cigar==expected_cigar) and (rev_complement=='0') and (nm == 'NM:i:0'))
            if conditions or vals[2]==empty_name: #if read meets condition, or belongs to empty vector
                rname = vals[2]
                if rname in counts:
                    counts[rname] += 1
                else:
                    counts[rname] = 1
            if verbose:
                print(vals)
        print('Mismatches:', mismatches_count)
        return counts

samples = ['T0A','T0B','T0C','T0D','T5A','T5B','T5C','T5D']
#reads are single read, 100 bp long 
#we have trimmed 28 bp from front, none from back
#only looked at forward reads
#expect them to match at base 21 (20 in 0-index), be of length 72
counts = {}
for s in samples:
    print(s)
    counts[s] = read_and_count_sam('../data/pooled_comp/bowtie_results/'+s+'.sam', expected_start='21', expected_cigar='72M', allow_mismatches=True)
counts_db = pd.DataFrame(counts)

#save only counts belonging to ylib2
keep = lib[lib.library != 'ylib1'].index
counts_db = counts_db[counts_db.index.isin(keep)]
#remove variants that do not have at least 10 reads in any sample
counts_db = counts_db[counts_db.max(axis = 1) >= 10]
counts_db.fillna(0)
counts_db.to_csv('../data/pooled_comp/counts_db.csv')