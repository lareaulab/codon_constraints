#make '../data/results/combined_results.csv'
#combine predictions from undersampled and unweighted models, DMS Map seq, pLDDT scores, and gene rpkm
#run evaluate.py first to get predictions from models
import pandas as pd
import numpy as np

print('Loading model results')
undersampled = pd.read_csv('../data/models/nov12_24_undersample/scer_evaluation.csv', index_col=0)
unweighted =  pd.read_csv('../data/models/june18_24_unweighted/scer_evaluation.csv', index_col=0)

print('Loading DMS-Mapseq scores')
#dms-mapseq scores obtained from 
#Zubradt M, Gupta P, Persad S, Lambowitz AM, Weissman JS, Rouskin S. DMS-MaPseq for genome-wide or targeted RNA structure probing in vivo. 
#Nat Methods. 2017 Jan;14(1):75-82. doi: 10.1038/nmeth.4057. Epub 2016 Nov 7. PMID: 27819661; PMCID: PMC5508988.
#Files used:
#'GSM2241644_TGIRT_DMSvivo_YeastmRNA_COUNTS_Minus_NormSignal_Rep1.txt'
#'GSM2241644_TGIRT_DMSvivo_YeastmRNA_COUNTS_Plus_NormSignal_Rep1.txt'
dms = pd.read_csv('../data/reference/dms_mapseq.csv', index_col=0)

print('Loading pLDDT scores')
plddt_db = pd.read_csv('../data/reference/plddt_db.csv', index_col=0)

print('Loading per-gene RPKM values')
gene_expression = pd.read_csv('../data/reference/gene_rpkms.txt', index_col=0, sep='\t', names=['gene', 'gene_rpkm'])

print('Combine model scores, dms mapseq data, and gene rpkm values')
results = undersampled
results['target'] = results.gene+'_'+results.position.astype(str)
unweighted['target']=unweighted.gene+'_'+unweighted.position.astype(str)
results['unweighted_score'] = results.target.map(unweighted.set_index('target').model_score)
results['gene_rpkm'] = results.gene.map(gene_expression.gene_rpkm)

dms = dms.dropna()
dms['codon_number'] = dms.gene_pos//3
dms['pos_in_codon'] = dms.gene_pos%3
dms['target'] = dms.gene + '_' + dms.codon_number.astype(int).astype(str) 
for i in range(3):
    #DMS-Mapseq data
    results['dms_'+str(i)] = results.target.map(dms[dms.pos_in_codon == i].set_index('target').score)

print("remove pLDDT scores for genes where amino acids don't match expected")
aa_from_pdb = results.target.map(plddt_db.set_index('target').AA)
bad_idx = (~aa_from_pdb.isna()) & (aa_from_pdb != results.AA)
print(len(results[bad_idx].gene.unique()), 'have AA disagreements. skip plddt scores for them')
bad_genes = results[bad_idx].gene.unique()
results['pLDDT'] = results.target.map(plddt_db.set_index('target').plDDT)
results.pLDDT[results.gene.isin(bad_genes)] = None

print("Save")
results.to_csv('../data/results/model_results.csv')
