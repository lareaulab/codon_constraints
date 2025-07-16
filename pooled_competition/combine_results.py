#make pooled_comp_results.csv
#combine information from pooled competition deseq2 results, model results, and libbrary info 
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

comp = pd.read_csv('../data/library/pooled_competition_library.csv', index_col=0)
deseq2 = pd.read_csv('../data/pooled_comp/deseq2_t0v5_l2fc058.csv', index_col=0)
deseq2 = deseq2.rename({'log2FoldChange':'l2fc'}, axis=1)
deseq2['finalMean'] = deseq2.baseMean*2**(deseq2.l2fc)
for m in ['baseMean', 'finalMean', 'l2fc', 'padj']:
    comp['comp_'+m] = comp.index.map(deseq2[m])

results = pd.read_csv('../data/results/model_results.csv', index_col=0)
db_genes = pd.read_csv("../data/reference/db_genes.csv", index_col=0) 
orftogene = pd.read_csv('../data/reference/orftogene_all.csv', index_col=0, sep=';', usecols=[0,1,2], names=['index','orf','gene'])


comp['gene_name'] = comp.gene.map(orftogene.gene)
comp['model_score'] = comp.target.map(results.set_index('target').model_score)
comp['label'] = comp.target.map(results.set_index('target').label)


#label guide-donor oligos by what kind of control they are
comp['control_type'] = 'not_control'
comp.control_type[comp.variant == 'r'] = 'random'
comp.control_type[(comp.variant == 'c') & (comp.gene.isin(db_genes[db_genes.essential].index))] = 'essential'
comp.control_type[(comp.variant == 'c') & (~comp.gene.isin(db_genes[db_genes.essential].index)) & (comp.gene.map(db_genes.rel_growth) <= 0.98) & (comp.gene.map(db_genes.rel_growth) > 0)] = 'nonessential_strong'
comp.control_type[(comp.variant == 'c') & (~comp.gene.isin(db_genes[db_genes.essential].index)) & (comp.gene.map(db_genes.rel_growth)> 0.98) & (comp.gene.map(db_genes.rel_growth) < 1)] = 'nonessential_weak'
comp.control_type[(comp.variant == 'c') & (~comp.gene.isin(db_genes[db_genes.essential].index)) &  (comp.gene.map(db_genes.rel_growth) >= 1)] = 'nonessential_adv'
comp.control_type[(comp.variant == 'd') & (comp.gene == 'YDL227C')] = 'HO_locus'

comp.to_csv('../data/results/pooled_comp_results.csv')
