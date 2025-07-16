import pandas as pd
import numpy as np
import subprocess

lib_counts = pd.read_csv('../data/pooled_comp/counts_db.csv', index_col=0)

metadata = {'sample':[x+y for x in ['T0','T5'] for y in ['A','B','C','D']],
            'edit_pool':['A','A','B','B']*2,
            'time':['start']*4+['end']*4,
            'comp_pool': ['A','B','C','D']*2}
pd.DataFrame(metadata).to_csv('../data/pooled_comp/metadata.csv')

#calculate custom normalization factors for DESeq2
def median_of_ratios(db):
    #use logs to prevent overflow
    geometric_mean = np.e**(np.sum(np.log(db), axis=1)*1/db.shape[1]) #geometric means of components across samples
    ratios = db/np.array(geometric_mean).reshape((-1,1)) #divide by column vector of geometric means of components
    return np.nanmedian(ratios, axis=0)

geometric_mean = np.e**(np.sum(np.log(lib_counts), axis=1)*1/lib_counts.shape[1]) #geometric means of components across samples
ratios = lib_counts/np.array(geometric_mean).reshape((-1,1)) #divide by column vector of geometric means of components
nlc_start = (lib_counts /median_of_ratios(lib_counts))[['T0A','T0B','T0C','T0D']].mean(axis=1)
nlc_end = (lib_counts /median_of_ratios(lib_counts))[['T5A','T5B','T5C','T5D']].mean(axis=1)
ratios = ratios.dropna()
ratios['start'] = ratios.index.map(nlc_start)
q_low = np.quantile(ratios.start, 0.25)
q_high = np.quantile(ratios.start, 1)
in_q = (ratios.start > q_low) & (ratios.start < q_high)
norm_params = {}
norm_factors = pd.DataFrame()
for i,m in enumerate(['T0A','T0B','T0C','T0D','T5A','T5B','T5C','T5D']):
    a, b = np.polyfit(np.log10(ratios.start[in_q]), ratios[m][in_q], deg=1)
    norm_params[m] = {'slope':a, 'intercept':b}
    norm_factors[m] = np.log10(nlc_start)*a+b
adj_counts = lib_counts/norm_factors #how counts are adjusted in DESeq2

np.savetxt('../data/pooled_comp/lin_log_normalization_factors.csv', norm_factors, delimiter=",")
subprocess.run(['Rscript','DESeq2.R'])