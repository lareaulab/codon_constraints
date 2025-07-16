#make predictions for S. cerevisiae test set using undersampled and unweighted models
#run get_esm_rep.py first to generate ESM2 representations for S. cerevisae genes
import pandas as pd
import numpy as np
import torch
from train_model import MultiInputNet

def load_model(model_save, params, input_1d_shape=(1280,)):
    model = MultiInputNet(include_2d=False, input_1d_shape=input_1d_shape, model_parameters=params)
    model.load_state_dict(torch.load(model_save, map_location="cpu"))
    model.eval()
    return model

results_dir = '../data/models/'
for run_name in ['nov12_24_undersample', 'june18_24_unweighted']:
    db_genes = pd.read_csv('../data/reference/db_genes.csv', index_col=0)
    db_folds = pd.read_csv(results_dir+run_name+'/db_folds.csv', index_col=0)
    #add fold information to db_genes
    #we make predictions using the out-of-fold model
    gene_indices = pd.DataFrame(index = list(db_folds.index))
    gene_indices['i'] = np.arange(len(gene_indices))
    gene_indices['locus_tag'] = gene_indices.index.map(db_folds.locus_tag)
    gene_indices['protein_id'] = list(db_folds.index)
    gene_indices = gene_indices.set_index('locus_tag')
    gene_indices = gene_indices[~gene_indices.index.duplicated()]
    db_genes['i'] = db_genes.index.map(gene_indices.i)
    db_genes['protein'] = db_genes.index.map(gene_indices.protein_id)
    db_genes['fold'] = db_genes.protein.map(db_folds.fold)
    db_genes = db_genes.where(pd.notnull(db_genes), None) #fill na with None

    protein_db = pd.read_csv('../data/reference/uniprot_features_db.csv', index_col=0)
    orftogene = pd.read_csv('../data/reference/orftogene_all.csv', index_col=0, sep=';', usecols=[0,1,2], names=['index','orf','gene'])
    genetoorf = {orftogene.gene[orf]:orf for orf in orftogene.index}
    db_genes['gene_name'] = db_genes.index.map(orftogene.gene)

    #remove dubious orfs. List aquired from SGD on 11/27/24
    #http://sgd-archive.yeastgenome.org/sequence/S288C_reference/orf_dna/orf_genomic_dubious.fasta.gz
    with open('../data/reference/orf_genomic_dubious.fasta', 'r') as f:
        dubious = []
        for line in f.readlines():
            line = line.strip()
            if line[0] == '>':
                dubious.append(line.split()[0][1:])

    go_annotation_file = '../data/GO_enrichment/downloads/sgd.gaf'
    columns = ['db', 'db_id', 'gene', 'relation', 'GO_id', 'reference', 'evidence_code', 
               'with', 'aspect', 'name', 'synonym', 'object_type', 'taxon', 'date', 
               'assigned_by', 'extension', 'gene_product_id']
    gaf = pd.read_csv(go_annotation_file, names=columns, comment='!', sep='\t')

    #remove dubious
    print('Number of genes')
    print(len(db_genes))
    db_genes = db_genes[~db_genes.index.isin(dubious)]
    print('after removing dubious')
    print(len(db_genes))
    #and remove retrotransposons
    retrotransposons = gaf[gaf.GO_id == 'GO:0032197'].gene.unique()
    db_genes = db_genes[~db_genes.index.isin(retrotransposons)]
    print('after removing retrotransposons')
    print(len(db_genes))

    slow_codons = ['GCG','CGA','CGG', 'TCG', 'AGC', 'GGG', 'GGA', 'GGC', 'TGC', 
                                   'ATA', 'CTC', 'CTG', 'CCG', 'CCC']
    fast_codons = ['GCC','GCT', 'AGA', 'TCT', 'TCC', 'GGT', 'TGT', 
                   'ATC', 'ATT', 'TTG', 'CCT', 'CCA']
    allowed_codons = slow_codons + fast_codons

    codon_info = pd.read_csv("../data/reference/codon_info.csv", index_col=0)
    codon_info['label'] = [(1 if x in fast_codons else (0 if x in slow_codons else None)) for x in codon_info.index]

    to_label = np.vectorize(lambda x: codon_info.label.get(x, default=None))
    to_AA = np.vectorize(lambda x: codon_info.AA.get(x, default=None))

    def get_weights(labels, AA_seq):
        #weights for AA in gene that have only slow or only fast are 0
        #otherwise, sum(weight_fast) = sum(weight_slow) = min(slow, fast)
        #e.g. weight higher-frequency label for each AA for the gene to match the lower-frequency label
        assert(labels.shape == AA_seq.shape)
        weights = np.zeros(AA_seq.shape)
        for aa in np.unique(AA_seq):
            if np.all(labels[AA_seq==aa] == 0) or np.all(labels[AA_seq==aa] == 1):
                continue #all fast or all slow, weight = 0
            sum_fast = np.sum(labels[AA_seq == aa] == 1)
            sum_slow = np.sum(labels[AA_seq == aa] == 0)
            weights[(AA_seq == aa) & (labels == 1)] = min(sum_fast, sum_slow)/(sum_fast)
            weights[(AA_seq == aa) & (labels == 0)] = min(sum_fast, sum_slow)/(sum_slow)
        return weights

    #so, for all genes, for all codons that are in allowed codons - get esm-based model score
    #have gene and pos (in gene, in codons) and old_codon I guess
    model_path =results_dir+run_name+'/fold'
    models = {x:load_model(model_save=model_path+str(x)+'_best_model.pt',
                           params={'dropout_rate':0.5, 'layers_1d_size':128, 'layers_combined':2, 
                                   'layers_combined_size':64, 'batch_norm':False}
                          ) for x in range(5)}
    input_size = 1280
    default_fold = 0
    max_pos = 1021 #model only looks at first 1022 positions ... ignore the rest
    results = []

    c = 0
    print('Running model for', run_name)
    for g in db_genes.index:
        seq = db_genes.sacCer3[g]
        codon_seq = np.array([seq[i*3:i*3+3] for i in range(0, len(seq)//3)])
        allowed_idx = np.where(np.in1d(codon_seq, allowed_codons))[0] 
        allowed_idx = allowed_idx[allowed_idx <= max_pos]
        if len(allowed_idx) == 0:
            continue #just ignore this gene, no AA work
        #ESM2 representations can be created by running get_esm_rep_scer.py
        esm_rep = torch.load('../data/scer_esm_rep/'+g+'.pt')['representations'][33].numpy() #length - 1 (no stop codon)
        assert(esm_rep.shape[0] >= max(allowed_idx)), f'{g}, len {len(codon_seq)}, esm_rep {esm_rep.shape[0]}, max idx {max(allowed_idx)}'
        esm_rep = esm_rep[allowed_idx]
        fold = (default_fold if db_genes.fold[g] is None else db_genes.fold[g])
        model = models[fold]
        scores = model(torch.tensor(esm_rep, dtype=torch.float32).reshape((-1,input_size))).reshape((-1,)).detach().numpy()
        old_codons = codon_seq[allowed_idx]
        AA_seq = to_AA(old_codons)
        labels = to_label(old_codons)
        weights = get_weights(labels, AA_seq)   
        results.append(pd.DataFrame({'gene':g, 'position':allowed_idx, 'codon':old_codons, 'AA':AA_seq, 'label':labels,
                                     'model_score':scores, 'weight':weights}))
        if c%500 == 0:
            print(c, 'complete')
        c+=1
    results = pd.concat(results, ignore_index=True)
    results.to_csv(results_dir+run_name+'/scer_evaluation.csv')
