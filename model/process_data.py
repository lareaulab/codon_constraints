import time
import os, sys
print(os.path.dirname(sys.executable))
import numpy as np
import pandas as pd
import torch
print(torch.__version__)
print(torch.__file__)
import h5py
from torch.utils.data import Dataset, DataLoader
import subprocess
import shutil
from contextlib import redirect_stdout
import json
import sklearn.decomposition

DATA_DIR = '../data/model_training/'

FEATURE_SETS = ['control_1d', 'exclude_esm', 'only_esm', 'full_1d', 
                'no_esm_1d', 'positive_control', 'no_llm_1d', 'noisy_control']
FEATURE_SETS_1D = ['control_1d', 'only_esm', 'full_1d', 
                   'no_esm_1d', 'positive_control', 'no_llm_1d', 'noisy_control']

'''
collation functions to use when using DataLoader to load from HDF5_dataset
'''
#use if list_of_tuples is (input_1d, input_2d), labels
def collate_all(list_of_tuples):
    inputs, labels = zip(*list_of_tuples)
    inputs_1d, inputs_2d = zip(*inputs)
    inputs_2d = torch.permute(torch.cat(inputs_2d), (0,2,1)) #this is the orientation pytorch expects
    return (torch.cat(inputs_1d), inputs_2d), torch.cat(labels)

#use if list_of_tuples is (input_1d), labels
def collate_1d(list_of_tuples):
    inputs_1d, labels = zip(*list_of_tuples)
    return torch.cat(inputs_1d), torch.cat(labels)

"""
Tests that HDF5_maker and HDF5_dataset are working correctly
"""
def unit_test_hdf5(directory=""):
    unit_test = pd.DataFrame(index = ["gene"+str(i) for i in range(1,11)])
    unit_test["codon_seq"] = [np.array(["ATG","GAT","GAC","TAG"]), np.array(["GAA"]*2000)]+[np.array(["AAA"]*100)]*8
    unit_test["positions"] = [np.array([0,1,2]), np.array([1,5,1400])] + [np.array([10,20])]*8
    unit_test["species"] = ["Saccharomyces_cerevisiae"]*10
    hm = HDF5_maker(unit_test, directory, "test.hdf5", "test_aa.fa", "test_esm_save_dir", erase_existing=True)
    hm.run()
    hd = HDF5_dataset(directory+"test.hdf5", gene_list=np.array(unit_test.index), shuffle=False, validate=True)
    
    (r1, r2), l = hd[0]
    #just for 1
    assert(r1[0][66] == 1) #first is m
    assert(torch.all(torch.sum(r2[:,:,0] == 0, axis=1) == 4))  #padding codon
    assert(torch.all(torch.sum(r2[:,:,1] == 1, axis=1) == 3))  #masking codon
    assert(torch.all(torch.sum(r2[:,:,66] == 0, axis=1) == 3)) #padding aa #because X counts as padding
    with h5py.File(directory+'test.hdf5', 'r') as f:
        assert(np.all(f["gene1/rel_tai_seq"][...] == np.array([0.5,0,1.,0.5]))) #THIS SHOULD BE 0.5, 0, 1, 0.5
        assert(np.all(f["gene1/codon_seq"][...] != 1)) #no mask
   
    for i in range(len(unit_test.index)):
        try:
            (r1, r2), l = hd[i]
            val_pos =  np.array(unit_test["positions"][unit_test.index[i]][unit_test["positions"][unit_test.index[i]] <= 1021])
            assert(np.all(np.array(r1[:,-1], dtype='int64') == val_pos)), i
            assert(torch.all(r1[:,16] == 1)) #is yeast
            assert(torch.all(torch.sum(r1[:,:65+21], axis=1) == 2)) #one hot
            assert(torch.all(torch.sum(r2[:,:,:-1], axis=2) == 2)) #one hot
            assert(torch.all(torch.sum(l, axis=1) == 1)) #one hot
        except Exception as e:
            print("Problem with", i, unit_test.index[i])
            print(e)
    os.remove(directory+"test.hdf5")


"""
Acts as a pytorch dataset. 

Loads, preprocesses, and concatenates desired data from an hdf5 file
Hdf5 file is organized with a path for each gene, with a dataset for each feature under that path
    Datasets include aa_seq, codon_seq, species (integer label of gene), esm (esm-1b per-aa representations), 
    tai_seq (tAI for each codon), rel_tai_seq (relative tAI for each codon), positions (conserved positions of interest)

__getitem__ returns the dataset for each gene. Use a dataloader to concatenate these datasets smoothly
    Generally returns (input_1d, input_2d), labels unless otherwise specified
    
hdf5_fname : filename of hdf5 file to use
gene_list : genes to include in dataset
pos_name : name of positions dataset to use in hdf5 file
tai_name : name of tAI dataset to use in hdf5 file 
window_size : window of sequence to truncate/pad to for 2d input. [window is centered on position of interest]
validate : use sanity checks or not
control_1d : Return only the amino acid (one-hot), species (one-hot), and position (integer) for each position of interest
    used as a control 
shuffle : shuffle the gene_list (no need to do this, dataloader will shuffle)
bin_labels : if True, return an integer label (0 for slow, 1 for fast, 2 for neither). 
            If False, return the tAI for the position
binary : exclude the 'neither' category. Returns a boolean (0 for slow, 1 for fast)
only_1d : only return 1d inputs (so __getitem__ will return input_1d, labels)
exclude_esm : don't include esm representations in 1d inputs
avg_esm : include the average esm rep for the gene, rather than the per-amino acid representation for the position of interest
include_index : include extra information (index of gene, codon label, etc.) that is NOT intended to be given to model,
    but just to identify datapoints [testing loop will exclude this information]
balance_by_aa : Keep the same number of slow and fast codons for each amino acid for each gene. 
        (will randomly undersample the overrepresented category each time __getitem__ is called on that gene)
        Returned labels will be binary, (0 for slow, 1 for fast) with no neither category
filter_by_aa : will return only positions corresponding to given amino acids [use one-letter codes]
"""
#TODO update documentation
class HDF5_dataset(Dataset):
    #thinking of adding weights
    #if there are weights, they will be the last column before indices
    def __init__(self, hdf5_fname, gene_list = None, pos_name="positions", tai_name="rel_tai_seq", esm_name='esm',
                  window_size=512, 
                features = 'only_esm', validate=False, weight_by_aa=False, shuffle=False, bin_labels=True, 
                  change_species={}, #a dictionary, ex: {13:10} -- (convert s_cer to k_afr - affect onehot but not indices)
                 avg_esm=False, include_index=False, balance_by_aa=False, balance_type='undersample', filter_by_aa=None, binary=True,
                  num_species=13, label_mode='multi_er', apply_pca=False, pca_dir=None, positions_in_order=True, unmixed=False):
        #aa_list only 
        #label_mode can be
            #tai, er, multi_er
        #multi er gets rid of codons that seem to vary in tAI between species
        #also restricted to ALA, ARG, CYS, GLY, ILE, LEU, PRO, SER
        self.weight_by_aa = weight_by_aa
        self.has_weights = weight_by_aa #if something (e.g. preloader) sets weights, you can change this
        #use has_weights (not weight_by_aa) to check if last column should be used as weights
        if weight_by_aa:
            print('Set weight by aa, balance by aa is False')
            balance_by_aa = False
        
        self.balance_type = balance_type
        assert(self.balance_type in ['undersample','oversample'])

        self.label_mode = label_mode
        self.change_species = change_species
        self.positions_in_order = positions_in_order
        print(f'positions in order is {self.positions_in_order}, change species is {self.change_species}, label mode is {self.label_mode}')
        self.hdf5_fname = hdf5_fname
        if gene_list is None:
            with h5py.File(self.hdf5_fname, "r") as f:
                self.genes = np.array(list(f.keys()))
        else:
            self.genes = gene_list
        if shuffle: #shuffling can be done by dataloader, no need to do it here
            print("shuffling")
            np.random.shuffle(self.genes)
        self.unmixed=unmixed
        if self.unmixed:
            print('Unmixed')
        else:
            print('mixed')
        #load pca components
        self.apply_pca = apply_pca
        if self.apply_pca:
            self.pca_dir = pca_dir #need to supply a directory where pre-computed PCA values are stored
            self.pca_components = torch.tensor(np.load(self.pca_dir+'_components.npy'))
            self.pca_mean = torch.tensor(np.load(self.pca_dir+'_mean.npy'))
        
        self.pos_name = pos_name
        self.tai_name = tai_name
        self.esm_name = esm_name

        self.features = features
        #what to load from HDF5 file
        self.load_names = ["codon_seq","aa_seq","species",self.esm_name, self.pos_name,self.tai_name]
        #what to look at by position
        self.at_pos_names = ["codon_seq","aa_seq",self.esm_name, self.tai_name]
        #what to mask by position
        self.mask_values = {"codon_seq":1, self.tai_name:0.5}
        #what to pad
        self.pad_values = {"codon_seq":0, "aa_seq":0, self.tai_name:self.mask_values[self.tai_name]} #this?
        #what to one-hot encode
        self.num_classes = {"codon_seq":66, "aa_seq":21, "species":num_species, "aa_seq_at_pos":21}
        
        #output
        #concatenate 2d: codon_seq, aa_seq, tai_seq ... 1d: species, aa_label, esm_rep ... label: tai_label
        self.names_2d = ["codon_seq","aa_seq"]
        self.names_1d = ["species","aa_seq_at_pos",self.esm_name+"_at_pos",self.pos_name]
        self.label = self.tai_name+"_at_pos"
        
        self.avg_esm = avg_esm
        
        if self.features != 'full':
            validate = False
            print('Unusual settings: set validation == false for dataset')
        
        features_1d = FEATURE_SETS_1D
        self.only_1d = (self.features in features_1d) 

        self.add_noise = False
        if self.features == 'only_esm':
            self.names_1d = [self.esm_name+'_at_pos'] 
        elif self.features == 'full_1d':
            self.names_1d = ["species","aa_seq_at_pos",self.esm_name+"_at_pos",self.pos_name]
        
        elif self.features == 'exclude_esm':
            self.load_names = ["codon_seq","aa_seq","species",self.pos_name,self.tai_name]
            self.at_pos_names = ["aa_seq", self.tai_name]
            self.names_1d = ["species","aa_seq_at_pos",self.pos_name]
        elif self.features == 'control_1d':
            self.names_1d = ["species","aa_seq_at_pos"]
        elif self.features == 'positive_control':
            self.names_1d = ["species", "aa_seq_at_pos", self.label]
        elif self.features == 'no_llm_1d':
            self.names_1d = ["species", "aa_seq_at_pos", self.pos_name]
        elif self.features == 'noisy_control':
            self.names_1d = ["species", "aa_seq_at_pos", self.label]
            self.add_noise = True
            self.flip = 0.2

        #don't load esm or other large datasets if they are not in names_1d
        for x in (self.esm_name, ):
            if not x+'_at_pos' in self.names_1d:
                print(f'Removing "{x}" from load and pos names')
                self.load_names = [y for y in self.load_names if y != x]
                self.at_pos_names = [y for y in self.at_pos_names if y != x]

            
        if self.pos_name == 'all':
            #if we want to use all positions
            #then there is no need to load positions from esm
            self.load_names.remove(self.pos_name)
            if filter_by_aa == None:
                #since there's invalid codons/aa that would have previously been just not included in positions
                raise ValueError("All positions - need to specify amino acids to use.")
        
        self.bin_labels = bin_labels
        self.binary = binary
        self.verbose = False
        #don't use positions above max_pos
        self.max_pos = 1021 #all pos must be <= this - ESM truncation 
        #when we return position (in 1d) - divide by this to keep within 0-1
        # #don't return more than num_pos_max positions
        #window size for 2d reps:
        self.window_size = window_size #in codons

        self.include_index = include_index
        self.num_index_rows = 0
        if self.include_index:
            self.num_index_rows = 5 #gene_id, position, species, codon, label
            self.at_pos_names.append('codon_seq') #need codon seq at position to correctly return index info
        self.balance_by_aa = balance_by_aa
        if self.balance_by_aa:
            self.binary=True 
            print("Balancing by aa, so binary set to True")
        self.filter_by_aa = filter_by_aa

        if self.filter_by_aa is not None: #will narrow 
            aa_list = "XMKVELCSFGYIPHRADQNWT" #filter by aa needs to use one-letter codes
            self.filter_by_aa = sorted([aa_list.index(aa) for aa in filter_by_aa]) #need integer aa labels
            self.num_classes['aa_seq_at_pos'] = len(self.filter_by_aa)
            self.aa_labels = {self.filter_by_aa[i]:i for i in range(len(self.filter_by_aa))}
            self.aa_replacement = torch.zeros(len(aa_list)+1, dtype=torch.long)
            for key, value in self.aa_labels.items():
                self.aa_replacement[key] = value
            print("AA replacement dictionary is ", self.aa_replacement)
            #NOTE: anything NOT in the dictionary is set to 0 
        self.validate=validate

        if self.binary and not self.balance_by_aa:
            print("NOTE that since using binary and not balancing by aa, neutral is considered as fast.")

        #standard
        self.rel_tai_low_th = 0.2
        self.rel_tai_high_th = 0.6

        if self.label_mode != 'tai': #label_mode is er, or multi_er
            #don't load esm if they are not in names_1d
            assert('er' in self.label_mode)
            if (not self.tai_name+'_at_pos' in self.names_1d) and (not self.tai_name in self.names_2d):
                print(f'Using er, removing "{self.tai_name}" from load, pos, mask and pad names')
                self.load_names = [y for y in self.load_names if y != self.tai_name]
                self.at_pos_names = [y for y in self.at_pos_names if y != self.tai_name]
                self.mask_values = {y:self.mask_values[y] for y in self.mask_values if y != self.tai_name}
                self.pad_values = {y:self.pad_values[y] for y in self.pad_values if y != self.tai_name}
            self.label = 'er' 
            self.bin_labels = False
            print(f'Using er, set bin_labels to False, label is {self.label}')
            #codons where er in sacCer3 < 0.85, excluding trp, stop
            slow_codons = ['TGC', 'TCG', 'AGC', 'CAG', 'CCG', 'CCC', 'ACA', 'ACG', 'GCA', 'GCG', 'GGG', 
                           'GGA', 'GGC', 'ATA', 'CTG', 'CGA', 'CGC', 'CGG', 'AGG', 'GTG', 'GAG']
            #codons where er in sacCer3 > 0.85, excluding phe, asn, tyr, asp, his, lys
            fast_codons = ['TGT', 'TCT', 'TCA', 'TCC', 'AGT', 'CAA', 'ATG', 'CCT', 'CCA', 'ACC', 
                           'ACT', 'GCC', 'GCT', 'GGT', 'ATC', 'ATT', 'TTA', 'TTG', 'CTC', 'CTT', 
                           'CTA', 'CGT', 'AGA', 'GTA', 'GTC', 'GTT', 'GAA']
            if self.label_mode == 'multi_er': #this is the labeling strategy used in the paper
                #keeps only ALA, ARG, CYS, GLY, ILE, LEU, PRO, SER
                #drops codons with intermediate elongation rates (normalized-by-AA ER values of 0.35<_<0.65)
                #codons with ER < 0.85 = slow, codons with ER > 0.85 are fast
                #drops codons with very inconsistent relative tAI across 14 species (both slow and fast values)
                #exception made for CTC (LEU) - labeled as slow even though ER is 0.91
                slow_codons = ['GCG','CGA','CGG', 'TCG', 'AGC', 'GGG', 'GGA', 'GGC', 'TGC', 
                               'ATA', 'CTC', 'CTG', 'CCG', 'CCC']
                fast_codons = ['GCC','GCT', 'AGA', 'TCT', 'TCC', 'GGT', 'TGT', 
                               'ATC', 'ATT', 'TTG', 'CCT', 'CCA']
            print(f'Fast codons are: {fast_codons}.\nSlow codons are: {slow_codons}')
            with open(self.hdf5_fname.split('.')[0]+'_meta.json', 'r') as f:
                codon_to_int_label = json.load(f)['codon'] 
                self.slow_labels = torch.tensor([codon_to_int_label[x] for x in slow_codons], dtype=torch.int64)
                self.fast_labels = torch.tensor([codon_to_int_label[x] for x in fast_codons], dtype=torch.int64)
            
        self.weight_strategy = 'undersample'
        assert(self.weight_strategy in ['undersample','oversample','equal'])
        #equal - total weight per gene/aa = number of codons included
        #undersample - total weight per gene/aa = 2*number of underrepresented codons
        #oversample - total weight per gene/aa = 2*number of overrepresented codons
        print(f'Weight strategy is', self.weight_strategy)

        print(f"Dataset initilalized, binary is {self.binary}, label mode is {self.label_mode}, th is {(self.rel_tai_low_th, self.rel_tai_high_th)}")
        print(f"balance by aa is {self.balance_by_aa}, balance type is {self.balance_type}, weight by aa is {self.weight_by_aa}, has weights is {self.has_weights}, filter_by_aa is {self.filter_by_aa}")
        print(f"Bin labels is {self.bin_labels}")
        print(f'Features is {self.features}, index rows is {self.num_index_rows}')
        print(f'Names 1d: {self.names_1d}, only_1d is {self.only_1d}, names 2d is {self.names_2d}')
        print(f'Load names: {self.load_names}')
        print(f'At pos names: {self.at_pos_names}')
        print(f'mask values {self.mask_values}')
        print(f'pad values {self.pad_values}')
        print(f'Num classes {self.num_classes}')

        
    def __len__(self):
        return len(self.genes)

    def convert_to_bins(self, labels):
        if self.binary and not self.balance_by_aa: #not balancing by aa later so 
            #everything that isn't slow is 'fast'
            binned = torch.ones(labels.shape, dtype=torch.int64)
            binned[labels <= self.rel_tai_low_th] = 0 #slow is slow, rest are fast
            return binned
        else:
            binned = torch.full(labels.shape, 2, dtype=torch.int64) #neither - set to 2. If binary, filter out later
            binned[labels <= self.rel_tai_low_th] = 0
            binned[labels >= self.rel_tai_high_th] = 1
            return binned
        
    def convert_to_slow(self, codons_at_pos):
        if self.binary and not self.balance_by_aa:  #not balancing by aa later so
            #everything that isn't slow is 'fast'
            converted = torch.ones(codons_at_pos.shape, dtype=torch.int64)
            converted[torch.isin(codons_at_pos, self.slow_labels)] = 0
            return converted
        else:
            converted = torch.full(codons_at_pos.shape, 2, dtype=torch.int64) #neither - set to 2. If binary, filter out later
            converted[torch.isin(codons_at_pos, self.slow_labels)] = 0 #slow
            converted[torch.isin(codons_at_pos, self.fast_labels)] = 1 #fast
            return converted
    
    def __getitem__(self, idx):
        #return input, output for the next gene
        gene = self.genes[idx]
        with h5py.File(self.hdf5_fname, "r") as f:
            loaded = {}
            for name in self.load_names:
                try:
                    loaded[name] = torch.tensor(f[gene+"/"+name][...])
                except Exception as e:
                    print(e)
                    msg = "Could not find "+str(name)+" for "+str(gene)+", gene idx "+str(idx)
                    print(msg)
                    raise ValueError(msg)

        if self.pos_name == 'all': #no saved positions, use all codons that fit balance/filter criteria
            loaded[self.pos_name] = torch.arange(loaded["codon_seq"].shape[0] - 1) 
            #don't include stop codon as a position

        if self.apply_pca:
            loaded[self.esm_name] = torch.matmul(loaded[self.esm_name] - self.pca_mean, self.pca_components.T)
        
        if self.validate:
            assert(loaded["codon_seq"].shape == loaded["aa_seq"].shape)
            assert(min(loaded["codon_seq"].shape[0], self.max_pos+2) == loaded[self.esm_name].shape[0]+1) #esm rep does not include stop codon 
            assert(loaded["codon_seq"].shape == loaded[self.tai_name].shape)
        #ensure positions threshold
        loaded[self.pos_name] = loaded[self.pos_name][loaded[self.pos_name] <= self.max_pos]
        #[v] = can vary, [o] = optional
        #select by positions (aa_seq, tai_seq[v], esm_rep[o]) -> aa_label, tai_label, esm_rep
        for name in self.at_pos_names:
            if name == self.esm_name and self.avg_esm:
                try:
                    loaded[self.esm_name+'_at_pos'] = torch.mean(loaded[self.esm_name], 0, True).expand(len(loaded[self.pos_name]), -1) 
                except Exception as e:
                    print('got an exception')
                    print(e)
                    print('shape of esm', loaded[self.esm_name].shape)
                    print('type of esm', type(loaded[self.esm_name]))
                    print('dtype of esm', loaded[self.esm_name].dtype)
                    raise(ValueError('failed avg esm'))
            else:
                try:
                    loaded[name+"_at_pos"] = loaded[name][loaded[self.pos_name]]
                except Exception as e:
                    print('got an exception')
                    print(e)
                    print(f'{name} failed, with {self.pos_name}, at idx {idx}')
                    raise(ValueError('Failed to get "at_pos" '))
        #mask by positions (codon_seq, tai_seq[v]) 
        for name in self.mask_values:
            try:
                loaded[name][loaded[self.pos_name]] = self.mask_values[name]
            except Exception as e:
                print(e)
                print(name, self.pos_name, idx)
                print(self.mask_values[name])
                print(loaded[self.pos_name])
                raise ValueError('couldnt mask positions')
        #assemble into windows by positions (codon_seq, aa_seq, tai_seq[v])
        w = self.window_size // 2 #assumes window size is even
        pos_idx = torch.arange(self.window_size)[None,:]+loaded[self.pos_name][:, None]
        for name in self.pad_values:
            padding = torch.full((w,), self.pad_values[name])
            loaded[name] = torch.hstack((padding, loaded[name], padding))
            #previously had w+1 to account for stop codon, but now they should be ==
            
            loaded[name] = loaded[name][pos_idx]
        #apply one hot encoding (codon_seq, aa_seq, tai_seq[v], species, aa_seq_label)
        #reshape species
        
        loaded["species"] = torch.full(loaded[self.pos_name].shape, loaded["species"])
        
        
        #apply bins on labels (this is for binning by tai thresholds)
        if self.bin_labels:
            y = loaded[self.label]
            loaded[self.label] = self.convert_to_bins(y) 

        #change to codon
        if 'er' in self.label_mode:
            loaded[self.label] = self.convert_to_slow(loaded['codon_seq_at_pos'])


        if (not self.filter_by_aa is None) and (not self.balance_by_aa):
            new_idx = np.where(np.isin(loaded['aa_seq_at_pos'], self.filter_by_aa))[0]

        if self.weight_by_aa: #weight slow/fast equally by amino acid, for the gene
            assert(self.binary) #weight by AA currently only implemented for binary. 
            assert(not self.balance_by_aa)
            #set all weights to 0, and then change weights for valid positions
            #IF either slow or fast category does not exist, set all weights to 0 for that aa
            weights = torch.zeros(loaded['aa_seq_at_pos'].shape) #same shape as aa labels
            if self.filter_by_aa is not None:
                aa_list = self.filter_by_aa #no need to go through all aa, only those being used
            else:
                aa_list = loaded['aa_seq_at_pos'].unique()
            for aa in aa_list: 
                fast_idx = np.where((loaded['aa_seq_at_pos'] == aa) & (loaded[self.label] == 1))[0] #integer index
                slow_idx = np.where((loaded['aa_seq_at_pos'] == aa) & (loaded[self.label] == 0))[0] 
                if len(fast_idx) == 0 or len(slow_idx) == 0:
                    slow_weight = 0
                    fast_weight = 0
                else:
                    if self.weight_strategy == 'equal':
                        slow_weight = (len(fast_idx)+len(slow_idx))/2/len(slow_idx) 
                        fast_weight = (len(fast_idx)+len(slow_idx))/2/len(fast_idx)
                    elif self.weight_strategy == 'undersample':
                        num_under = min(len(fast_idx), len(slow_idx))
                        slow_weight = num_under/len(slow_idx)
                        fast_weight = num_under/len(fast_idx)
                    elif self.weight_strategy == 'oversample':
                        num_over = max(len(fast_idx), len(slow_idx))
                        slow_weight = num_over/len(slow_idx)
                        fast_weight = num_over/len(fast_idx)
                    #sum(weights) = slow_weight*len(slow_idx) + fast_weight*(len(fast_idx)) = len(fast_idx)+len(slow_idx) = num_samples
                    #slow_weight*len(slow_idx) = fast_weight*(len(fast_idx))
                weights[slow_idx] = slow_weight
                weights[fast_idx] = fast_weight
            new_idx = np.where(weights != 0)[0] #wherever weight is not zero

        if self.balance_by_aa:
            new_idx = np.array([], dtype=int)
            if self.filter_by_aa is not None:
                aa_list = self.filter_by_aa #no need to go through all aa
            else:
                aa_list = loaded['aa_seq_at_pos'].unique() 
            if self.verbose:
                print('Balancing by aa', len(aa_list))
                print(aa_list)
                print('Length of sequence', len(loaded['aa_seq_at_pos']))
                print(f'Lets print the label of type {loaded[self.label].dtype}')
                print(loaded[self.label][0:10]) #just print a part
            for aa in aa_list: 
                assert(self.binary) #balance by AA currently only implemented for binary. 
                fast_idx = np.where((loaded['aa_seq_at_pos'] == aa) & (loaded[self.label] == 1))[0] #integer index
                slow_idx = np.where((loaded['aa_seq_at_pos'] == aa) & (loaded[self.label] == 0))[0]
                if self.verbose:
                    print(f'for aa {aa} slow {len(slow_idx)}. fast {len(fast_idx)}')
                #remove if empty or unneeded?
                if self.balance_type == 'undersample':
                    n_to_keep = min(len(fast_idx), len(slow_idx))
                    fast_idx = np.random.choice(fast_idx, n_to_keep, replace=False)
                    slow_idx = np.random.choice(slow_idx, n_to_keep, replace=False)
                elif self.balance_type == 'oversample':
                    if min(len(fast_idx), len(slow_idx)) == 0:
                        #if either is empty, don't use this aa
                        slow_idx = np.empty((0,))
                        fast_idx = np.empty((0,))
                    else:
                        n_to_keep = max(len(fast_idx), len(slow_idx))
                        #only resample rarer label
                        if len(fast_idx) < n_to_keep:
                            fast_idx = np.random.choice(fast_idx, n_to_keep, replace=True)
                        if len(slow_idx) < n_to_keep:
                            slow_idx = np.random.choice(slow_idx, n_to_keep, replace=True)
                else:
                    raise ValueError('invalid balance type')
                if self.unmixed:
                    both_idx = np.concatenate((fast_idx, slow_idx)) 
                else:
                    both_idx = np.random.permutation(np.concatenate((slow_idx, fast_idx)))
                new_idx = np.concatenate((new_idx, both_idx))
                if self.verbose:
                    print(f'current length new idx {len(new_idx)}')
            if self.positions_in_order:
                new_idx = np.sort(new_idx) #put positions in order

        if self.filter_by_aa is not None:
            #if 'aa_seq_at_pos' in loaded:
            loaded['aa_seq_at_pos'] = self.aa_replacement[loaded['aa_seq_at_pos']]

        if self.include_index:
            spec = torch.clone(loaded["species"]) #save before one hot encoding
            cs = loaded["codon_seq_at_pos"] 
            truth = loaded[self.label]

        for key in self.change_species:
            loaded["species"][loaded["species"] == key] = self.change_species[key]


        if self.verbose:
            for name in self.num_classes:
                print(f"\n{name} has {self.num_classes[name]} classes")
                print(f"{np.unique(loaded[name])}") 

        
        for name in self.num_classes:
            loaded[name] = torch.nn.functional.one_hot(loaded[name], num_classes = self.num_classes[name])
        # concatenate 2d: codon_seq, aa_seq, tai_seq ... 1d: species, aa_label, esm_rep ... label: tai_label
        # some reshaping
        loaded[self.pos_name] = torch.reshape(loaded[self.pos_name], (-1,1)) #to concat with 1d
        if self.tai_name in loaded:
            loaded[self.tai_name] = torch.reshape(loaded[self.tai_name],  #to concat with 2d
                                    (loaded[self.tai_name].shape[0], loaded[self.tai_name].shape[1], 1)) 
        
        if self.include_index: #save before normalizing
            pos = torch.clone(loaded[self.pos_name])
        #normalize by max
        loaded[self.pos_name] = loaded[self.pos_name]/self.max_pos
        
        reps_1d = torch.hstack([loaded[x] if x!=self.label else torch.reshape(loaded[x], (-1,1)) for x in self.names_1d])

        if self.weight_by_aa:
            #last column of reps_1d is weights
            reps_1d = torch.hstack((reps_1d, torch.reshape(weights, (-1,1))))
        #this is to include the index of the gene - drop during training, but can be used to recover
        #the position when looking at results
        if self.include_index:
            #include index, position, species, codon
            spec = torch.reshape(spec, (-1,1))
            cs = torch.reshape(cs, (-1,1))
            truth = torch.reshape(truth, (-1,1))
            ind = torch.full(loaded[self.pos_name].shape, idx)
            #pos is already reshaped
            reps_1d = torch.hstack((ind, pos, spec, cs, truth, reps_1d))

        reps_2d = torch.concatenate([loaded[x] for x in self.names_2d], axis=2)

        #rep_1d is already set
        if self.add_noise:
            n_to_flip = int(self.flip * len(self.label))
            idx_to_flip = np.random.choice(len(self.label), n_to_flip, replace=False)
            loaded[self.label][idx_to_flip] = 1-loaded[self.label][idx_to_flip]
        labels = loaded[self.label]

        if self.balance_by_aa or (self.filter_by_aa is not None):
            #this throws out all 'neither' positions btw
            labels = labels[new_idx]
            reps_1d = reps_1d[new_idx]
            reps_2d = reps_2d[new_idx]

        if self.binary:
            labels = torch.reshape(labels, (-1,1)) #expects 2d vector....?
            labels = labels.to(torch.float32)
            assert(not (2 in labels)), "not binary?"
        else:
            labels = labels.long() #need integer?

        reps_1d = reps_1d.type(torch.float32)
        reps_2d = reps_2d.type(torch.float32)
        if self.verbose:
            print("Including index?", self.include_index)
            print("Shapes (1,2,labels)", reps_1d.shape, reps_2d.shape, labels.shape)
        if self.only_1d:
            return reps_1d, labels
        if self.validate:
            assert(reps_1d.shape[0]==len(loaded[self.pos_name]))
            assert(reps_1d.shape[1] == 65+21+1280+1) #assumes esm
            assert(reps_2d.shape[0]==len(loaded[self.pos_name]))
            assert(reps_2d.shape[1]==self.window_size)
            assert(reps_2d.shape[2]==66+21+1)
            assert(labels.shape[0] == len(loaded[self.pos_name]))
            if self.bin_labels:
                assert(labels.shape[1]==3)
        return (reps_1d, reps_2d), labels


'''
Create an hdf5 file that will have a path for each gene of interest, and all datasets of interest
for each gene. 

Datasets include:
["codon_seq", "aa_seq", "species", "tai_seq", "rel_tai_seq", "positions"] esm_rep
codon_seq: sequence of codons (integer labels)
aa_seq: sequence of amino acids (integer labels)
species: integer representing species that gene belongs to
tai_seq: sequence of tAI values for each codon (float64, 0.5 if no tai info) 
rel_tai_seq: sequence of relative (to a given amino acid) tAI values for each codon (float64, 0.5 if no tai info)
    note: use this as an aa-indepenent measure of tAI
positions: the first set of conserved positions that I chose to look at
esm_rep: the esm-1b representations (1280 features per position) of the sequence

This hdf5 file is used to store the data in a readily accessible form
'''
class HDF5_maker():
    
    def __init__(self, data_db, data_dir=DATA_DIR, hdf5_fname='test.hdf5', skip_tai=False,
                 aa_fasta ='test_aa_fasta.fa', esm_save_dir='esm_save_dir',
                 erase_existing=False, extra_features=['positions']): #mask name is positions
        
        self.skip_tai = skip_tai
        self.hdf5_fname = data_dir+hdf5_fname 
        self.genes = data_db.index #load list of genes
        self.data_db = data_db #load db contains the sequences
        self.extra_features = extra_features
        #TODO: avoid hardcoding directories
        self.tai_db = pd.read_csv("../data/reference/many_yeast_tai.csv", index_col = 0)  #load db that contains the tAI values
        self.rel_tai_db = pd.read_csv("../data/reference/many_yeast_rel_tai.csv", index_col = 0) #load db that contains the normalized-by-AA tAI values
        self.codon_to_aa_encoder = self.make_codon_to_aa_encoder()
        self.aa_encoder = self.make_aa_encoder()
        self.codon_encoder = self.make_codon_encoder()
        self.tai_encoders = {spec:self.make_tai_encoder(spec, self.tai_db) for spec in self.tai_db.columns}
        self.rel_tai_encoders = {spec:self.make_tai_encoder(spec, self.rel_tai_db) for spec in self.rel_tai_db.columns}
        if not self.skip_tai:
            self.species_dict = {self.tai_db.columns[i]:i for i in range(len(self.tai_db.columns))}
        else:
            self.species_dict = {spec:i for i, spec in enumerate(sorted(list(data_db.species.unique())))}
        print('Number of species:', len(self.species_dict))
        #make a fasta file -> for running ESM
        self.mask_positions = False
        self.make_aa_fasta = True
        self.run_esm = True
        self.aa_fasta_file = data_dir+ aa_fasta
        self.esm_save_dir = data_dir+ esm_save_dir
        self.erase_existing = erase_existing
        #erase fasta file automatically only if we call 'run'
        if self.erase_existing:
            for path in (self.hdf5_fname, self.aa_fasta_file):
                if os.path.isfile(path):
                    os.remove(path)
            if os.path.exists(self.esm_save_dir):
                shutil.rmtree(self.esm_save_dir)   

       

        
                       
    #expecting rel_tai_db (relative tai db)
    def make_tai_encoder(self, species, tai_db):
        tai_dict = {x:tai_db[species][x] for x in tai_db[species].index
                    if tai_db[species][x]!=np.nan} #don't include nan values
        tai_encoder = np.vectorize(lambda x: tai_dict.get(x, 0.5)) #return 0.5 for everything that doesn't have a value
        return tai_encoder
            
    def __len__(self):
        return len(self.genes)
    
    def make_aa_encoder(self):
        aa_labels = "XMKVELCSFGYIPHRADQNWT" #0, X, is blank
        aa_dict = {aa_labels[i]:i for i in range(len(aa_labels))}
        aa_dict["B"] = aa_dict["X"]
        aa_dict["U"] = aa_dict["X"]
        aa_dict["*"] = aa_dict["X"]
        self.aa_dict = aa_dict
        aa_encoder = lambda x: aa_dict[x]
        aa_encoder = np.vectorize(aa_encoder)
        return aa_encoder
    
    def make_codon_to_aa_encoder(self):
        c_to_aa =   {'TGT': 'C', 'TGC': 'C', 'GAT': 'D', 'GAC': 'D', 
         'TCT': 'S', 'TCG': 'S', 'TCA': 'S', 'TCC': 'S', 'AGC': 'S', 'AGT': 'S', 
         'CAA': 'Q', 'CAG': 'Q', 'ATG': 'M', 'AAC': 'N', 'AAT': 'N', 
         'CCT': 'P', 'CCG': 'P', 'CCA': 'P', 'CCC': 'P', 'AAG': 'K', 'AAA': 'K', 
         'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'ACT': 'T', 
         'TTT': 'F', 'TTC': 'F', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A', 
         'GGT': 'G', 'GGG': 'G', 'GGA': 'G', 'GGC': 'G', 'ATC': 'I', 'ATA': 'I', 'ATT': 'I', 
         'TTA': 'L', 'TTG': 'L', 'CTC': 'L', 'CTT': 'L', 'CTG': 'L', 'CTA': 'L', 'CAT': 'H', 'CAC': 'H', 
         'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R', 'AGG': 'R', 'AGA': 'R', 
         'TGG': 'W', 'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 
         'GAG': 'E', 'GAA': 'E', 'TAT': 'Y', 'TAC': 'Y', 
         'TAG': 'X', 'TAA':'X', 'TGA':'X', 'MSK':'X'} #stop as X, any invalid value as X
        c_to_aa_encoder = np.vectorize(lambda x: c_to_aa.get(x, 'X'))#return X if it's not there
        return c_to_aa_encoder
    
    def save_meta(self):
        meta_name = self.hdf5_fname.split('.')[0]+"_meta.json"
        meta_info = {'species':self.species_dict, 'aa':self.aa_dict, 'codon':self.codon_dict}
        json_object = json.dumps(meta_info)
        with open(meta_name, "w") as f:
            f.write(json_object)
        
    def make_codon_encoder(self):
        nts = ['A','T','G','C']
        codons = [x + y + z for x in nts for y in nts for z in nts]
        codon_dict = {codons[i]:i+2 for i in range(len(codons))}
        codon_dict["PAD"] = 0
        codon_dict["MSK"] = 1
        self.codon_dict = codon_dict
        self.codon_vals = np.array(list(self.codon_dict.keys()))
        codon_encoder = lambda x: codon_dict.get(x, 65)#return symbol for 'MSK' if it's not there
        codon_encoder = np.vectorize(codon_encoder)
        return codon_encoder
    
    def save_all(self):
        print("Saving", len(self.genes), "genes to", self.hdf5_fname)
        start = time.time()
        c = 0
        for i in self.genes:
            self.save_gene(i)
            c+=1
            if c%(len(self.genes)//10) == 0:
                print(c, "out of", len(self.genes), "complete after", (time.time() - start)/60, "minutes")
                sys.stdout.flush()
        print("Done in ", (time.time() - start)/60, "minutes")

    
    def add_extra_info(self, db_info, feature, dtype='int64'): #will mostly be used for positions
        print(f'Adding extra info: {feature}')
        with h5py.File(self.hdf5_fname, 'a') as f:
            db_info=db_info[~db_info[feature].isna()] #don't try to add none values
            for i in db_info.index:
                assert(i in f), i+" not in file" #check genes already exist
                if i+'/'+feature not in f: #only add info that doen't exist yet
                    dset = f.create_dataset(i+"/"+feature, db_info[feature][i].shape, dtype=dtype)
                    dset[...] = db_info[feature][i]

    def add_esm_info(self, i):
        with h5py.File(self.hdf5_fname, 'a') as f:
            if i+'/esm' in f:
                print(i, "esm already exists")
                return 
            esm = torch.load(self.esm_save_dir+"/"+i+".pt")["representations"][33].numpy()
            dset = f.create_dataset(i+'/esm', esm.shape, dtype='float64') 
            dset[...] = esm
        sys.stdout.flush()
    

    def make_masked_codon_fasta(self, mask_name, savename):
        #make a fasta with codons, with each masked position replaced with [M]
        with open(savename, 'w') as f:
            for i in self.genes:
                codon_seq = np.array(self.data_db.codon_seq[i]).copy() #should already split into groups of 3
                if type(codon_seq) == str:
                    assert(len(codon_seq)%3 == 0)
                    assert(len(codon_seq) > 3)
                    codon_seq = [codon_seq[i*3:i*3+3] for i in range(len(codon_seq)//3)]
                mask_pos = self.data_db[mask_name][i]
                if not mask_pos is np.nan:
                    codon_seq[mask_pos] = '[M]'
                f.write('>'+i+'\n')
                f.write(''.join(codon_seq)+'\n')
                
    def run(self):
        if os.path.isfile(self.aa_fasta_file):
            print(f'Removing existing aa fasta, {self.aa_fasta_file}')
            os.remove(self.aa_fasta_file)
        self.save_meta()
        self.save_all()
        if self.run_esm:
            print("Running ESM")
            sys.stdout.flush()
            #Run 
            #running extract.py from fair-esm
            start = time.time()
            esm_model_name = "esm2_t33_650M_UR50D"
            #need to include extract.py
            print(f'running command {" ".join([sys.executable,"extract.py", esm_model_name, self.aa_fasta_file, self.esm_save_dir, "--repr_layers", "33", "--include", "per_tok"])}')
            process = subprocess.run([sys.executable,"extract.py", esm_model_name, self.aa_fasta_file, 
                                        #sys.executable is the python that this script is using
                            self.esm_save_dir, "--repr_layers", "33", "--include", "per_tok"],
                                      capture_output=True, text=True) 
            print(process.stdout)
            print(process.stderr)
            print("Run finished in", (time.time() - start)/60, "minutes")
            sys.stdout.flush()
            print("Adding ESM info for", len(self.genes), "genes to", self.hdf5_fname)
            start = time.time()
            c = 0
            for i in self.genes:
                self.add_esm_info(i)
                if c%(len(self.genes)//100) == 0:
                    print(c, "out of", len(self.genes), "complete after", (time.time() - start)/60, "minutes")
                    sys.stdout.flush()
                c+=1
            print("Done in ", (time.time() - start)/60, "minutes")
           

    def finish_esm(self, rerun_esm=False, use_esm1b=False):
        #use this if stopped in the middle of run
        esm_model_name = "esm2_t33_650M_UR50D"
        esm_save_dir = self.esm_save_dir
        esm_dataset = '/esm'
        if use_esm1b:
            print('Using esm1b')
            esm_model_name = 'esm1b_t33_650M_UR50S'
            esm_save_dir = esm_save_dir+'_1b/'
            esm_dataset = '/esm_1b'
        if rerun_esm:
            print("New: Finishing remaining: calculating ESM info and adding to file")
            sys.stdout.flush()
            files = subprocess.run(['ls', esm_save_dir], capture_output=True, text=True)
            files = [x[:-3] for x in files.stdout.split()]
            remaining = self.genes[~self.genes.isin(files)]
            print("There are", len(remaining), "proteins remaining to feed to ESM")
            print("Building fasta for new genes - overwrite old fasta")
            sys.stdout.flush()
            with open(self.aa_fasta_file, 'w') as f:
                for gene in remaining:
                    codon_seq = self.data_db.codon_seq[gene] #currently expects already split into 3
                    aa_seq = self.codon_to_aa_encoder(codon_seq[:]) #AA encode codons [ignores that Debaromyces hansenii has a different code]
                    aa_string = "".join(aa_seq)[:-1] #string does not include stop
                    f.write(">"+gene+'\n')
                    f.write(aa_string+'\n') #does not feed stop codon into ESM
            print("Done building fasta")
            print("Running ESM")
            sys.stdout.flush()
            start = time.time()
            process = subprocess.run([sys.executable,"extract.py", esm_model_name, self.aa_fasta_file, 
                                        #sys.executable is the python that this script is using
                            esm_save_dir, "--repr_layers", "33", "--include", "per_tok"],
                                        capture_output=True, text=True) 
            print(process.stdout)
            print(process.stderr)
            print("Run finished in", (time.time() - start)/60, "minutes")
            sys.stdout.flush()
        print("Adding ESM info for", len(self.genes), "genes to", self.hdf5_fname)
        start = time.time()
        c = 0
        with h5py.File(self.hdf5_fname, 'a') as f:
            print("opened hdf5", self.hdf5_fname)
            sys.stdout.flush()
            for i in self.genes:
                if i+esm_dataset in f: #dataset name is "gene/esm" or "gene/esm_1b"
                    pass
                    #print(i, "esm already exists")
                else:
                    esm = torch.load(esm_save_dir+"/"+i+".pt")["representations"][33].numpy()
                    dset = f.create_dataset(i+esm_dataset, esm.shape, dtype='float64')
                    dset[...] = esm
                    #print(i, "succesfully added esm")    
                sys.stdout.flush()
                c+=1
                if c%(len(self.genes)//100) == 0:
                    print(c, "out of", len(self.genes), "complete after", (time.time() - start)/60, "minutes")
                    sys.stdout.flush()
        print("Done in ", (time.time() - start)/60, "minutes")
        if self.erase_existing:
            print("Deleting aa_fasta and tmp folder...")
            os.remove(self.aa_fasta_file)
            shutil.rmtree(esm_save_dir)  
            print("Done")
        
    def save_gene(self, i): #i is the gene
        #rep_names = ["codon_seq", "aa_seq", "species", "tai_seq", "rel_tai_seq", "positions"]
        reps = {}
        codon_seq = self.data_db.codon_seq[i] #currently expects already split into 3, but we could change this
        if type(codon_seq) == str:
            assert(len(codon_seq)%3 == 0)
            assert(len(codon_seq) > 3)
            codon_seq = [codon_seq[i*3:i*3+3] for i in range(len(codon_seq)//3)]
        aa_seq = self.codon_to_aa_encoder(codon_seq[:]) #AA encode codons [ignores that Debaromyces hansenii has a different code]
        aa_string = "".join(aa_seq)[:-1] #string does nOT include stop
        #cut off stop codon
        assert(len(aa_seq) == len(codon_seq)) #aa seq changed to be full length
            
        
        if self.make_aa_fasta:
            with open(self.aa_fasta_file, 'a') as f:
                f.write(">"+i+'\n')
                f.write(aa_string+'\n') #does not feed stop codon into ESM
                
        encoded_codon_seq = self.codon_encoder(codon_seq)
        reps["codon_seq"] = encoded_codon_seq

        for n in self.extra_features:
            reps[n] = self.data_db[n][i]
        if self.mask_positions:
            raise ValueError('currently not supporting masks')
        encoded_aa_seq = self.aa_encoder(aa_seq)
        reps["aa_seq"] = encoded_aa_seq
        species = self.data_db.species[i]
        reps["species"] = np.array(self.species_dict[species])
        if not self.skip_tai:
            tai_encoder = self.tai_encoders[species]
            tai_seq = tai_encoder(codon_seq)
            reps["tai_seq"] = tai_seq
            rel_tai_encoder = self.rel_tai_encoders[species]
            rel_tai_seq = rel_tai_encoder(codon_seq)
            reps["rel_tai_seq"] = rel_tai_seq

        with h5py.File(self.hdf5_fname, 'a') as f:
            for rep in reps.keys():
                dtype = "int64" #16 would be enough btw
                if rep in ["tai_seq", "rel_tai_seq", "esm_rep"]:
                    dtype="float64"
                dset = f.create_dataset(i+"/"+rep, reps[rep].shape, dtype=dtype)
                dset[...] = reps[rep]

'''
Run PCA on a given (1d) dataset (intended use: PCA of ESM)
Save the PCA components and mean for future use.
'''
def run_pca(dataset, n_components=64, name='pca64', save_dir=DATA_DIR+'yeast_species/pca/'):
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_1d, num_workers=10, pin_memory=True, shuffle=False)
    st = time.time()
    pca = sklearn.decomposition.IncrementalPCA(n_components=n_components)
    i = 0
    for x, y in iter(dataloader):
        x = x.numpy()
        pca.partial_fit(x)
        if i % 10 == 0:
            print(f'finished batch {i} of {len(dataloader)}, samples seen = {pca.n_samples_seen_}')
        i+=1
    et = time.time()
    print('')
    print(f'run finished in {(et-st)/60} minutes')
    print(f'total samples: {pca.n_samples_seen_}')
    print(f'Shape mean {pca.mean_.shape} shape components {pca.components_.shape})')
    print(f'saving to {save_dir+name}_components.npy')
    np.save(save_dir+name+'_components.npy', pca.components_)
    np.save(save_dir+name+'_mean.npy', pca.mean_)

records_file = DATA_DIR + 'test_process_data_record.txt' #file to print output to
if __name__ == '__main__':
    with open(records_file, 'w') as stdout_file:
        with redirect_stdout(stdout_file):
            print(os.path.dirname(sys.executable))
            print(torch.__version__)
            print(torch.__file__)

            print("Running new")
            print("Run tests")
            sys.stdout.flush()
            unit_test_hdf5(directory=DATA_DIR+'tmp/')
            print("Test complete")
            sys.stdout.flush()
            
            
            hdf5_name = 'many_yeast.hdf5'
            hdf5_path = DATA_DIR  + hdf5_name
            
            
            sys.stdout.flush()
            data_dir = DATA_DIR +'tmp/'
            
            
            print("Loading dataset")
            print('Dataset is transistive-clusters 70, jan24_many_yeast_db_70_cm1.csv')
            print('Clusters were generated using mmseqs cluster --cluster-mode 1 --min-seq-id 0.7 -c 0.8 -s 7.5')
            data_db = pd.read_csv(DATA_DIR + 'jan24_many_yeast_db_70_cm1.csv', index_col=0) #sacCer3
            data_db['codon_seq'] = data_db.seq.apply(lambda x: [x[i*3:i*3+3] for i in range(len(x)//3)])
            print(f'Making a HDF5 file, data_db size is {len(data_db)}')
            print("Making HDF5 file at", hdf5_path)
            hm = HDF5_maker(data_db = data_db, data_dir=data_dir, hdf5_fname=hdf5_name, erase_existing=False, 
                    aa_fasta="tmp_aa_fasta.fa", esm_save_dir="tmp_esm_save_dir", extra_features=[]) #yeast!
            hm.run()
            print("Done making HDF5 file")


