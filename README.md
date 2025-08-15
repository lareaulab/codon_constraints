# Protein language models reveal evolutionary constraints on synonymous codon choice

This repository contains all code used in data analysis, figure creation, and model training and evaluation for the codon constraints project, currently available as a preprint at https://www.biorxiv.org/content/10.1101/2025.08.05.668603v1.

## Setup and installation
Dependencies are listed in the codon_env.yml file.

A conda environment containing required packages can be set up via:

```bash
conda create --name codon_env --file spec_file
conda activate codon_env
pip3 install -r requirements.txt
```

## Replicating Figures and Analysis
Most of the figures and statistics used in the paper can be replicated by running `analysis/main_figures.ipynb`. 
This notebook takes two files as input - `data/results/model_results.csv`, which includes the output of our model on positions from the _S. cerevisiae_ test set, and `data/results/pooled_comp_results.csv`, 
which records the results from our pooled competition experiment. Instructions on how to replicate these files are provided below.
Figure panels are saved in the `figures` folder. Figure panels for Supplementary Figure 3 (Paired Fluorescent Competion) can be replicated by running `paired_competition/paired_competition_analysis.ipynb`.
Code used to train the random forest comparison can be found in `analysis/random_forest.ipynb`.

### Evaluating our model on the _S. cerevisiae_ test dataset
Trained models are stored under `data/models`, including both our main model trained on a balanced dataset (`nov12_24_undersample`) and a comparison model trained on the full, unbalanced dataset (`june18_24_unweighted`).
In order to repeat the evaluation process, run `model/generate_esm_rep_scer.py` to create ESM2 representations (https://github.com/facebookresearch/esm) for all _S. cerevisiae_ genes. 
Then run `model/evaluate.py` to load and evaluate both models on _S. cerevisiae_. Next, call `model/combine_results.py` in order to create `data/results/model_results.csv`. 

### Replicating pooled competition analysis
A library of synonymous yeast variants was grown together in a pooled competition for 81 generations, with two editing replicates (AB and CD) and two experimental replicates per editing replidate (A, B, C, and D). The library was sequenced at the start (T0) and end (T5) of the competition. Raw sequencing data is obtainable from the NCBI Sequence Read Archive via accession number PRJNA1297196, and was processed using `pooled_competition/process_illumina.py`. The raw read counts per variant per sample are stored in `data/pooled_comp/counts_db.csv`.
DESeq2 was used to call significantly advantageous or deleterious variants. This step can be replicated via `pooled_competition/run_deseq2.py`, and requires DESeq2 and R to be installed.
The final results file (`data/results/pooled_comp_results.csv`) is created using `pooled_competition/combine_results.py`.

## Training the model
Trained model weights are saved in `data/models`. In order to replicate the model training process, first generate an HDF5 file containing ESM2 representations of all genes in the training dataset using `model/process_data.py`. 
Model training can be replicated via `model/train_model.py`. Model settings and parameters can be changed within the file. 

## Additional information
Plasmids and oligo sequences can be found under `data/plasmids_and_oligos`. OD measurements from the pooled competition can be found under `data/pooled_comp/pooled_comp_od.csv`.


