# Learning with AuToEncoder (LATE) and TRANSfer Learning with AuToEncoder (TRANSLATE)
Due to dropout and other technical limitations in single cell sequencing technologies. Single-cell RNA-seq 
(scRNA-seq) gene expression profile is
highly sparse 
with many zero expression values (typically above 80%, or even 95%). With an Autoencoder traind on 
nonzero values of the data, LATE leverages information of dependence between genes/cells, and recovers the missing values (zeros). With TRANSLATE that allows for transfer learning, the user can train the autoencoder on a reference 
gene expression dataset and then use the weights and biases as initial values for imputing the dataset of interest.

Referece:

Md. Bahadur Badsha, Rui Li, Boxiang Liu, Yang I. Li, Min Xian, Nicholas E. Banovich, Audrey Qiuyan Fu 
Imputation of single-cell gene expression with an autoencoder neural network. bioRxiv 504977; doi: https://doi.org/10.1101/504977.

## Installation
This implementation is written in Python (3.5+) and builds on TensorFlow (1.0-1.4). Additional Python modules needed are:
numpy, pandas, matplotlib, scipy, seaborn, tables, sklearn, MulticoreTSNE

For example, one may use conda or pip to install these modules:
`conda install numpy` or `pip install numpy`

Install `tensorflow` to use CPUs and `tensorflow-gpu` to use GPUs.

## Usage
Note: In the current version of LATE, code described below needs to run where the Python scripts included in this distribution are stored on your computer.  Imputation and analysis will generate folders in the same directory.  Datasets used for imputation or analysis (e.g., the input, the reference, or the ground truth) may be stored elsewhere.
### Running LATE
- Imputation: `python example_late.py -mode='late' -infile='../data/example.msk90.hd5'`
  - `example_late.py` reads in the input data, sets parameters and calls function `late_main` to perform imputation.  
  - Default values for imputation and analysis parameters are provided in `global_params.py`, and may be modified in `example_late.py` (for example, see how `example_late.py` specifies a short run).
### Analyzing imputation results
- After running imputation and generating the folder 'step2/', one may run analysis to summarize and visualize imputation results. 
    `python example_analysis.py -mode='analysis' -infile='../data/example.msk90.hd5'`
### An example of running TRANSLATE
- Step 1: `python example_translate_step1.py -mode='pre-training' -infile='../data/ref_example.mdk50.hd5'`
  - This step runs LATE on the reference data `ref_example.mdk50.hd5`.
  - This step creates folder `step1/` and stores the output.
- Step 2: `python example_translate_step2.py -mode='translate' -infile='../data/example.msk90.hd5'`
  - This step runs LATE on the input data `example.msk90.hd5`.
  - This step creates folder `step2/` and stores the output.
- Note: `example_translate_step1.py` and `example_translate_step2.py` are identical in this example, although the parameters used for training may be specified differently. 

## Additional details
### Mode: 
- `late`:
  - Random initialization;
  - Trains the autoencoder on the input dataset (no reference data);
  - Generates folder `step2/`.
- `pre-training`:
  - Step 1 of TRANSLATE;
  - Random initialization;
  - Trains the autoencoder on the reference dataset.
  - Generates folder `step1/`.
- `translate`:
  - Step 2 of TRANSLATE;
  - Uses the results from `pre-training` as initialization;
  - Trains the autoencoder on the input dataset.
  - Generates folder `step2/`.
- `impute`:
  - Uses results from LATE or TRANSLATE (results that are currently stored in the folder `step2/`);
  - Calculates imputed values for the input dataset without training.
- `analysis`:
  - Uses imputation results in `step2` for summary and visualization;
  - Generates folder `Eval/`.
  
### Input data
- A data matrix of sequencing read counts, with row names (cell IDs) and column names (gene IDs) in one of the following formats:
    - csv: comma seperated values.
    - tsv: tab seperated values.
    - h5: 10x Genomics sparse matrix:
        - https://support.10xgenomics.com/single-cell-gene-expression/datasets
        - https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices

- example of 'df':
  
  empty|gene1|gene2
  ---|---|---
  cell1|0.392652|0.127627
  cell2|0.377387|0.213198
    
- The input data will be transformed with log10(count+1) for imputation.

### Output files

Imputed data matrix is in the hd5 format and stored in `step2/imputation.step2.hd5`.  Values in this matrix have the same layout as the input, and are on the log10(count+1) scale.

### Ground truth
The ground truth data should also be a matrix of read counts with the same format as the input.  

### Parameters
Default parameters are specified in `global_params.py`.  They may be modified in `example.py`.  See `example.py` for examples of modifications.  



