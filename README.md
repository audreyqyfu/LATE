# scImpute
* transAutoencoder: 
- autoencoder structure
- pre-trained Autoencoder on dataset A (bulk RNA-seq) (step1)
- transfer weights/biases learned to a new alternate training multi-task network
- re-train the network with non-zero (nz) cells for gene-j in 

# workflow (with new version of code, only step1 available now, 10/24/2017)
1. preprocessing:
  - download gene expression matrix (row: genes, column: cells)
  - 'python -u normalization.logXplus1.py'
    - tpm_like_normalization(rescaled back to median cell read counts)
    - log(x+1) transformation
  - outputs:
    - tag.norm.log.hd5 (normed, log-transformed)(for step1, step2)
    - tag.norm.hd5 (normed)
    - xxx.csv.gz (csv.gz format, slow, large, better compatability)
1. step1: 
  - script: scImpute/one_gene_a_time/2steps3layers/bin/step1.n_to_n.new.py (by default 7L, 11/03)
  - modules: scImpute/one_gene_a_time/2steps3layers/bin/scimpute.py
  - command: python -u step1.n_to_n.new.7L.py
  
# input data format
- 
  - so that 'df' contains input_data_matrix [cell, genes]
  - 'df2' should be identical to 'df' in step1
  - example of 'df':
  
  empty|gene1|gene2
  ---|---|---
  cell1|0.392652|0.127627
  cell2|0.377387|0.213198
  
  - function 'scimpute.read_csv' is a good option in implementation

# development environment:
  - python version: 3.5.2 (default, Dec 13 2016, 14:11:32)
  - tf.__version__ 1.2.1





