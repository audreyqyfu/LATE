# scImpute
# transAutoencoder: autoencoder structure, pre-trained on A, re-train on B.msk
# workflow (with new version of code, only step1 available now, 10/24/2017)
* step1: 
  - script: scImpute/one_gene_a_time/2steps3layers/bin/step1.n_to_n.new.7L.py
  - library: scImpute/one_gene_a_time/2steps3layers/bin/scimpute.py
  - command: python step1.n_to_n.new.7L.py

# development environment:
  - python version: 3.5.2 (default, Dec 13 2016, 14:11:32)
  - tf.__version__ 1.2.1

# input data format
- edit function 'read_data' in 'scimpute.py'
  - so that 'df' contains input_data_matrix [cell, genes]
  - 'df2' should be identical to 'df' in step1
  - example of 'df':
    - _ gene1 gene2
    - cell1 0.392652  0.127627
    - cell2 0.377387 0.213198
   - function 'scimpute.read_csv' is a good option in implementation



