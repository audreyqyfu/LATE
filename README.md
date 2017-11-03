# scImpute
* transAutoencoder: 
- autoencoder structure
- pre-trained Autoencoder on dataset A (bulk RNA-seq) (step1)
- transfer weights/biases learned to a new alternate training multi-task network
- re-train the network with non-zero (nz) cells for gene-j in dataset B (scRNA-seq) (step2)
  - during re-train, learning rate should be smaller than step1
  - expect slow converge in step2, because of the m-task mechanism

# workflow (with new version of code, only step1 available now, 10/24/2017)
* preprocessing (normalization/log-transformation):
  - download gene expression matrix (row: genes, column: cells)
  - `python -u normalization.logXplus1.py`
    - tpm_like_normalization(rescaled back to median cell read counts)
    - log(x+1) transformation
  - outputs:
    - tag.norm.log.hd5 (normed, log-transformed)(recommended)
    - tag.norm.hd5 (normed)
    - xxx.csv.gz (csv.gz format, slow, large, better compatability)
    
* splitting data (TBA):
  - for large dataset, pre-splitting is preferred
  
* step1: 
  - script: scImpute/one_gene_a_time/2steps3layers/bin/step1.n_to_n.new.py (by default 7L, 11/03)
  - modules: scImpute/one_gene_a_time/2steps3layers/bin/scimpute.py
  - command: `python -u step1.n_to_n.new.7L.py`
  
# input data format
- read hd5 or csv into pandas data-frames
  - 'df' contains input_data_matrix [cell, genes]
  - 'df2' contains ground-truth
  - In step1, 'df2' should be identical to 'df'
  
  - example of 'df':
  
  empty|gene1|gene2
  ---|---|---
  cell1|0.392652|0.127627
  cell2|0.377387|0.213198
  
  - function 'scimpute.read_csv', 'scimpute.read_hd5) are good options in implementation
  - step1 example: 
  `file = "EMT.MAGIC.9k.A.log.hd5"  # input`
  `file2 = "EMT.MAGIC.9k.A.log.hd5"  # ground truth (same as input in step1)`
  `name1 = '(EMT_MAGIC_A)'`
  `name2 = '(EMT_MAGIC_A)'`
  `df = pd.read_hdf(file).transpose()  # [cells,genes]`
  `df2 = pd.read_hdf(file2).transpose()  # [cells,genes]`
  - step2 example: 
  `file = "EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk90.log.hd5"  # input`
  `file2 = "EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"  # ground truth (same as input in step1)`
  `name1 = '(EMT_MAGIC_B.msk90)'`
  `name2 = '(EMT_MAGIC_B)'`
  `df = pd.read_hdf(file).transpose()  # [cells,genes]`
  `df2 = pd.read_hdf(file2).transpose()  # [cells,genes]`

# development environment:
  - python version: 3.5.2 (default, Dec 13 2016, 14:11:32)
  - tf.__version__ 1.2.1


# parameter setting
Here is a good start point for parameter setting
  - (num_nodes in bottle-neck) x (hidden_node retain rate) == data dimension after PCA reduce dim
  - learning rate = 3e-4 for 7L, 3e-5 for 9L 
  - rand_init_sd = 1e-4 for 7L, 1e-5 for 9L 




