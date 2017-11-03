# Description
Autoencoder -> transfer learning -> multi-task network 
- step1: pre-trained Autoencoder on dataset A (bulk RNA-seq)
  - autoencoder structure
- transfer weights/biases learned to a new alternate training multi-task network
- step2: re-train the network on dataset B (scRNA-seq)
  - m-task structure
  - traing process only included non-zero (nz) cells for gene-j

# Workflow
* working dir: **scImpute/one_gene_a_time/2steps3layers/bin/**

## General
* preprocessing (normalization/log-transformation):
  - download gene expression matrix (row: genes, column: cells)
    - e.g.: 'All_Tissue_Site_Details.combined.reads.gct'
  - run command: `python -u normalization.logXplus1.py`
    - tpm_like_normalization(rescaled back to median cell read counts)
    - log(x+1) transformation
    - change 'in_name' and 'out_prefix' in the code
  - select output:
    - tag.norm.log.hd5 (normed, log-transformed)(recommended)
    - tag.norm.hd5 (normed)
    - xxx.csv.gz (csv.gz format, slow, large, better compatability)
  
* step1: 
  - script: **step1.n_to_n.new.py** (7L, 11/03)
  - library: scimpute.py
  - parameter file: step1_params.py (where user change num_nodes, learning_rate...)
  - run command: `python -u step1.n_to_n.new.7L.py`
  
 * step2:
  - script: **step2.new.mtask.py** (7L, 11/03)
  - library: scimpute.py
  
## On ibest cluster
  1. create a step1.slurm file:
  ```
  #!/bin/bash
  #SBATCH --mail-user=rui@uidaho.edu
  #SBATCH --mail-type=BEGIN,END

  echo $(hostname)
  nvidia-smi -L
  source /usr/modules/init/bash
  module load python/3.5.2
  python -u ./step1.n_to_n.new.py
  python -u ./result_analysis.py step1
  echo "*--done--*"
  ```
  2. create 
  
  
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
  ```python
  file = "EMT.MAGIC.9k.A.log.hd5"  # input
  file2 = "EMT.MAGIC.9k.A.log.hd5"  # ground truth (same as input in step1)
  name1 = '(EMT_MAGIC_A)'
  name2 = '(EMT_MAGIC_A)'
  df = pd.read_hdf(file).transpose()  # [cells,genes]
  df2 = pd.read_hdf(file2).transpose()  # [cells,genes]
  ```
  - step2 example: 
  ```python
  file = "EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk90.log.hd5"  # input
  file2 = "EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"  # ground truth
  name1 = '(EMT_MAGIC_B.msk90)'
  name2 = '(EMT_MAGIC_B)'
  df = pd.read_hdf(file).transpose()  # [cells,genes]
  df2 = pd.read_hdf(file2).transpose()  # [cells,genes]
  ```

# development environment:
  - python version: 3.5.2 (default, Dec 13 2016, 14:11:32)
  - tf.__version__ 1.2.1


# parameter setting
Here is a good start point for parameter setting
  - (num_nodes in bottle-neck) x (hidden_node retain rate) == data dimension after PCA reduce dim
  - learning rate = 3e-4 for 7L, 3e-5 for 9L 
  - rand_init_sd = 1e-4 for 7L, 1e-5 for 9L 




