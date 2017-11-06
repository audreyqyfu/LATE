# Description
Autoencoder -> transfer learning -> multi-task network 
- step1: pre-trained Autoencoder on dataset A (bulk RNA-seq)
  - autoencoder structure
- step2: re-train the network on dataset B (scRNA-seq)
  - m-task structure
  - traing process only included non-zero (nz) cells for gene-j
  - weights/biases initialized to weights/biases trained in step1

# Workflow
* working dir: **scImpute/one_gene_a_time/2steps3layers/bin/**

## General
### preprocessing (normalization/log-transformation):
- download gene expression matrix (row: genes, column: cells)
  - e.g.: 'All_Tissue_Site_Details.combined.reads.gct'
  
- script: **normalization.logXplus1.py**

- run command: `python -u normalization.logXplus1.py`
  - change 'in_name' and 'out_prefix' in the code
  - code performs: tpm_like_normalization(rescaled back to median cell read counts)
  - code performs: log(x+1) transformation

- select output:
  - **xxx.norm.log.hd5** (normed, log-transformed)(recommended)
  - xxx.norm.hd5 (normed)
  - xxx.csv.gz (csv.gz format, slow, large, better compatability)

### creating simulation single cell RNA-seq dataset
Can down-sample from bulk RNA-seq dataset or other high quality dataset and simulate zero_inflated scRNA-seq dataset
Example Command: `python -u down_sampling.py gtex_v7.norm.hd5 60000 10 gtex_v7.ds_60000_10.hd5`
- first, each sample in data matrix was downsampled to typical scRNA-seq lib-size
- then, additional random zeros introduced to meet the user-defined zero percentage

### step1_training: 
- script: **step1.n_to_n.new.py** (7L, 11/03)
- library: **scimpute.py**
- parameter file: **step1_params.py** (where user change num_nodes, learning_rate...)
1. put the 3 files in the same folder, 
2. edit step1_params.py
3. edit variables run command: `python -u step1.n_to_n.new.7L.py`

### step1_result analysis:
`python -u ./result_analysis.py step1`
  
### step2(To Be Adapted to 7L):
- script: **step2.new.mtask.py** (7L, 11/03)
- library: scimpute.py
- run command: `python -u step2.new.mtask.py`

### step2_result analysis:
`python -u ./result_analysis.py step2`

## On ibest cluster
1. create a step1.slurm file:
```slurm
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
2. create a step1.sh file
```bash
sbatch -p gpu-long --gres=gpu:1 --nodelist=n105 step1.slurm
```
3. login into 'fortyfour.ibest.uidaho.edu', start training with `sh step1.sh`


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
  
** test_flag ** = 1 makes the program runs very fast with a subset of data loaded and few epoch trained




