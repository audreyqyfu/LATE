# Description
You can read readme.terse.md first
## Option1: transfer learning (step1 -> TL -> step2) 
- step1: `step1.omega.py`
  - pre-trained Autoencoder on dataset A (bulk RNA-seq reference / huge scRNA-seq reference)
  - autoencoder OMEGA structure 
- step2: 'step2.omega.py'
  - load parameters(weights/biases) pre-trained in step1
  - re-train them on dataset B (scRNA-seq/msk/ds)
  - autoencoder OMEGA structure, excluding zeros from cost function 

## Option2: 1step training
- 1step: `step1.omega.py`
  - randomly initialized parameters
  - directly trained on scRNA-seq/msk/ds dataset

# Workflow
* working dir: **scImpute/bin/**

## Pre-processing (normalization/log-transformation):
- Download gene expression matrix:
  - e.g.: 'All_Tissue_Site_Details.combined.reads.gct' (GTEx)

- Filtering data:
  - `data_filter_stat.py` (min-reads/cell, min-reads/gene)
  - `data_gene_selection.py` (select genes from reference datasets, so that )
  - `data_sample_selection.py` (select cells ...)
  
- Normalization: 

- Output:
  - **xxx.norm.log.hd5** (normed, log-transformed)(recommended)
  - xxx.norm.hd5 (normed)
  - xxx.csv.gz (csv.gz format, slow, large, better compatability)

## Creating simulated scRNA-seq data
- msk
- ds
    - Can down-sample from bulk RNA-seq dataset or other high quality dataset 
    and simulate zero_inflated scRNA-seq dataset
    - `python -u data_down_sampling.py gtex_v7.norm.hd5 60000 10 gtex_v7.ds_60000_10.hd5`
    - first, each sample in data matrix was downsampled to typical scRNA-seq lib-size
    - then, additional random zeros introduced to meet the user-defined zero percentage

## step1_training: 
- script: **step1.omega.py**
- library: **scimpute.py**
- parameter file: **step1_params.py** (where user change num_nodes, learning_rate...)
1. put the 3 files in the same folder, 
2. edit step1_params.py
3. edit variables run command: `python -u step1.n_to_n.new.7L.py`

## step1_result analysis:
`python -u ./result_analysis.py step1`
  
## step2_training:
- script: **step2.omega.py** (7L, 11/03)
- library: **scimpute.py**
- parameter file: **step2_params.py** (change num_nodes, learning_rate...)
- step1 output: **./step1/**
- put these 4 files in the same folder

## step2_result analysis:
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
python -u ./step1.xxx.py
python -u ./result_analysis.py step1
echo "*--done--*"
```
2. create a step1.sh file
```bash
sbatch --mem=100G -p gpu-long --gres=gpu:1 --nodelist=n105 step1.slurm
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
  - (num_nodes in bottle-neck) x (hidden_node retain rate) == 
  data dimension after PCA reduce dim
  - learning rate = 3e-4 for 7L, 3e-5 for 9L 
  - rand_init_sd = 1e-4 for 7L, 1e-5 for 9L 
  - test_flag = 1 (fast run: a subset of data and few epochs)




