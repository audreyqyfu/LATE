# Description

Recover Single Cell RNA-seq (scRNA-seq) gene expression profile with Autoencoder, leveraging information of non-linear relationships between genes, from non-zero values in data.

# Installation
This imputation method is written in python, with deep learning models enpowered by tensorflow. 

## CPU version
- install anaconda
  download from https://conda.io/docs/user-guide/install/index.html
  `bash Anaconda-latest-Linux-x86_64.sh`
  `conda create -n py35 python=3.5`
- activate conda environment
  `source activate py35`
- install numpy, pandas, matplotlib, and tensorflow, if not already installed by anaconda automatically
  `conda install numpy pandas matplotlib scipy tensorflow`

## GPU version (only NVIDIA GPU supported)
  https://www.tensorflow.org/install/install_linux
- install GPU version of tensorflow
- install numpy, pandas, matplotlib, scipy

# Usage
- The main program is called `late.py`
- User specific parameters should be modified and put in `params.py`, which contains information about input/output, imputation mode, and machine learning parameters. This parameter file can be renamed.

## Option1: 1step training (LATE: Learning with AuToEncoder)
- 1step: `late.py params.late.py`
  - randomly initialized parameters (weights and biases)
  - directly trained on scRNA-seq (single-cell RNA-seq) dataset
  
## Option2: 2 step training (TRANSLATE: TRANSfer Learning with AuToEncoder) 
- step1: `step1.omega.py`
  - pre-trained Autoencoder on dataset A (bulk RNA-seq reference / huge scRNA-seq reference)
  - autoencoder OMEGA structure 
- step2: 'step2.omega.py'
  - load parameters(weights/biases) pre-trained in step1
  - re-train them on dataset B (scRNA-seq/msk/ds)
  - autoencoder OMEGA structure, excluding zeros from cost function 

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




