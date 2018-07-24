# Description

Recover Single Cell RNA-seq (scRNA-seq) gene expression profile with Autoencoder, leveraging information of non-linear
 relationships between genes, learned from non-zero values in data.

# Installation
This imputation method is written in python (3.5), with deep learning models enpowered by tensorflow. 

## CPU version
- install anaconda
  - download from https://conda.io/docs/user-guide/install/index.html
  - `bash Anaconda-latest-Linux-x86_64.sh`
  - `conda create -n py35 python=3.5`
- activate conda environment
  - `source activate py35`
- install numpy, pandas, matplotlib, and tensorflow, if not already installed by anaconda automatically
  - `conda install numpy pandas matplotlib scipy tensorflow`

## GPU version (only NVIDIA GPUs supported)
- https://www.tensorflow.org/install/install_linux
- install GPU version of tensorflow
- install numpy, pandas, matplotlib, scipy

# Usage
- The main program is called `late.py`
- User specific parameters should be modified and put in `params.py`, 
which contains information about input/output, imputation mode, and machine learning parameters. 
This parameter file can be renamed.
- example usage: `python3 late.py params.py`

## Option1: 1step training (LATE: Learning with AuToEncoder)
- 1step: `late.py params.late.py`
  - randomly initialized parameters (weights and biases)
  - directly trained on scRNA-seq (single-cell RNA-seq) dataset
  
## Option2: 2 step training (TRANSLATE: TRANSfer Learning with AuToEncoder) 
- step1: `late.py params.pre_training.py`
  - pre-trained Autoencoder on reference dataset (bulk RNA-seq reference / huge scRNA-seq reference)
  - parameter modified `mode = 'pre-training'` 
- step2: `late.py params.translate.py`
  - load parameters(weights/biases) pre-trained in step1
  - re-train them on dataset B (scRNA-seq expression profile of interest)
  - parameter modified `mode = 'translate'` 


# Example Workflow
* work dir: **./script/**

## 1. Load data

### Input format
- csv/csv.gz/tsv/h5/hd5 formats supported
    - csv: comma seperated values in text format
    - tsv: tab seperated values in text format
    - h5: 10x genomics sparse matrix format 
        - https://support.10xgenomics.com/single-cell-gene-expression/datasets
        - https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices
    - hd5: output of `late.py`, compressed hd5 format

- data orientation:
    - both cell_row/gene_row are supported, just specify matrix orientation in `ori_input/ori_ground_truth` in `params.py`
    - inside the code, data are transformed into cell_row matrix
    
- Data transformation
    - as_is/log/rpm_log/exp_rpm_log
    - as_is: no transformation
    - log: log10(x+1)
    - rpm_log: log(rpm+1), rpm (reads per million)
    - exp_rpm_log: 10^x - 1, to reverse the effect of log10(x+1), usually only useful for testing purposes

- `fname_input` and `fname_ground_truth`
    - `fname_input` specifies input file, on which the model is trained
    - `fname_ground_truth` is for evaluation purposes, when ground_truth is available for simulated dataset. For 
    imputation purposes without ground-truth, this parameter should be set the same to `fname_input` 

### Pre-processing for h5 format data
- Download gene expression matrix:
  - e.g.: 'All_Tissue_Site_Details.combined.reads.gct' (GTEx)

- Filtering data:
  - `./data_preprocessing/data_filter_stat.py` (min-reads/cell, min-reads/gene, histogram of reads/gene reads/cell)
  - `./data_preprocessing/data_gene_selection.py` (select genes from datasets )
  - `./data_preprocessing/data_sample_selection.py` (select cells)
  
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




