# Learning with AuToEncoder (LATE)
Due to dropout and other technical limitations in single cell sequencing technologies. Single Cell RNA-seq 
(scRNA-seq) gene expression profile is
noisy 
with many zero expression values (typically above 80%, or even 95%). With an Autoencoder traind on 
non-zero values of the data, LATE leverages information of non-linear relationships between genes/cells, and restore 
gene-gene relationships ovscured by those zeros. With TRANSfer Learning with AuToEncoder (TRANSLATE), a reference 
dataset was used to pre-train weights and biases, so that better imputation result can be achieved.



## Installation
This imputation method is written in python (3.5), with deep learning models enpowered by tensorflow. After 
installing python 3.5, Tensorflow 1.0-1.4, and other related python libraries, this code can run correctly.

### CPU version (easier to install, slower to run)
- install anaconda
  - download from https://conda.io/docs/user-guide/install/index.html
  - `bash Anaconda-latest-Linux-x86_64.sh`
  - `conda create -n py35 python=3.5`
- activate conda environment
  - `source activate py35`
- install numpy, pandas, matplotlib, and tensorflow, if not already installed by anaconda automatically
  - `conda install numpy pandas matplotlib scipy tensorflow`

### GPU version (about 5-10 times faster, only NVIDIA GPUs supported)
- install anaconda version of python 3.5 as shown above (for simplicity)
- install GPU version of tensorflow
    - https://www.tensorflow.org/install/install_linux
        - install CUDA first
        - then install tensorflow-gpu
- install numpy, pandas, matplotlib, scipy, seaborn, tables, sklearn, MulticoreTSNE,  as shown above



## Imputation
- example: `python3 -u late.py params.py`
- The main program is called `late.py`
- User specific parameters should be modified and put in `params.py`, 
which contains information about input/output, imputation mode, and machine learning hyper-parameters. 
This file can be renamed.

### Mode: 1step training (LATE: Learning with AuToEncoder)
- 1step: `python3 -u late.py params.late.py`
  - randomly initialized weights and biases
  - directly trained on scRNA-seq (single-cell RNA-seq) dataset with non-zero values in the dataset to find 
  gene-gene/cell-gene relationships for imputation.
  
### Mode: 2 step training (TRANSLATE: TRANSfer Learning with AuToEncoder) 
- step1: `python3 -u late.py params.pre_training.py`
  - pre-trained Autoencoder on reference dataset (bulk RNA-seq reference / huge scRNA-seq reference)
  - parameter modified `mode = 'pre-training'` 
- step2: `python3 -u late.py params.translate.py`
  - load parameters(weights/biases) pre-trained in step1
  - re-train them on dataset B (scRNA-seq expression profile of interest)
  - parameter modified `mode = 'translate'` 


  
## Result Analysis
Generates plots, statistics by comparing imputation results with input and ground-truth (if applicable)

- `python3 -u result_analysis.py params.py`
- `params.py` is the same parameter file used for LATE imputation, or step2 parameter file used in TRANSLATE impuation
    - `tag = 'Eval'`: folder name for analysis results
    - `pair_list = [[0,1], [2,3]]`: index of genes to be put in gene-gene plots
    - `pair_list = [['gene1', 'gene2'], ['gene3', 'gene4']]`: gene names of interest



## Parameters

### Input format
- A simple data matrix with cell_id and gene_id as column/row names, csv/csv.gz/tsv/h5/hd5 formats supported
    - csv: comma seperated values in text format
    - tsv: tab seperated values in text format
    - h5: 10x genomics sparse matrix format 
        - https://support.10xgenomics.com/single-cell-gene-expression/datasets
        - https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices
    - hd5: output of `late.py`, compressed hd5 format
- example of 'df':
  
  empty|gene1|gene2
  ---|---|---
  cell1|0.392652|0.127627
  cell2|0.377387|0.213198
    
- Both gene_row/cell_row matrices accepted, by specifying parameter `ori_input`

### Input parameter settings
```
fname_input = '../data/cell_row/example.msk90.hd5'  # csv/csv.gz/tsv/h5/hd5 formats supported
name_input = 'example'
ori_input = 'cell_row'  # cell_row/gene_row
transformation_input = 'log'  # as_is/log/rpm_log/exp_rpm_log
```
- `fname_input`: path to input_file
- `name_input`: the name shown in plots generated
- `ori_input`: data matrix orientation, 
    - 'cell_row'
    - 'gene_row'    
- `transformation_input`: Data transformation options
    - as_is: no transformation (recommended)
    - log: log10(x+1)
    - rpm_log: log(rpm+1), rpm (reads per million)
    - exp_rpm_log: 10^x - 1, to reverse the effect of log10(x+1), usually only useful for testing purposes

### Output parameter settings
```
fname_imputation = './step2/imputation.step2.hd5'  # do not modify for pre-training
name_imputation = '{}_({})'.format(name_input, mode)  # recommend not to modify
ori_imputation = 'cell_row'  # gene_row/cell_row
transformation_imputation = 'as_is'  # log/rpm_log/exp_rpm_log
tag = 'Eval'  # folder name for analysis results
```
- `fname_imputation`: output file_name, do not modify for `pre-training` mode, because `translate` mode will not find
 pre-training results 

### Ground-truth parameter settings
Ground-truth is used for testing purposes, you can leave settings default (same with parameters for 'input') if you 
don't have ground-truth dataset.
```buildoutcfg
fname_ground_truth = fname_input
name_ground_truth = name_input
ori_ground_truth = ori_input 
transformation_ground_truth = transformation_input
```

### Hyper parameter settings
The Autoencoder and Transfer Learning hypyer-parameter tuning requires some machine learning / deep learning model 
experience. If you are not sure what hyper parameter to use, we strongly recommend to use the default settings in the
 parameter files provided.
 ```buildoutcfg
# HYPER PARAMETERS
# Model structure
L = 5  # num of layers for Autoencoder, accept: 3/5/7
l = L//2  # inner usage, do not modify
n_hidden_1 = 800  # needed for 3/5/7 layer design
n_hidden_2 = 400  # needed for 5/7 layer design
# n_hidden_3 = 200 # needed for 7 layer design

# SD for rand_init of weights
sd = 1e-3  # for rand_init of weights. 3-7L:1e-3, 9L:1e-4
if run_flag == 'rand_init':
    learning_rate = 3e-4  # step1: 3e-4 for 3-7L, 3e-5 for 9L
elif run_flag == 'load_saved':
    learning_rate = 3e-5  # step2: 3e-5 for 3-7L, 3e-6 for 9L
elif run_flag == 'impute':
    learning_rate = 0.0

# Mini-batch learning parameters
batch_size = int(256)  # mini-batch size for training
sample_size = int(1000)  # sample_size for learning curve, slow output
large_size = int(1e5)  # if num-cells larger than this, use slow but robust method for imputation and output

# Length of training
max_training_epochs = int(100)  # num_mini_batch / (training_size/batch_size)
display_step = int(5)  # interval on learning curve, 20 displays recommended
snapshot_step = int(50)  # interval of saving session, saving imputation
patience = int(3)  # early stop patience epochs, just print warning, no real stop

# Regularization
pIn = 0.8  # dropout rate at input layer, default 0.8
pHidden = 0.5  # dropout rate at hidden layers, default 0.5
reg_coef = 0.0  # reg3=1e-2, set to 0.0 to desable it
```



## Inner-usage
### Creating simulated scRNA-seq data
- msk
- ds
    - Can down-sample from bulk RNA-seq dataset or other high quality dataset 
    and simulate zero_inflated scRNA-seq dataset
    - `python -u data_down_sampling.py gtex_v7.norm.hd5 60000 10 gtex_v7.ds_60000_10.hd5`
    - first, each sample in data matrix was downsampled to typical scRNA-seq lib-size
    - then, additional random zeros introduced to meet the user-defined zero percentage

### On ibest cluster
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
  
### development environment:
  - python version: 3.5.2 (default, Dec 13 2016, 14:11:32)
  - tf.__version__ 1.2.1

### parameter setting
Here is a good start point for parameter setting
  - (num_nodes in bottle-neck) x (hidden_node retain rate) == 
  data dimension after PCA reduce dim
  - learning rate = 3e-4 for 7L, 3e-5 for 9L 
  - rand_init_sd = 1e-4 for 7L, 1e-5 for 9L 
  - test_flag = 1 (fast run: a subset of data and few epochs)




