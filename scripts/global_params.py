# -*- coding: utf-8 -*-

# MODE
mse_mode = 'mse_nz'  # mse, mse_nz
mode = 'late'  # takes pre-training, translate, late, impute for late.py; takes 'scVI/DCA/anything' for
# result_analysis.py

# Input (should contain index and header)
fname_input = '../data/pbmc.g949_c10k.msk90.hd5'  # csv/csv.gz/tsv/h5/hd5 formats supported
name_input = 'example'
ori_input = 'cell_row'  # cell_row/gene_row
transformation_input = 'log10'  # as_is/log10/rpm_log10/exp_rpm_log10
genome_input = 'mm10'  # only for sparse matrix h5 data from 10x Genomics

# Output
fname_imputation = './step2/imputation.step2.hd5'  # do not modify for pre-training
name_imputation = '{}_({})'.format(name_input, mode)  # recommend not to modify
ori_imputation = 'cell_row'  # gene_row/cell_row
transformation_imputation = 'as_is'  # log10/rpm_log10/exp_rpm_log10
tag = 'Eval'  # folder name for analysis results

# Ground Truth
fname_ground_truth = fname_input
name_ground_truth = name_input
ori_ground_truth = ori_input
transformation_ground_truth = transformation_input

# Pre-defined cluster for PCA/tSNE visualization
# Format: cluster_info.csv with cell_id as index and cluster_id as 1st column
cluster_file = None  # if you don't have cluster information, no coloring of tSNE
# cluster_file = './data/exampleclusters.csv'  # coloring cells in tSNE plots with clusters defined here

# DATA SPLIT PROPORTION
[a, b, c] = [0.7, 0.15, 0.15]  # train/valid/test  todo: use 100% or 85% for real biological application

# HYPER PARAMETERS
# Model structure
L = 5  # num of layers for Autoencoder, accept: 3/5/7
l = L//2  # inner usage, do not modify
n_hidden_1 = 800  # needed for 3/5/7 layer design
n_hidden_2 = 400  # needed for 5/7 layer design
# n_hidden_3 = 200 # needed for 7 layer design

# SD for rand_init of weights
sd = 1e-3  # for rand_init of weights. 3-7L:1e-3, 9L:1e-4

# Mini-batch learning parameters
batch_size = int(256)  # mini-batch size for training
sample_size = int(1000)  # sample_size for learning curve, slow output
large_size = int(1e5)  # if num-cells larger than this, use slow but robust method for imputation and output

# Length of training
max_training_epochs = int(20)  # num_mini_batch / (training_size/batch_size)
display_step = int(5)  # interval on learning curve, 20 displays recommended
snapshot_step = int(50)  # interval of saving session, saving imputation
patience = int(3)  # early stop patience epochs, just print warning, no real stop

# Regularization
pIn = 0.8  # dropout rate at input layer, default 0.8
pHidden = 0.5  # dropout rate at hidden layers, default 0.5
reg_coef = 0.0  # reg3=1e-2, set to 0.0 to desable it

# For development usage
seed_tf = 3
test_flag = False  # [True, False]
run_flag = 'rand_init'
learning_rate = 3e-4

# choose gene columns/names for scatter plot
gene_pair_list = [
	# Index
	[2, 3],
	# Gene names
	# ['ENSG00000173372', 'ENSG00000087086'],
]

