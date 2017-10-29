#!/usr/bin/python

#  todo:
# 1. restore (TL)
# 2. use functions in model
# 3. mtask (one gene a time with one network)
# 4. exclude zeros or not

# import
from __future__ import division  # fix division // get float bug
from __future__ import print_function  # fix printing \n
import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import math
import os
import time
import seaborn as sns
# sys.path.append('./bin')
import scimpute

# print versions / sys.path
print ('python version:', sys.version)
print('tf.__version__', tf.__version__)
print('sys.path', sys.path)

# refresh folder
log_dir = './re_train'
scimpute.refresh_logfolder(log_dir)

# read data
data = 'EMT9k_log_msk90'  # EMT2730 or splatter
if data is 'splatter':  # only this mode creates gene-gene plot
    file = "../data/v1-1-5-3/v1-1-5-3.F3.msk.hd5" #data need imputation
    file_benchmark = "../data/v1-1-5-3/v1-1-5-3.F3.hd5"
    Aname = '(F3.msk)'
    Bname = '(F3)'
    df = pd.read_hdf(file).transpose() #[cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT2730':
    file = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5" #data need imputation
    file_benchmark = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5"
    Aname = '(EMT2730)'
    Bname = '(EMT2730)'
    df = pd.read_hdf(file).transpose() #[cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.hd5"
    Aname = '(EMT9k.B.msk)'
    Bname = '(EMT9k.B)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k_log_msk50':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk50.log.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"
    Aname = '(EMT9kLog_Bmsk50)'
    Bname = '(EMT9kLog_B)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k_log_msk90':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk90.log.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"
    Aname = '(EMT9kLog_Bmsk90)'
    Bname = '(EMT9kLog_B)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
else:
    raise Warning("data name not recognized!")

# split data and save indexes
df_train, df_valid, df_test = scimpute.split_df(df, a=0.7, b=0.15, c=0.15)
df2_train, df2_valid, df2_test = df2.ix[df_train.index], df2.ix[df_valid.index], df2.ix[df_test.index]
df_train.to_csv('re_train/df_train.step2_index.csv', columns=[], header=False)  # save index for future use
df_valid.to_csv('re_train/df_valid.step2_index.csv', columns=[], header=False)
df_test.to_csv('re_train/df_test.step2_index.csv', columns=[], header=False)

# parameters
# todo: update for different depth
n_input = n
n_hidden_1 = 200
n_hidden_2 = 600
n_hidden_3 = 400
n_hidden_4 = 200
pIn = 0.8
pHidden = 0.5
learning_rate = 0.0003  # 0.0003 for 3-7L, 0.00003 for 9L # todo: change with layer
sd = 0.0001  # 3-7L:1e-3, 9L:1e-4
batch_size = 256
training_epochs = 3000  #3L:100, 5L:1000, 7L:1000, 9L:3000
display_step = 20
snapshot_step = 1000
# print_parameters()
j_lst = [4058, 7496, 8495, 12871]  # Cd34, Gypa, Klf1, Sfpi1

# Start model
tf.reset_default_graph()

# define placeholders and variables
X = tf.placeholder(tf.float32, [None, n_input], name='X_input')  # input
M = tf.placeholder(tf.float32, [None, n_input], name='M_ground_truth')  # benchmark
pIn_holder = tf.placeholder(tf.float32, name='pIn')
pHidden_holder = tf.placeholder(tf.float32, name='pHidden')

# define layers and variables
tf.set_random_seed(3)  # seed
# todo: adjust based on depth
with tf.name_scope('Encoder_L1'):
    e_w1, e_b1 = scimpute.weight_bias_variable('encoder1', n, n_hidden_1, sd)
    e_a1 = scimpute.dense_layer('encoder1', X, e_w1, e_b1, pIn_holder)
# with tf.name_scope('Encoder_L2'):
#     e_w2, e_b2 = scimpute.weight_bias_variable('encoder2', n_hidden_1, n_hidden_2, sd)
#     e_a2 = scimpute.dense_layer('encoder2', e_a1, e_w2, e_b2, pHidden_holder)
# with tf.name_scope('Encoder_L3'):
#     e_w3, e_b3 = scimpute.weight_bias_variable('encoder3', n_hidden_2, n_hidden_3, sd)
#     e_a3 = scimpute.dense_layer('encoder3', e_a2, e_w3, e_b3, pHidden_holder)
# with tf.name_scope('Encoder_L4'):
#     e_w4, e_b4 = scimpute.weight_bias_variable('encoder4', n_hidden_3, n_hidden_4, sd)
#     e_a4 = scimpute.dense_layer('encoder4', e_a3, e_w4, e_b4, pHidden_holder)
# with tf.name_scope('Decoder_L4'):
#     d_w4, d_b4 = scimpute.weight_bias_variable('decoder4', n_hidden_4, n_hidden_3, sd)
#     d_a4 = scimpute.dense_layer('decoder4', e_a4, d_w4, d_b4, pHidden_holder)
# with tf.name_scope('Decoder_L3'):
#     d_w3, d_b3 = scimpute.weight_bias_variable('decoder3', n_hidden_3, n_hidden_2, sd)
#     d_a3 = scimpute.dense_layer('decoder3', d_a4, d_w3, d_b3, pHidden_holder)
# with tf.name_scope('Decoder_L2'):
#     d_w2, d_b2 = scimpute.weight_bias_variable('decoder2', n_hidden_2, n_hidden_1, sd)
#     d_a2 = scimpute.dense_layer('decoder2', d_a3, d_w2, d_b2, pHidden_holder)
with tf.name_scope('Decoder_L1'):
    d_w1, d_b1 = scimpute.weight_bias_variable('decoder1', n_hidden_1, n, sd)
    d_a1 = scimpute.dense_layer('decoder1', e_a1, d_w1, d_b1, pHidden_holder)  # todo: change input activations if model changed

# define input/output
y_input = X
y_groundTruth = M
h = d_a1

# define loss
with tf.name_scope("Metrics"):
    mse_input = tf.reduce_mean(tf.pow(y_input - h, 2))
    mse_groundTruth = tf.reduce_mean(tf.pow(y_groundTruth - h, 2))
    tf.summary.scalar('mse_input', mse_input)
    tf.summary.scalar('mse_groundTruth', mse_groundTruth)

for j in j_lst:

