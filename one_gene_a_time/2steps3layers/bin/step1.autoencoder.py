#!/usr/bin/python
# 10/06/2017 use tf.contrib.layers function, re-write from scratch
# inspired by: http://ruishu.io/2016/12/27/batchnorm/
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

print('python version:', sys.version)
print('tf.__version__', tf.__version__)

sys.path.append('./bin')
print('sys.path', sys.path)
import scimpute


# read data #
data = 'EMT9k_log'  # EMT2730 or splatter

if data is 'splatter':  # only this mode creates gene-gene plot
    file = "../data/v1-1-5-3/v1-1-5-3.E3.hd5"  # data need imputation
    file_benchmark = "../data/v1-1-5-3/v1-1-5-3.E3.hd5"
    Aname = '(E3)'
    Bname = '(E3)'  # careful
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT2730':  # 2.7k cells used in magic paper
    file = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5" #data need imputation
    file_benchmark = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5"
    Aname = '(EMT2730)'
    Bname = '(EMT2730)'
    df = pd.read_hdf(file).transpose() #[cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.hd5"
    Aname = '(EMT9k)'
    Bname = '(EMT9k)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k_log':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"
    Aname = '(EMT9kLog)'
    Bname = '(EMT9kLog)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
else:
    raise Warning("data name not recognized!")

max = max(df.values.max(), df2.values.max())

# rand split data
[df_train, df_valid, df_test] = scimpute.split_df(df, a=0.9, b=0.1, c=0.0)

df2_train = df2.ix[df_train.index]
df2_valid = df2.ix[df_valid.index]
df2_test = df2.ix[df_test.index]

# Parameters #
learning_rate = 0.0003
annealing_constant = 0.98  # for each epoch
training_epochs = 4000  # todo change epochs
batch_size = 256
pIn = 0.8
pHidden = 0.5
sd = 0.0001  # stddev for random init
n_input = n
n_hidden_1 = 1600
n_hidden_2 = 800
n_hidden_3 = 400
n_hidden_4 = 200
print_parameters()

display_step = 20
snapshot_step = 1000

log_dir = './pre_train'
scimpute.refresh_logfolder(log_dir)
corr_log = []
epoch_log = []


# Define model #
def dense_batch_relu(x, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, 200,
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_input])  # input
M = tf.placeholder(tf.float32, [None, n_input])  # benchmark
phase = tf.placeholder(tf.bool, name='phase')

# keep_prob_input = tf.placeholder(tf.float32)
# keep_prob_hidden = tf.placeholder(tf.float32)

h1 = dense_batch_relu(x, phase,'layer1')
h2 = dense_batch_relu(h1, phase, 'layer2')
logits = dense(h2, 10, 'logits')


