#!/usr/bin/python
print('reads h.hd5 and data.hd5, then analysis the result')
print('usage: python -u result_analysis.py [step1/step2]')

import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import math
import os
import time
import scimpute

flag = sys.argv[1].strip()
print(flag + ",\n")
# stage specific params
if flag == 'step1':
    print('in step1')
    import step1_params as p
    file1 = p.file1
    file2 = p.file1
elif flag == 'step2':
    print('in step2')
    import step2_params as p
    file1 = p.file1
    file2 = p.file2
else:
    raise Exception('argument wrong')

# some common params
file_pred = '{}/imputation.{}.hd5'.format(p.stage, p.stage)
tag = '({})'.format(p.stage)
train_idx = scimpute.read_csv('{}/df1_train.{}_index.csv'.format(p.stage, p.stage)).index
valid_idx = scimpute.read_csv('{}/df1_valid.{}_index.csv'.format(p.stage, p.stage)).index


# read data #
if p.file_orientation == 'gene_row':
    df1 = scimpute.read_hd5(file1).transpose()
    df2 = scimpute.read_hd5(file2).transpose()
    h = scimpute.read_hd5(file_pred)
elif p.file_orientation == 'cell_row':
    df1 = scimpute.read_hd5(file1)
    df2 = scimpute.read_hd5(file2)
    h = scimpute.read_hd5(file_pred)
else:
    raise Exception('p.file_orientation spelled wrong')


# If test
if p.test_flag > 0:
    print('in test mode')
    df1 = df1.ix[0:p.m, 0:p.n]
    df2 = df2.ix[0:p.m, 0:p.n]

# input summary
print('df1:', df1.ix[0:3, 0:2])
print('df2:', df2.ix[0:3, 0:2])
print('h:', h.ix[0:3, 0:2])

print('df1.shape', df1.shape)
print('df2.shape', df2.shape)
print('h.shape', h.shape)

# split data
h_train, h_valid = h.ix[train_idx], h.ix[valid_idx]
df1_train, df1_valid = df1.ix[train_idx], df1.ix[valid_idx]
df2_train, df2_valid = df2.ix[train_idx], df2.ix[valid_idx]

# M vs H, M vs X
def m_vs_h():
    print("> M vs H, M vs X")
    for j in p.gene_list:  # Cd34, Gypa, Klf1, Sfpi1
            scimpute.scatterplot2(df2_valid.values[:, j], h_valid.values[:, j], range='same',
                                  title=str('M_vs_H, Gene' + str(j) + ' (valid)'+tag),
                                  xlabel='Ground Truth (M)',
                                  ylabel='Prediction (H)',
                                  dir=p.stage
                                  )
            # scimpute.scatterplot2(df2_valid.values[:, j], h_valid.values[:, j], range='flexible',
            #                       title=str('M_vs_H(zoom), Gene' + str(j) + ' (valid)'+tag),
            #                       xlabel='Ground Truth (M)',
            #                       ylabel='Prediction (H)',
            #                       dir=p.stage
            #                      )
            scimpute.scatterplot2(df2_valid.values[:, j], df1_valid.values[:, j], range='same',
                                  title=str('M_vs_X, Gene' + str(j) + ' (valid)'+tag),
                                  xlabel='Ground Truth (M)',
                                  ylabel='Input (X)',
                                  dir=p.stage
                                 )
m_vs_h()


# Gene-Gene in M, X, H
def gene_gene_relationship():
    print('> Gene-gene relationship, before/after inference')
    List = p.pair_list
    # Valid, H
    for i, j in List:
        scimpute.scatterplot2(h_valid.ix[:, i], h_valid.ix[:, j],
                              title='Gene' + str(i) + ' vs Gene' + str(j) + ' (H,valid)' + tag,
                              xlabel='Gene' + str(i), ylabel='Gene' + str(j + 1),
                              dir=p.stage)
    # Valid, M
    for i, j in List:
        scimpute.scatterplot2(df2_valid.ix[:, i], df2_valid.ix[:, j],
                              title="Gene" + str(i) + ' vs Gene' + str(j) + ' (M,valid)' + tag,
                              xlabel='Gene' + str(i), ylabel='Gene' + str(j),
                              dir=p.stage)
    # Valid, X
    for i, j in List:
        scimpute.scatterplot2(df1_valid.ix[:, i], df1_valid.ix[:, j],
                              title="Gene" + str(i) + ' vs Gene' + str(j) + ' (X,valid)' + tag,
                              xlabel='Gene' + str(i), ylabel='Gene' + str(j),
                              dir=p.stage)
gene_gene_relationship()


# Hist of df1
scimpute.hist_df(df1_valid, title='X(valid)({})'.format(p.name1), dir=p.stage)
scimpute.hist_df(df2_valid, title='M(valid)({})'.format(p.name2), dir=p.stage)
scimpute.hist_df(h_valid, title='H(valid)({})'.format(p.name1), dir=p.stage)


# Hist Cell/Gene corr
hist = scimpute.gene_corr_hist(h_valid.values, df2_valid.values,
                               title="Hist Gene-Corr (H vs M)"+tag,
                               dir=p.stage
                               )
hist = scimpute.cell_corr_hist(h_valid.values, df2_valid.values,
                               title="Hist Cell-Corr (H vs M)"+tag,
                               dir=p.stage
                               )


# Visualization of dfs
def visualization_of_dfs():
    print('> Visualization of dfs')
    max, min = scimpute.max_min_element_in_arrs([df1_valid.values,
                                                 h_valid.values,
                                                 df2_valid.values])
    mse1 = ((h_valid.values - df1_valid.values) ** 2).mean()
    mse2 = ((h_valid.values - df2_valid.values) ** 2).mean()
    mse1 = round(mse1, 5)
    mse2 = round(mse2, 5)
    scimpute.heatmap_vis(df1_valid.values,
                         title='X.valid.{}'.format(p.name1),
                         xlab='genes', ylab='cells', vmax=max, vmin=min,
                         dir=p.stage)
    scimpute.heatmap_vis(h_valid.values,
                         title='H.valid.{}'.format(p.name1),
                         xlab='genes\nmse1={}\nmse2={}'.format(mse1, mse2),
                         ylab='cells', vmax=max, vmin=min,
                         dir=p.stage)
    scimpute.heatmap_vis(df2_valid.values, title='M.valid.{}'.format(p.name2),
                         xlab='genes', ylab='cells', vmax=max, vmin=min,
                         dir=p.stage)
visualization_of_dfs()


# Clustmap of weights, bottle-neck-activations (slow on GPU, moved to CPU)
os.system('for file in ./{}/*npy; do python -u weight_visualization.py $file {}; done'.format(p.stage, p.stage))

# # gene MSE
# j = 0
# input_j = df1.ix[:, j:j+1].values
# pred_j = h.ix[:, j:j+1].values
# groundTruth_j = df2.ix[:, j:j+1].values
#
# mse_j_input = ((pred_j - input_j) ** 2).mean()
# mse_j_groundTruth = ((pred_j - groundTruth_j) ** 2).mean()
#
# matrix MSE
