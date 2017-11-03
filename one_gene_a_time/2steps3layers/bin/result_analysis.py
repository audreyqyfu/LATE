#!/usr/bin/python
print('10/17/2017, reads h.hd5 and data.hd5, then analysis the result')
print('python -u result_analysis.py [step1/step2]')

import tensorflow as tf
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
# set filename #
if flag == 'step2':
    print('in step2')
    file1 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk90.log.hd5"  # data need imputation
    file2 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"  # data need imputation
    file_pred = 're_train/imputation.step2.hd5'
    tag = '(step2)'
    # read index
    train_idx = scimpute.read_csv('re_train/df_train.step2_index.csv').index
    valid_idx = scimpute.read_csv('re_train/df_valid.step2_index.csv').index
elif flag == 'step1':
    print('in step1')
    file1 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # data need imputation
    file2 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # data need imputation
    file_pred = 'pre_train/imputation.step1.hd5'
    tag = '(step1)'
    # read index
    train_idx = scimpute.read_csv('pre_train/df_train.index.csv').index
    valid_idx = scimpute.read_csv('pre_train/df_valid.index.csv').index
else:
    raise Exception('argument wrong')

# read data #
df1 = scimpute.read_hd5(file1).transpose()
print('input:', df1.ix[0:4, 0:4])
print('df1.shape', df1.shape)
df2 = scimpute.read_hd5(file2).transpose()
print('df2.shape', df2.shape)
h = scimpute.read_hd5(file_pred)
print('h.shape', h.shape)

# split data
h_train, h_valid = h.ix[train_idx], h.ix[valid_idx]
df_train, df_valid = df1.ix[train_idx], df1.ix[valid_idx]
df2_train, df2_valid = df2.ix[train_idx], df2.ix[valid_idx]

# M vs H
def m_vs_h():
    print("> Ground truth vs prediction")
    for j in [1, 2, 3, 4, 205, 206, 4058, 7496, 8495, 12871]:  # Cd34, Gypa, Klf1, Sfpi1
            scimpute.scatterplot2(df2_valid.values[:, j], h_valid.values[:, j], range='same',
                                  title=str('M_vs_H, Gene' + str(j) + ' (valid)'+tag),
                                  xlabel='Ground Truth (M)',
                                  ylabel='Prediction (H)'
                                  )
            scimpute.scatterplot2(df2_valid.values[:, j], h_valid.values[:, j], range='flexible',
                                  title=str('M_vs_H(zoom), Gene' + str(j) + ' (valid)'+tag),
                                  xlabel='Ground Truth (M)',
                                  ylabel='Prediction (H)'
                                 )
            scimpute.scatterplot2(df2_valid.values[:, j], df_valid.values[:, j], range='same',
                                  title=str('M_vs_X, Gene' + str(j) + ' (valid)'+tag),
                                  xlabel='Ground Truth (M)',
                                  ylabel='Input (X)'
                                 )
m_vs_h()

# Gene-Gene in M, X, H
def gene_gene_relationship():
    print('> gene-gene relationship, before/after inference')
    List = [[4058, 7496],
            [8495, 12871],
            [2, 3],
            [205, 206]
            ]
    # Valid, H
    for i, j in List:
        scimpute.scatterplot2(h_valid.ix[:, i], h_valid.ix[:, j],
                              title='Gene' + str(i) + ' vs Gene' + str(j) + ' (H,valid)' + tag,
                              xlabel='Gene' + str(i), ylabel='Gene' + str(j + 1))
    # Valid, M
    for i, j in List:
        scimpute.scatterplot2(df2_valid.ix[:, i], df2_valid.ix[:, j],
                              title="Gene" + str(i) + ' vs Gene' + str(j) + ' (M,valid)' + tag,
                              xlabel='Gene' + str(i), ylabel='Gene' + str(j))
    # Valid, X
    for i, j in List:
        scimpute.scatterplot2(df_valid.ix[:, i], df_valid.ix[:, j],
                              title="Gene" + str(i) + ' vs Gene' + str(j) + ' (X,valid)' + tag,
                              xlabel='Gene' + str(i), ylabel='Gene' + str(j))
gene_gene_relationship()

# todo: Hist of df1 (need to fix x)
# scimpute.hist_df(df_valid, 'X(valid)')
# scimpute.hist_df(df2_valid, 'M(valid)')
# scimpute.hist_df(h_valid, 'H(valid)')

# Hist Cell/Gene corr
hist = scimpute.gene_corr_hist(h_valid.values, df2_valid.values,
                               title="Hist Gene-Corr (H vs M)"+tag
                               )
hist = scimpute.cell_corr_hist(h_valid.values, df2_valid.values,
                               title="Hist Cell-Corr (H vs M)"+tag
                               )

def visualization_of_dfs():
    print('visualization of dfs')
    max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid])
    # max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid, h, df.values])
    scimpute.heatmap_vis(df_valid.values, title='df.valid'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(h_valid, title='h.valid'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(df.values, title='df'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(h, title='h'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)


# # gene MSE
# j = 0
# input_j = df1.ix[:, j:j+1].values
# pred_j = h.ix[:, j:j+1].values
# groundTruth_j = df2.ix[:, j:j+1].values
#
# mse_j_input = ((pred_j - input_j) ** 2).mean()
# mse_j_groundTruth = ((pred_j - groundTruth_j) ** 2).mean()
#
# # matrix MSE
# matrix_mse_input = ((h.values - df1.values) ** 2).mean()
# matrix_mse_groundTruth = ((h.values - df2.values) ** 2).mean()