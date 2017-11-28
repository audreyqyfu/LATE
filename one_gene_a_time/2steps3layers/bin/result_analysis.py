#!/usr/bin/python
print('10/17/2017, reads h.hd5 and data.hd5, then analysis the result')
print('python -u result_analysis.py [step1/step2]')

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
df_train, df_valid = df1.ix[train_idx], df1.ix[valid_idx]
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
            scimpute.scatterplot2(df2_valid.values[:, j], df_valid.values[:, j], range='same',
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
        scimpute.scatterplot2(df_valid.ix[:, i], df_valid.ix[:, j],
                              title="Gene" + str(i) + ' vs Gene' + str(j) + ' (X,valid)' + tag,
                              xlabel='Gene' + str(i), ylabel='Gene' + str(j),
                              dir=p.stage)
gene_gene_relationship()

# todo: Hist of df1 (need to fix x)
# scimpute.hist_df(df_valid, 'X(valid)')
# scimpute.hist_df(df2_valid, 'M(valid)')
# scimpute.hist_df(h_valid, 'H(valid)')

# Hist Cell/Gene corr
hist = scimpute.gene_corr_hist(h_valid.values, df2_valid.values,
                               title="Hist Gene-Corr (H vs M)"+tag,
                               dir=p.stage
                               )
hist = scimpute.cell_corr_hist(h_valid.values, df2_valid.values,
                               title="Hist Cell-Corr (H vs M)"+tag,
                               dir=p.stage
                               )

def visualization_of_dfs():
    print('> Visualization of dfs')
    max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid])
    # max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid, h, df.values])
    scimpute.heatmap_vis(df_valid.values, title='df.valid'+Aname,
                         xlab='genes', ylab='cells', vmax=max, vmin=min,
                         dir=p.stage)
    scimpute.heatmap_vis(h_valid, title='h.valid'+Aname,
                         xlab='genes', ylab='cells', vmax=max, vmin=min,
                         dir=p.stage)
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