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


# read data into h, df1/2 [cells, genes]
h = scimpute.read_hd5(file_pred)

if p.file1_orientation == 'gene_row':
    df1 = pd.read_hdf(p.file1).transpose()
elif p.file1_orientation == 'cell_row':
    df1 = pd.read_hdf(p.file1)
else:
    raise Exception('parameter err: file1_orientation not correctly spelled')

if p.file2_orientation == 'gene_row':
    df2 = pd.read_hdf(p.file2).transpose()
elif p.file2_orientation == 'cell_row':
    df2 = pd.read_hdf(p.file2)
else:
    raise Exception('parameter err: file2_orientation not correctly spelled')


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


# Factors Affecting Gene Prediction
def gene_corr_list(arr1, arr2):
    '''calculate correlation between genes [columns]
    arr [cells, genes], note, some genes don't have corr'''
    # if arr1.shape is arr2.shape:
    n = arr2.shape[1]
    list = []
    for j in range(n):
        corr = pearsonr(arr1[:, j], arr2[:, j])[0]
        if math.isnan(corr):
            list.append(-1.1)  # NA becomes -1.1
        else:
            list.append(corr)
    list = np.array(list)
    return list


def gene_mse_list(arr1, arr2):
    '''mse for each gene(column)
    arr [cells, genes]
    arr1: X
    arr2: H'''
    n = arr2.shape[1]
    list = []
    for j in range(n):
        mse = ((arr1[:, j] - arr2[:, j]) ** 2).mean()
        list.append(mse)
    list = np.array(list)
    return list


def gene_nz_rate_list(arr1):
    '''nz_rate for each gene(column)
    arr [cells, genes]
    arr1: X'''
    n = arr1.shape[1]
    list = []
    for j in range(n):
        nz_rate = np.count_nonzero(arr1[:, j]) / n
        list.append(nz_rate)
    list = np.array(list)
    return list


def gene_var_list(arr1):
    '''variation for each gene(column)
    arr [cells, genes]
    arr: X'''
    n = arr1.shape[1]
    list = []
    for j in range(n):
        var = np.var(arr1[:, j])
        list.append(var)
    list = np.array(list)
    return list


def gene_nzvar_list(arr1):
    '''variation for non-zero values in each gene(column)
    arr [cells, genes]
    arr: X'''
    n = arr1.shape[1]
    list = []
    for j in range(n):
        data = arr1[:, j]
        nz_data = data[data.nonzero()]
        var = np.var(nz_data)
        list.append(var)
    list = np.array(list)
    return list


gene_corr = gene_corr_list(df2.values, h.values)
gene_mse = gene_mse_list(df2.values, h.values)
gene_mean_expression = df1.sum(axis=0).values / df1.shape[1]  # sum for each column
gene_nz_rate = gene_nz_rate_list(df1.values)
gene_var = gene_var_list(df1.values)
gene_nzvar = gene_nzvar_list(df1.values)

scimpute.density_plot(gene_mean_expression, gene_mse,
                      title='Factors, expression vs mse, {}'.format(p.stage),
                      dir=p.stage,
                      xlab='gene mean expression',
                      ylab='gene mse')

scimpute.density_plot(gene_mean_expression, gene_corr,
                      title='Factors, expression vs corr, {}'.format(p.stage),
                      dir=p.stage,
                      xlab='gene mean expression',
                      ylab='gene corr (NA: -1.1)')

scimpute.density_plot(gene_nz_rate, gene_mse,
                      title='Factors, nz_rate vs mse, {}'.format(p.stage),
                      dir=p.stage,
                      xlab='gene nz_rate',
                      ylab='gene mse')

scimpute.density_plot(gene_nz_rate, gene_corr,
                      title='Factors, nz_rate vs corr, {}'.format(p.stage),
                      dir=p.stage,
                      xlab='gene nz_rate',
                      ylab='gene corr (NA: -1.1)')

scimpute.density_plot(gene_var, gene_mse,
                      title='Factors, var vs mse, {}'.format(p.stage),
                      dir=p.stage,
                      xlab='gene variation',
                      ylab='gene mse')

scimpute.density_plot(gene_var, gene_corr,
                      title='Factors, var vs corr, {}'.format(p.stage),
                      dir=p.stage,
                      xlab='gene variation',
                      ylab='gene corr (NA: -1.1)')

scimpute.density_plot(gene_nzvar, gene_mse,
                      title='Factors, nz_var vs mse, {}'.format(p.stage),
                      dir=p.stage,
                      xlab='gene nz_variation',
                      ylab='gene mse')

scimpute.density_plot(gene_nzvar, gene_corr,
                      title='Factors, nz_var vs corr, {}'.format(p.stage),
                      dir=p.stage,
                      xlab='gene nz_variation',
                      ylab='gene corr (NA: -1.1)')


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

# Clustmap of weights, bottle-neck-activations (slow on GPU, moved to CPU)
# os.system('for file in ./{}/*npy; do python -u weight_visualization.py $file {}; done'.format(p.stage, p.stage))
