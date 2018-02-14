#!/usr/bin/python
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
import scimpute

import result_analysis_matrix_based_params as p


# read cmd
print('reads H.hd5 and M.hd5, then analysis the result')
print('usage: python -u result_analysis.py H.hd5 gene_row/cell_row M.hd5 gene_row/cell_row X.hd5 gene_row/cell_row out_tag')
print('H means prediction, M means ground truth')

if len(sys.argv) != 8:
    raise Exception('cmd err')
else:
    print('cmd: ', sys.argv)
    
file_h = str(sys.argv[1]).strip()
file_h_ori = str(sys.argv[2]).strip()
file_m = str(sys.argv[3]).strip()
file_m_ori = str(sys.argv[4]).strip()
file_x = str(sys.argv[5]).strip()
file_x_ori = str(sys.argv[6]).strip()
tag = str(sys.argv[7]).strip()

# read data
if file_h_ori == 'gene_row':
    H = pd.read_hdf(file_h).transpose()
elif file_h_ori == 'cell_row':
    H = pd.read_hdf(file_h)
else:
    raise Exception('parameter err: h_orientation not correctly spelled')

if file_m_ori == 'gene_row':
    M = pd.read_hdf(file_m).transpose()
elif file_m_ori == 'cell_row':
    M = pd.read_hdf(file_m)
else:
    raise Exception('parameter err: m_orientation not correctly spelled')

if file_x_ori == 'gene_row':
    X = pd.read_hdf(file_x).transpose()
elif file_x_ori == 'cell_row':
    X = pd.read_hdf(file_x)
else:
    raise Exception('parameter err: x_orientation not correctly spelled')

# Test mode or not
test_flag = 0
m = 100
n = 200
if test_flag > 0:
    print('in test mode')
    H = H.ix[0:m, 0:n]
    M = M.ix[0:m, 0:n]
    X = X.ix[0:m, 0:n]

# input summary
print('inside this code, matrices are supposed to be cell_row')
print('H:', file_h, file_h_ori, '\n', H.ix[0:3, 0:2])
print('M:', file_m, file_m_ori, '\n', M.ix[0:3, 0:2])
print('X:', file_x, file_x_ori, '\n', X.ix[0:3, 0:2])
print('H.shape', H.shape)
print('M.shape', M.shape)
print('X.shape', X.shape)

# Hist of H todo combine
scimpute.hist_df(H, title='H({})'.format(file_h), dir=tag)
scimpute.hist_df(M, title='M({})'.format(file_m), dir=tag)
scimpute.hist_df(X, title='X({})'.format(file_x), dir=tag)


# Hist Cell/Gene corr todo combine
print('between H and M')
hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist Gene-Corr (H vs M)\n"+file_h+'\n'+file_m,
                               dir=tag, mode='column-wise', nz_mode='ignore'
                               )

hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist Cell-Corr (H vs M)\n"+file_h+'\n'+file_m,
                               dir=tag, mode='row-wise', nz_mode='ignore'
                               )

hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist nz1-Gene-Corr (H vs M)\n"+file_h+'\n'+file_m,
                               dir=tag, mode='column-wise', nz_mode='first'
                               )

hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist nz1-Cell-Corr (H vs M)\n"+file_h+'\n'+file_m,
                               dir=tag, mode='row-wise', nz_mode='first'
                               )

hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist nz2-Gene-Corr (H vs M)\n"+file_h+'\n'+file_m,
                               dir=tag, mode='column-wise', nz_mode='strict'
                               )

hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist nz2-Cell-Corr (H vs M)\n"+file_h+'\n'+file_m,
                               dir=tag, mode='row-wise', nz_mode='strict'
                               )

hist = scimpute.hist_2matrix_corr(X.values, H.values,
                               title="Hist nz1-Gene-Corr (H vs X)\n"+file_h+'\n'+file_m,
                               dir=tag, mode='column-wise', nz_mode='first'
                               )

hist = scimpute.hist_2matrix_corr(X.values, H.values,
                               title="Hist nz1-Cell-Corr (H vs X)\n"+file_h+'\n'+file_m,
                               dir=tag, mode='row-wise', nz_mode='first'
                               )

# Visualization of dfs
print('> Visualization of dfs')
max_h, min_h = scimpute.max_min_element_in_arrs([H.values])
print('Max in H is {}, Min in H is{}'.format(max_h, min_h))
max_m, min_m = scimpute.max_min_element_in_arrs([M.values])
print('Max in M is {}, Min in M is{}'.format(max_m, min_m))

mse1_omega = scimpute.mse_omega(H, X)
mse1_omega = round(mse1_omega, 5)
print('mse1_omega between H and X: ', mse1_omega)

mse2 = scimpute.mse(H, M)
mse2 = round(mse2, 5)
print('MSE2 between H and M: ', mse2)

mse2_omega = scimpute.mse_omega(H, M)
mse2_omega = round(mse2_omega, 5)
print('mse2_omega between H and M: ', mse2_omega)


max, min = scimpute.max_min_element_in_arrs([H.values, M.values])
scimpute.heatmap_vis(H.values,
                     title='H ({})'.format(file_h),
                     xlab='genes\nMSE1_OMEGA(H vs X)={}'.format(mse1_omega),
                     ylab='cells', vmax=max, vmin=min,
                     dir=tag)
scimpute.heatmap_vis(M.values,
                     title='M ({})'.format(file_m),
                     xlab='genes\nMSE2(H vs M)={}'.format(mse2),
                     ylab='cells', vmax=max, vmin=min,
                     dir=tag)


# # Factors Affecting Gene Prediction todo: validate
# print('Mean and Var are calculated from H')
# gene_corr = scimpute.gene_corr_list(M.values, H.values)
# gene_mse = scimpute.gene_mse_list(M.values, H.values)
# gene_mean_expression = M.sum(axis=0).values / M.shape[1]  # sum for each column
# gene_nz_rate = scimpute.gene_nz_rate_list(M.values)
# gene_var = scimpute.gene_var_list(M.values)
# gene_nzvar = scimpute.gene_nzvar_list(M.values)
#
# scimpute.density_plot(gene_mean_expression, gene_mse,
#                       title='Factors, expression vs mse, {}'.format('test'),
#                       dir=tag,
#                       xlab='gene mean expression',
#                       ylab='gene mse')
#
# scimpute.density_plot(gene_mean_expression, gene_corr,
#                       title='Factors, expression vs corr, {}'.format('test'),
#                       dir=tag,
#                       xlab='gene mean expression',
#                       ylab='gene corr (NA: -1.1)')
#
# scimpute.density_plot(gene_nz_rate, gene_mse,
#                       title='Factors, nz_rate vs mse, {}'.format('test'),
#                       dir=tag,
#                       xlab='gene nz_rate',
#                       ylab='gene mse')
#
# scimpute.density_plot(gene_nz_rate, gene_corr,
#                       title='Factors, nz_rate vs corr, {}'.format('test'),
#                       dir=tag,
#                       xlab='gene nz_rate',
#                       ylab='gene corr (NA: -1.1)')
#
# scimpute.density_plot(gene_var, gene_mse,
#                       title='Factors, var vs mse, {}'.format('test'),
#                       dir=tag,
#                       xlab='gene variation',
#                       ylab='gene mse')
#
# scimpute.density_plot(gene_var, gene_corr,
#                       title='Factors, var vs corr, {}'.format('test'),
#                       dir=tag,
#                       xlab='gene variation',
#                       ylab='gene corr (NA: -1.1)')
#
# # todo: sometimes NA error for the following two plots
# scimpute.density_plot(gene_nzvar, gene_mse,
#                       title='Factors, nz_var vs mse, {}'.format('test'),
#                       dir=tag,
#                       xlab='gene nz_variation',
#                       ylab='gene mse')
#
# scimpute.density_plot(gene_nzvar, gene_corr,
#                       title='Factors, nz_var vs corr, {}'.format('test'),
#                       dir=tag,
#                       xlab='gene nz_variation',
#                       ylab='gene corr (NA: -1.1)')

# Gene-Gene in M, X, H
print('> Gene-gene relationship, before/after inference')
gene_pair_dir = tag+'/pairs'
List = p.pair_list
# Valid, H
for i, j in List:
    print(i, type(i), j, type(j))
    scimpute.scatterplot2(H.ix[:, i], H.ix[:, j],
                          title='Gene' + str(i) + ' vs Gene' + str(j) + ' (H)' + tag,
                          xlabel='Gene' + str(i), ylabel='Gene' + str(j),
                          dir=gene_pair_dir)
# Valid, M
for i, j in List:
    scimpute.scatterplot2(M.ix[:, i], M.ix[:, j],
                          title="Gene" + str(i) + ' vs Gene' + str(j) + ' (M)' + tag,
                          xlabel='Gene' + str(i), ylabel='Gene' + str(j),
                          dir=gene_pair_dir)
# Valid, X
for i, j in List:
    scimpute.scatterplot2(X.ix[:, i], X.ix[:, j],
                          title="Gene" + str(i) + ' vs Gene' + str(j) + ' (X)' + tag,
                          xlabel='Gene' + str(i), ylabel='Gene' + str(j),
                          dir=gene_pair_dir)


# M vs H, M vs X
print("> M vs H, M vs X")
gene_dir = tag+'/genes'
for j in p.gene_list:  # Cd34, Gypa, Klf1, Sfpi1
        scimpute.scatterplot2(M.ix[:, j], H.ix[:, j], range='same',
                              title=str('M_vs_H ' + str(j) +' '+tag),
                              xlabel='Ground Truth (M)',
                              ylabel='Prediction (H)',
                              dir=gene_dir
                              )
        scimpute.scatterplot2(M.ix[:, j], X.ix[:, j], range='same',
                              title=str('M_vs_X ' + str(j) +' '+tag),
                              xlabel='Ground Truth (M)',
                              ylabel='Input (X)',
                              dir=gene_dir
                             )


# # gene MSE
# j = 0
# input_j = H.ix[:, j:j+1].values
# pred_j = h.ix[:, j:j+1].values
# groundTruth_j = M.ix[:, j:j+1].values
#
# mse_j_input = ((pred_j - input_j) ** 2).mean()
# mse_j_groundTruth = ((pred_j - groundTruth_j) ** 2).mean()
#
# matrix MSE

# Clustmap of weights, bottle-neck-activations (slow on GPU, moved to CPU)
# os.system('for file in ./{}/*npy; do python -u weight_visualization.py $file {}; done'.format(p.stage, p.stage))
