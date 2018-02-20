#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import importlib
import scimpute

# READ CMD
print('reads H.hd5, X.hd5 and M.hd5, then analysis the result')
print('usage: python -u result_analysis.py params.py')

if len(sys.argv) == 2:
    param_file = sys.argv[1]
    param_file = param_file.rstrip('.py')
    p = importlib.import_module(param_file)
else:
    raise Exception('cmd err')


# READ DATA
print("> READ DATA..")
H = scimpute.read_data_into_cell_row(p.file_h, p.file_h_ori)
X = scimpute.read_data_into_cell_row(p.file_x, p.file_x_ori)
M = scimpute.read_data_into_cell_row(p.file_m, p.file_m_ori)

# Data Transformation for H
print('> DATA TRANSFORMATION..')
H = scimpute.df_transformation(H.transpose(), transformation=p.data_transformation).transpose()
X = scimpute.df_transformation(X.transpose(), transformation=p.data_transformation).transpose()
M = scimpute.df_transformation(M.transpose(), transformation=p.data_transformation).transpose()


# TEST MODE OR NOT
test_flag = 0
m = 100
n = 200
if test_flag > 0:
    print('in test mode')
    H = H.ix[0:m, 0:n]
    M = M.ix[0:m, 0:n]
    X = X.ix[0:m, 0:n]

# INPUT SUMMARY
print('\ninside this code, matrices are supposed to be transformed into cell_row')
print('H:', p.file_h, p.file_h_ori, p.data_transformation, '\n', H.ix[0:3, 0:2])
print('M:', p.file_m, p.file_m_ori, p.data_transformation, '\n', M.ix[0:3, 0:2])
print('X:', p.file_x, p.file_x_ori, p.data_transformation, '\n', X.ix[0:3, 0:2])
print('H.shape', H.shape)
print('M.shape', M.shape)
print('X.shape', X.shape)

# HIST OF H
scimpute.hist_df(H, title='H({})'.format(p.file_h), dir=p.tag)
scimpute.hist_df(M, title='M({})'.format(p.file_m), dir=p.tag)
scimpute.hist_df(X, title='X({})'.format(p.file_x), dir=p.tag)


# HIST CELL/GENE CORR
print('\n> Corr between X and H')
hist = scimpute.hist_2matrix_corr(X.values, H.values,
                               title="Hist nz1-Gene-Corr (X vs H)\n"+p.file_h+'\n'+p.file_m,
                               dir=p.tag, mode='column-wise', nz_mode='first'
                               )

hist = scimpute.hist_2matrix_corr(X.values, H.values,
                               title="Hist nz1-Cell-Corr (X vs H)\n"+p.file_h+'\n'+p.file_m,
                               dir=p.tag, mode='row-wise', nz_mode='first'
                               )


print('\n> Corr between M and H')
hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist Gene-Corr (M vs H)\n"+p.file_h+'\n'+p.file_m,
                               dir=p.tag, mode='column-wise', nz_mode='ignore'
                               )

hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist Cell-Corr (M vs H)\n"+p.file_h+'\n'+p.file_m,
                               dir=p.tag, mode='row-wise', nz_mode='ignore'
                               )

hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist nz1-Gene-Corr (M vs H)\n"+p.file_h+'\n'+p.file_m,
                               dir=p.tag, mode='column-wise', nz_mode='first'
                               )

hist = scimpute.hist_2matrix_corr(M.values, H.values,
                               title="Hist nz1-Cell-Corr (M vs H)\n"+p.file_h+'\n'+p.file_m,
                               dir=p.tag, mode='row-wise', nz_mode='first'
                               )


# MSE CALCULATION
print('\n> MSE Calculation')
max_h, min_h = scimpute.max_min_element_in_arrs([H.values])
print('Max in H is {}, Min in H is{}'.format(max_h, min_h))
max_m, min_m = scimpute.max_min_element_in_arrs([M.values])
print('Max in M is {}, Min in M is{}'.format(max_m, min_m))

mse1_omega = scimpute.mse_omega(H, X)
mse1_omega = round(mse1_omega, 5)
print('mse1_omega between H and X: ', mse1_omega)

mse2_omega = scimpute.mse_omega(H, M)
mse2_omega = round(mse2_omega, 5)
print('mse2_omega between H and M: ', mse2_omega)

mse2 = scimpute.mse(H, M)
mse2 = round(mse2, 5)
print('MSE2 between H and M: ', mse2)


#  VISUALIZATION OF DFS, todo clustering based on H
print('\n> Visualization of dfs')
max, min = scimpute.max_min_element_in_arrs([H.values, M.values])
scimpute.heatmap_vis(H.values,
                     title='H ({})'.format(p.file_h),
                     xlab='genes\nMSE1_OMEGA(H vs X)={}'.format(mse1_omega),
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)

scimpute.heatmap_vis(M.values,
                     title='M ({})'.format(p.file_m),
                     xlab='genes\nMSE2(H vs M)={}'.format(mse2),
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)


# Gene-Gene in M, X, H
print('\n> Gene-gene relationship (H, X, M), before/after inference')
gene_pair_dir = p.tag+'/pairs'
List = p.pair_list
scimpute.gene_pair_plot(H, list=List, tag='(H) '+p.tag, dir=gene_pair_dir)
scimpute.gene_pair_plot(X, list=List, tag='(X) '+p.tag, dir=gene_pair_dir)
scimpute.gene_pair_plot(M, list=List, tag='(M) '+p.tag, dir=gene_pair_dir)


# M vs H, M vs X
print("\n> M vs H, M vs X")
gene_dir = p.tag+'/genes'
for j in p.gene_list:
    try:
        print('for ', j)
        H_j = H.ix[:, j]
        M_j = M.ix[:, j]
        X_j = X.ix[:, j]
    except KeyError:
        print('KeyError: the gene index does not exist')
        continue

    scimpute.scatterplot2(M_j, H_j, range='same',
                          title=str(str(j) + ' (M_vs_H) ' + p.tag),
                          xlabel='Ground Truth (M)',
                          ylabel='Prediction (H)',
                          dir=gene_dir
                          )
    scimpute.scatterplot2(M_j, X_j, range='same',
                          title=str(str(j) + ' (M_vs_X) ' + p.tag),
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
#                       dir=p.tag,
#                       xlab='gene mean expression',
#                       ylab='gene mse')
#
# scimpute.density_plot(gene_mean_expression, gene_corr,
#                       title='Factors, expression vs corr, {}'.format('test'),
#                       dir=p.tag,
#                       xlab='gene mean expression',
#                       ylab='gene corr (NA: -1.1)')
#
# scimpute.density_plot(gene_nz_rate, gene_mse,
#                       title='Factors, nz_rate vs mse, {}'.format('test'),
#                       dir=p.tag,
#                       xlab='gene nz_rate',
#                       ylab='gene mse')
#
# scimpute.density_plot(gene_nz_rate, gene_corr,
#                       title='Factors, nz_rate vs corr, {}'.format('test'),
#                       dir=p.tag,
#                       xlab='gene nz_rate',
#                       ylab='gene corr (NA: -1.1)')
#
# scimpute.density_plot(gene_var, gene_mse,
#                       title='Factors, var vs mse, {}'.format('test'),
#                       dir=p.tag,
#                       xlab='gene variation',
#                       ylab='gene mse')
#
# scimpute.density_plot(gene_var, gene_corr,
#                       title='Factors, var vs corr, {}'.format('test'),
#                       dir=p.tag,
#                       xlab='gene variation',
#                       ylab='gene corr (NA: -1.1)')
#
# # todo: sometimes NA error for the following two plots
# scimpute.density_plot(gene_nzvar, gene_mse,
#                       title='Factors, nz_var vs mse, {}'.format('test'),
#                       dir=p.tag,
#                       xlab='gene nz_variation',
#                       ylab='gene mse')
#
# scimpute.density_plot(gene_nzvar, gene_corr,
#                       title='Factors, nz_var vs corr, {}'.format('test'),
#                       dir=p.tag,
#                       xlab='gene nz_variation',
#                       ylab='gene corr (NA: -1.1)')
