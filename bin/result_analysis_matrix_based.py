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
print('reads Y.hd5, X.hd5 and G.hd5, then analysis the result')
print('usage: python -u result_analysis.py params.py')

if len(sys.argv) == 2:
    param_file = sys.argv[1]
    param_file = param_file.rstrip('.py')
    p = importlib.import_module(param_file)
else:
    raise Exception('cmd err')


# READ DATA
print("> READ DATA..")
Y = scimpute.read_data_into_cell_row(p.file_h, p.file_h_ori)
X = scimpute.read_data_into_cell_row(p.file_x, p.file_x_ori)
G = scimpute.read_data_into_cell_row(p.file_m, p.file_m_ori)

# Data Transformation for Y
print('> DATA TRANSFORMATION..')
Y = scimpute.df_transformation(Y.transpose(), transformation=p.file_h_transformation).transpose()
X = scimpute.df_transformation(X.transpose(), transformation=p.file_x_transformation).transpose()
G = scimpute.df_transformation(G.transpose(), transformation=p.file_m_transformation).transpose()

# for MAGIC, discard missing genes from Y
X = X.loc[Y.index, Y.columns]
G = G.loc[Y.index, Y.columns]

# TEST MODE OR NOT
test_flag = 0
m = 100
n = 200
if test_flag > 0:
    print('in test mode')
    Y = Y.ix[0:m, 0:n]
    G = G.ix[0:m, 0:n]
    X = X.ix[0:m, 0:n]

# INPUT SUMMARY
print('\ninside this code, matrices are supposed to be transformed into cell_row')
print('Y:', p.file_h, p.file_h_ori, p.file_h_transformation, '\n', Y.ix[0:3, 0:2])
print('G:', p.file_m, p.file_m_ori, p.file_m_transformation, '\n', G.ix[0:3, 0:2])
print('X:', p.file_x, p.file_x_ori, p.file_x_transformation, '\n', X.ix[0:3, 0:2])
print('Y.shape', Y.shape)
print('G.shape', G.shape)
print('X.shape', X.shape)

# HIST OF Y
scimpute.hist_df(Y, title='Y({})'.format(p.name_h), dir=p.tag)
scimpute.hist_df(G, title='G({})'.format(p.name_m), dir=p.tag)
scimpute.hist_df(X, title='X({})'.format(p.name_x), dir=p.tag)


# HIST CELL/GENE CORR
print('\n> Corr between X and Y')
print('GeneCorr: X shape: ', X.shape, 'Y shape: ', Y.shape)
hist = scimpute.hist_2matrix_corr(X.values, Y.values,
                                  title="Hist nz1-Gene-Corr (X vs Y)\n"+p.name_x+'\n'+p.name_h,
                                  dir=p.tag, mode='column-wise', nz_mode='first'
                                  )
#
# print('CellCorr: X shape: ', X.shape, 'Y shape: ', Y.shape)
# hist = scimpute.hist_2matrix_corr(X.values, Y.values,
#                                   title="Hist nz1-Cell-Corr (X vs Y)\n"+p.name_x+'\n'+p.name_h,
#                                   dir=p.tag, mode='row-wise', nz_mode='first'
#                                   )


print('\n> Corr between G and Y')
hist = scimpute.hist_2matrix_corr(G.values, Y.values,
                                  title="Hist Gene-Corr (G vs Y)\n"+p.name_m+'\n'+p.name_h,
                                  dir=p.tag, mode='column-wise', nz_mode='ignore'
                                  )

# hist = scimpute.hist_2matrix_corr(G.values, Y.values,
#                                   title="Hist Cell-Corr (G vs Y)\n"+p.name_m+'\n'+p.name_h,
#                                   dir=p.tag, mode='row-wise', nz_mode='ignore'
#                                   )

hist = scimpute.hist_2matrix_corr(G.values, Y.values,
                                  title="Hist nz1-Gene-Corr (G vs Y)\n"+p.name_m+'\n'+p.name_h,
                                  dir=p.tag, mode='column-wise', nz_mode='first'
                                  )
#
# hist = scimpute.hist_2matrix_corr(G.values, Y.values,
#                                   title="Hist nz1-Cell-Corr (G vs Y)\n"+p.name_m+'\n'+p.name_h,
#                                   dir=p.tag, mode='row-wise', nz_mode='first'
#                                   )


# MSE CALCULATION
print('\n> MSE Calculation')
max_h, min_h = scimpute.max_min_element_in_arrs([Y.values])
print('Max in Y is {}, Min in Y is{}'.format(max_h, min_h))
max_m, min_m = scimpute.max_min_element_in_arrs([G.values])
print('Max in G is {}, Min in G is{}'.format(max_m, min_m))

mse1_omega = scimpute.mse_omega(Y, X)
mse1_omega = round(mse1_omega, 7)
print('mse1_omega between Y and X: ', mse1_omega)

mse2_omega = scimpute.mse_omega(Y, G)
mse2_omega = round(mse2_omega, 7)
print('mse2_omega between Y and G: ', mse2_omega)

mse2 = scimpute.mse(Y, G)
mse2 = round(mse2, 7)
print('MSE2 between Y and G: ', mse2)


#  VISUALIZATION OF DFS, todo clustering based on Y
print('\n> Visualization of dfs')
max, min = scimpute.max_min_element_in_arrs([Y.values, G.values, X.values])
scimpute.heatmap_vis(Y.values,
                     title='Y ({})'.format(p.name_h),
                     xlab='genes\nMSE1_OMEGA(Y vs X)={}'.format(mse1_omega),
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)

scimpute.heatmap_vis(G.values,
                     title='G ({})'.format(p.name_m),
                     xlab='genes\nMSE2(Y vs G)={}'.format(mse2),
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)

scimpute.heatmap_vis(X.values,
                     title='X ({})'.format(p.name_x),
                     xlab='genes',
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)


# Gene-Gene in G, X, Y
print('\n> Gene-gene relationship (Y, X, G), before/after inference')
gene_pair_dir = p.tag+'/pairs'
List = p.pair_list
scimpute.gene_pair_plot(Y, list=List, tag='(Y) '+p.tag, dir=gene_pair_dir)
scimpute.gene_pair_plot(X, list=List, tag='(X) '+p.tag, dir=gene_pair_dir)
scimpute.gene_pair_plot(G, list=List, tag='(G) '+p.tag, dir=gene_pair_dir)


# G vs Y, G vs X
print("\n> G vs Y, G vs X")
gene_dir = p.tag+'/genes'
for j in p.gene_list:
    try:
        print('for ', j)
        Y_j = Y.ix[:, j]
        M_j = G.ix[:, j]
        X_j = X.ix[:, j]
    except KeyError:
        print('KeyError: the gene index does not exist')
        continue

    scimpute.scatterplot2(M_j, Y_j, range='same',
                          title=str(str(j) + ' (M_vs_Y) ' + p.tag),
                          xlabel='Ground Truth (G)',
                          ylabel='Prediction (Y)',
                          dir=gene_dir
                          )
    scimpute.scatterplot2(M_j, X_j, range='same',
                          title=str(str(j) + ' (M_vs_X) ' + p.tag),
                          xlabel='Ground Truth (G)',
                          ylabel='Input (X)',
                          dir=gene_dir
                         )


# discretized plots
print('\n\n# Start discrete plots..')
Y = scimpute.df_exp_discretize_log(Y)
# Gene-Gene in G, X, Y
print('\n> Discrete Gene-gene relationship in Y')
gene_pair_dir = p.tag+'/pairs_discrete'
List = p.pair_list
scimpute.gene_pair_plot(Y, list=List, tag='(Y_discrete) '+p.tag, dir=gene_pair_dir)

# G vs Y, G vs X
print("\n> Discrete Y vs G")
gene_dir = p.tag+'/genes_discrete'
for j in p.gene_list:
    try:
        print('for ', j)
        Y_j = Y.ix[:, j]
        M_j = G.ix[:, j]
        X_j = X.ix[:, j]
    except KeyError:
        print('KeyError: the gene index does not exist')
        continue

    scimpute.scatterplot2(M_j, Y_j, range='same',
                          title=str(str(j) + ' (M_vs_Y_discrete) ' + p.tag),
                          xlabel='Ground Truth (G)',
                          ylabel='Prediction (Y)',
                          dir=gene_dir
                          )
    scimpute.scatterplot2(M_j, X_j, range='same',
                          title=str(str(j) + ' (M_vs_X_discrete) ' + p.tag),
                          xlabel='Ground Truth (G)',
                          ylabel='Input (X)',
                          dir=gene_dir
                         )

# # gene MSE
# j = 0
# input_j = Y.ix[:, j:j+1].values
# pred_j = h.ix[:, j:j+1].values
# groundTruth_j = G.ix[:, j:j+1].values
#
# mse_j_input = ((pred_j - input_j) ** 2).mean()
# mse_j_groundTruth = ((pred_j - groundTruth_j) ** 2).mean()
#
# matrix MSE

# Clustmap of weights, bottle-neck-activations (slow on GPU, moved to CPU)
# os.system('for file in ./{}/*npy; do python -u weight_visualization.py $file {}; done'.format(p.stage, p.stage))


# # Factors Affecting Gene Prediction todo: validate
# print('Mean and Var are calculated from Y')
# gene_corr = scimpute.gene_corr_list(G.values, Y.values)
# gene_mse = scimpute.gene_mse_list(G.values, Y.values)
# gene_mean_expression = G.sum(axis=0).values / G.shape[1]  # sum for each column
# gene_nz_rate = scimpute.gene_nz_rate_list(G.values)
# gene_var = scimpute.gene_var_list(G.values)
# gene_nzvar = scimpute.gene_nzvar_list(G.values)
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
