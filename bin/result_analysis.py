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
print('''
usage: python -u result_analysis.py params.py

reads Imputation(Y), Input(X) and Ground-truth(G), 
compare them and analysis the result

When no G is available, set X as G in params, so that the code can run
''')

if len(sys.argv) == 2:
    param_file = sys.argv[1]
    param_file = param_file.rstrip('.py')
    p = importlib.import_module(param_file)
else:
    raise Exception('cmd err')

# refresh folder
log_dir = './{}'.format(p.tag)
scimpute.refresh_logfolder(log_dir)

# READ DATA
print("> READ DATA..")
Y = scimpute.read_data_into_cell_row(p.fname_imputation, p.ori_imputation)
X = scimpute.read_data_into_cell_row(p.fname_input, p.ori_input)
G = scimpute.read_data_into_cell_row(p.fname_ground_truth, p.ori_ground_truth)

# Data Transformation for Y
print('> DATA TRANSFORMATION..')
Y = scimpute.df_transformation(Y.transpose(), transformation=p.transformation_imputation).transpose()
X = scimpute.df_transformation(X.transpose(), transformation=p.transformation_input).transpose()
G = scimpute.df_transformation(G.transpose(), transformation=p.transformation_ground_truth).transpose()

# subset/sort X, G to match Y
# todo: support sparse matrix
X = X.loc[Y.index, Y.columns]
G = G.loc[Y.index, Y.columns]

# TEST MODE OR NOT
if p.test_flag:
    print('in test mode')
    Y = Y.ix[0:p.m, 0:p.n]
    G = G.ix[0:p.m, 0:p.n]
    X = X.ix[0:p.m, 0:p.n]

# INPUT SUMMARY
print('\ninside this code, matrices are supposed to be transformed into cell_row')
print('Y:', p.fname_imputation, p.ori_imputation, p.transformation_imputation,
      '\n', Y.ix[0:20, 0:3])
print('X:', p.fname_input, p.ori_input, p.transformation_input,
      '\n', X.ix[0:20, 0:3])
print('G:', p.fname_ground_truth, p.ori_ground_truth, p.transformation_ground_truth,
      '\n', G.ix[0:20, 0:3])
print('Y.shape', Y.shape)
print('X.shape', X.shape)
print('G.shape', G.shape)

# HIST OF EXPRESSION
scimpute.hist_df(
    Y, xlab='expression', title='Imputation({})'.format(p.name_imputation),
    dir=p.tag)
scimpute.hist_df(
    X,  xlab='expression', title='Input({})'.format(p.name_input),
    dir=p.tag)
scimpute.hist_df(
    G,  xlab='expression', title='Ground_truth({})'.format(p.name_ground_truth),
    dir=p.tag)

# HIST OF CELL/GENE CORR
print('\n> Corr between X and Y')
print('GeneCorr: X shape: ', X.shape, 'Y shape: ', Y.shape)
hist = scimpute.hist_2matrix_corr(
    X.values, Y.values,
    title="Hist Gene-Corr-NZ (Input vs Imputation)\n{}\n{}".
        format(p.name_input, p.name_imputation),
    dir=p.tag, mode='column-wise', nz_mode='first'
)

print('CellCorr: X shape: ', X.shape, 'Y shape: ', Y.shape)
hist = scimpute.hist_2matrix_corr(
    X.values, Y.values,
    title="Hist Cell-Corr-NZ (Input vs Imputation)\n{}\n{}".
        format(p.name_input, p.name_imputation),
    dir=p.tag, mode='row-wise', nz_mode='first'
)

# todo: edit from there

print('\n> Corr between G and Y')
hist = scimpute.hist_2matrix_corr(G.values, Y.values,
                                  title="Hist Gene-Corr (G vs Y)\n"+p.name_ground_truth+'\n'+p.name_imputation,
                                  dir=p.tag, mode='column-wise', nz_mode='ignore'
                                  )

# hist = scimpute.hist_2matrix_corr(G.values, Y.values,
#                                   title="Hist Cell-Corr (G vs Y)\n"+p.name_ground_truth+'\n'+p.name_imputation,
#                                   dir=p.tag, mode='row-wise', nz_mode='ignore'
#                                   )

hist = scimpute.hist_2matrix_corr(G.values, Y.values,
                                  title="Hist nz1-Gene-Corr (G vs Y)\n"+p.name_ground_truth+'\n'+p.name_imputation,
                                  dir=p.tag, mode='column-wise', nz_mode='first'
                                  )
#
# hist = scimpute.hist_2matrix_corr(G.values, Y.values,
#                                   title="Hist nz1-Cell-Corr (G vs Y)\n"+p.name_ground_truth+'\n'+p.name_imputation,
#                                   dir=p.tag, mode='row-wise', nz_mode='first'
#                                   )


# MSE CALCULATION
print('\n> MSE Calculation')
max_y, min_y = scimpute.max_min_element_in_arrs([Y.values])
print('Max in Y is {}, Min in Y is{}'.format(max_y, min_y))
max_g, min_g = scimpute.max_min_element_in_arrs([G.values])
print('Max in G is {}, Min in G is{}'.format(max_g, min_g))

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
                     title='Y ({})'.format(p.name_imputation),
                     xlab='genes\nMSE1_OMEGA(Y vs X)={}'.format(mse1_omega),
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)

scimpute.heatmap_vis(G.values,
                     title='G ({})'.format(p.name_ground_truth),
                     xlab='genes\nMSE2(Y vs G)={}'.format(mse2),
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)

scimpute.heatmap_vis(X.values,
                     title='X ({})'.format(p.name_input),
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
        G_j = G.ix[:, j]
        X_j = X.ix[:, j]
    except KeyError:
        print('KeyError: the gene index does not exist')
        continue

    scimpute.scatterplot2(G_j, Y_j, range='same',
                          title=str(str(j) + ' (G_vs_Y) ' + p.tag),
                          xlabel='Ground Truth (G)',
                          ylabel='Prediction (Y)',
                          dir=gene_dir
                          )
    scimpute.scatterplot2(G_j, X_j, range='same',
                          title=str(str(j) + ' (G_vs_X) ' + p.tag),
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
        G_j = G.ix[:, j]
        X_j = X.ix[:, j]
    except KeyError:
        print('KeyError: the gene index does not exist')
        continue

    scimpute.scatterplot2(G_j, Y_j, range='same',
                          title=str(str(j) + ' (G_vs_Y_discrete) ' + p.tag),
                          xlabel='Ground Truth (G)',
                          ylabel='Prediction (Y)',
                          dir=gene_dir
                          )
    scimpute.scatterplot2(G_j, X_j, range='same',
                          title=str(str(j) + ' (G_vs_X_discrete) ' + p.tag),
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
