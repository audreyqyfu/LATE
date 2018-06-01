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

# todo: save hist for boxplot across imputation methods?

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
print("> READ DATA..")  # todo: add support for h5 sparse
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

# STD of genes in Y and X
print('calculating STD for Y and X')
y_std_df = Y.std(axis=0)
x_std_df = X.std(axis=0)
g_std_df = G.std(axis=0)
std_ratio_yx_df = pd.DataFrame(data= y_std_df.values / x_std_df.values,
                            index=X.columns, columns=['std_ratio'])
std_ratio_yg_df = pd.DataFrame(data= y_std_df.values / g_std_df.values,
                            index=X.columns, columns=['std_ratio'])

std_min = min(y_std_df.min(), x_std_df.min(), g_std_df.min())
std_max = max(y_std_df.max(), x_std_df.max(), g_std_df.max())
scimpute.hist_df(
    y_std_df,
    xlab='STD(Imputation)', title='STD Imputation({})'.format(p.name_imputation),
    range=(std_min, std_max),
    dir=p.tag)
scimpute.hist_df(
    x_std_df,
    xlab='STD(Input)', title='STD Input({})'.format(p.name_input),
    range=(std_min, std_max),
    dir=p.tag)
scimpute.hist_df(
    g_std_df,
    xlab='STD(Ground Truth)', title='STD Ground Truth({})'.format(p.name_input),
    range=(std_min, std_max),
    dir=p.tag)
scimpute.hist_df(
    std_ratio_yx_df,
    xlab='Ratio STD(Imputation) div STD(Input)',
    title='Ratio STD Imputation div STD Input({})'.format(p.name_input),
    range=(std_min, std_max),  # todo remove hard code here
    dir=p.tag)
scimpute.hist_df(
    std_ratio_yg_df,
    xlab='Ratio STD(Imputation) div STD(Ground Truth)',
    title='Ratio STD Imputation div STD Ground Truth({})'.format(p.name_input),
    range=(std_min, std_max),
    dir=p.tag)

std_ratio_yx_df.to_csv('std_ratio_y_div_x.csv')
std_ratio_yg_df.to_csv('std_ratio_y_div_g.csv')


# HIST OF EXPRESSION
max_expression = max(G.values.max(), X.values.max(), Y.values.max())
min_expression = min(G.values.min(), X.values.min(), Y.values.min())
scimpute.hist_df(
    Y, xlab='expression', title='Imputation({})'.format(p.name_imputation),
    dir=p.tag, range=[min_expression, max_expression])
scimpute.hist_df(
    X,  xlab='expression', title='Input({})'.format(p.name_input),
    dir=p.tag, range=[min_expression, max_expression])
scimpute.hist_df(
    G,  xlab='expression', title='Ground_truth({})'.format(p.name_ground_truth),
    dir=p.tag, range=[min_expression, max_expression])

# HIST OF CELL/GENE CORR
print('\n> Corr between G and Y')
print('GeneCorr: G shape: ', G.shape, 'Y shape: ', Y.shape)
hist = scimpute.hist_2matrix_corr(
    G.values, Y.values,
    title="Hist Gene-Corr-NZ\n(Ground_truth vs Imputation)\n{}\n{}".
        format(p.name_ground_truth, p.name_imputation),
    dir=p.tag, mode='column-wise', nz_mode='first'  # or ignore
)

print('CellCorr: G shape: ', G.shape, 'Y shape: ', Y.shape)
hist = scimpute.hist_2matrix_corr(
    G.values, Y.values,
    title="Hist Cell-Corr-NZ\n(Ground_truth vs Imputation)\n{}\n{}".
        format(p.name_ground_truth, p.name_imputation),
    dir=p.tag, mode='row-wise', nz_mode='first'
)

# MSE CALCULATION
print('\n> MSE Calculation')
max_y, min_y = scimpute.max_min_element_in_arrs([Y.values])
print('Max in Y is {}, Min in Y is{}'.format(max_y, min_y))
max_g, min_g = scimpute.max_min_element_in_arrs([G.values])
print('Max in G is {}, Min in G is{}'.format(max_g, min_g))

mse1_omega = scimpute.mse_omega(Y, X)
mse1_omega = round(mse1_omega, 7)
print('MSE1_NZ between Imputation and Input: ', mse1_omega)

mse2_omega = scimpute.mse_omega(Y, G)
mse2_omega = round(mse2_omega, 7)
print('MSE2_NZ between Imputation and Ground_truth: ', mse2_omega)

mse2 = scimpute.mse(Y, G)
mse2 = round(mse2, 7)
print('MSE2 between Imputation and Ground_truth: ', mse2)



#  VISUALIZATION OF DFS
print('\n> Visualization of dfs')
max, min = scimpute.max_min_element_in_arrs([Y.values, G.values, X.values])
scimpute.heatmap_vis(Y.values,
                     title='Imputation ({})'.format(p.name_imputation),
                     xlab='genes\nMSE_NZ(Imputation vs Input)={}'.format(mse1_omega),
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)

scimpute.heatmap_vis(X.values,
                     title='Input ({})'.format(p.name_input),
                     xlab='genes',
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)

scimpute.heatmap_vis(G.values,
                     title='Ground_truth ({})'.format(p.name_ground_truth),
                     xlab='genes\nMSE_NZ(Imputation vs Ground_truth)={}'.format(
                         mse2_omega),
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)

# PCA, tSNE
print('\n> PCA and TSNE')
if p.cluster_file is not None:
    cluster_info = scimpute.read_data_into_cell_row(p.cluster_file)
    # cluster_info = cluster_info.astype('str')
else:
    cluster_info = None

tsne_df_y = scimpute.pca_tsne(df_cell_row=Y, cluster_info=cluster_info,
                            title=p.name_imputation, dir=p.tag)
tsne_df_x = scimpute.pca_tsne(df_cell_row=X, cluster_info=cluster_info,
                            title=p.name_input, dir=p.tag)
tsne_df_g = scimpute.pca_tsne(df_cell_row=G, cluster_info=cluster_info,
                            title=p.name_ground_truth, dir=p.tag)


# Gene/Pair plots
print('\n> Gene-pair relationship (Y, X, G), before/after inference')
gene_pair_dir = p.tag+'/pairs'
List = p.pair_list
scimpute.gene_pair_plot(Y, list=List, tag='(Imputation)', dir=gene_pair_dir)
scimpute.gene_pair_plot(X, list=List, tag='(Input)', dir=gene_pair_dir)
scimpute.gene_pair_plot(G, list=List, tag='(Ground_truth)', dir=gene_pair_dir)

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
                          title=str(str(j) + '\n(Ground_truth vs Imputation) '),
                          xlabel='Ground Truth',
                          ylabel='Imputation',
                          dir=gene_dir
                          )
    scimpute.scatterplot2(G_j, X_j, range='same',
                          title=str(str(j) + '\n(Ground_truth vs Input) '),
                          xlabel='Ground Truth',
                          ylabel='Input',
                          dir=gene_dir
                          )


# Discrete (changed Y, only use at end of script
Y = scimpute.df_exp_discretize_log(Y)

print('\n> Discrete Gene-pair relationship in Y')
gene_pair_dir = p.tag+'/pairs_discrete'
List = p.pair_list
scimpute.gene_pair_plot(Y, list=List, tag='(Imputation Discrete) ',
                        dir=gene_pair_dir)

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
                          title=str(str(j) + '\n(Ground_truth vs Imputation) '),
                          xlabel='Ground Truth',
                          ylabel='Imputation',
                          dir=gene_dir
                          )
    scimpute.scatterplot2(G_j, X_j, range='same',
                          title=str(str(j) + '\n(Ground_truth vs Input) '),
                          xlabel='Ground Truth',
                          ylabel='Input',
                          dir=gene_dir
                          )





























# # weight clustmap
# os.system(
#     '''for file in {0}/*npy
#     do python -u weight_clustmap.py $file {0}
#     done'''.format(p.stage)
# )

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
