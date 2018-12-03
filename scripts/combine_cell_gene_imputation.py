#!/usr/bin/python
import matplotlib
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

home = os.environ['HOME']

# READ CMD
print('''
usage: python -u combine_cell_gene_imputation.py gene_info_imputation(cell_row) cell_info_imputation(cell_row) 
input_data(cell_row)

select the best prediction for each gene based on gene-level MSE_NZ (smaller better)
''')

if len(sys.argv) != 4:
    raise Exception ('cmd arguments wrong')

print('cmd: ', sys.argv)

# READ DATA  (into cell_row, for consistancy with imputation.hd5)
file_gene = sys.argv[1]
file_cell = sys.argv[2]
file_input = sys.argv[3]

print("> READ DATA..")  # todo: add support for h5 sparse
Yg = scimpute.read_data_into_cell_row(file_gene, 'cell_row')
Yc = scimpute.read_data_into_cell_row(file_cell, 'gene_row')

X = scimpute.read_data_into_cell_row(file_input, 'cell_row')
X = scimpute.df_transformation(X, transformation='log10')

print('aligning index to X')
Yg = Yg.reindex(X.index)
Yc = Yc.reindex(X.index)

# test mode
if 0:
    Yg = Yg.iloc[0:200, 0:100]
    Yc = Yc.iloc[0:200, 0:100]
    X = X.iloc[0:200, 0:100]

# INPUT SUMMART
print('Yg.shape', Yg.shape, '\n', Yg.iloc[0:2, 0:2])
print('Yc.shape', Yc.shape, '\n', Yc.iloc[0:2, 0:2])
print('X.shape', X.shape, '\n', X.iloc[0:2, 0:2])


# COMBINE BY SMALLER MSE: cell_row assumed for DFs(Y and X)

# calculate mse
print('calculating mse_nz gene-info (Yg vs X):')
mse_Yg = scimpute.gene_mse_nz_from_df(Yg, X)
print(mse_Yg.mean())

print('calculating mse_nz cell-info (Yc vs X):')
mse_Yc = scimpute.gene_mse_nz_from_df(Yc, X)
print(mse_Yc.mean())

# combine by smaller mse
print('combining by smaller mse')
Y_mse_combined = scimpute.combine_gene_imputation_of_two_df(Yg, Yc, mse_Yg, mse_Yc, mode='smaller')
scimpute.save_hd5(Y_mse_combined, 'imputation.combined_by_mse.hd5')

# calculate mse
print('calculating mse_nz combined (Y_mse_combined vs X):')
mse_Y_combined = scimpute.gene_mse_nz_from_df(Y_mse_combined, X)
print('mse_nz between Y_combined and X:', mse_Y_combined.mean())


# # STD of genes in Y and X
# print('calculating STD for Y and X')
# x_std, yg_std = scimpute.nz_std(X, Yg)
# x_std, yc_std = scimpute.nz_std(X, Yc)
#
# std_df = pd.DataFrame(data=list(zip(x_std, yg_std[x_std.index], yc_std[x_std.index],
#                                     yg_std/x_std, yc_std/x_std, yg_std/yc_std)),
#                       index=x_std.index,
#                       columns=['X_std', 'Yg_std', 'Yc_std',
#                                'Yg_X_ratio', 'Yc_X_ratio', 'Yg_Yc_ratio'])
# std_df.head()
# std_df.to_csv('std_df.csv')
# print('std_df.csv saved')
#
# # HIST OF STD
# fig, ax = plt.subplots()
#
# ax.hist(std_df.loc[:, 'Yg_X_ratio'], bins=100, label='SD(Yg)/SD(X)', density=True, alpha=0.7)
# ax.hist(std_df.loc[:, 'Yc_X_ratio'], bins=100, label='SD(Yc)/SD(X)', density=True, alpha=0.7)
# ax.legend()
# fig.savefig('hist_std_ratio.png', bbox_inches='tight')
# plt.show()
# plt.close()
#
#
# fig, ax = plt.subplots()
# ax.hist(std_df.loc[:, 'Yg_std'], bins=100, label='SD(Yg)', density=True, alpha=0.7)
# ax.hist(std_df.loc[:, 'Yc_std'], bins=100, label='SD(Yc)', density=True, alpha=0.7)
# ax.hist(std_df.loc[:, 'X_std'], bins=100, label='SD(X)', density=True, alpha=0.7)
# ax.legend()
# plt.show()
# fig.savefig('hist_std.png', bbox_inches='tight')
# plt.close()

# HIST OF MSE
fig, ax = plt.subplots()
ax.hist(mse_Yg.values.astype('float'), bins=100, label='GENE_MSE_NZ(Yg vs X)', density=True, alpha=0.7)
ax.hist(mse_Yc.values.astype('float'), bins=100, label='GENE_MSE_NZ(Yc vs X)', density=True, alpha=0.7)
ax.legend()
fig.savefig('hist_gene_mse.png', bbox_inches='tight')
plt.close()
#
# # TOP GENES DIFFERENT IN Yc and Yg
# num_genes = 10
# _ = std_df.sort_values(by='Yg_Yc_ratio', ascending=False).head(num_genes)
# print('## sort by Yg_Yc_ratio (Yg better):\n', _)
# Yg_better_genes = _.index
#
# _ = std_df.sort_values(by='Yg_Yc_ratio', ascending=True).head(num_genes)
# print('## sort by Yg_Yc_ratio (Yc_better):\n', _)
# Yc_better_genes = _.index
#
# ## PLT TOP GENES
# gene_plt_dir = 'gene_feature_better_genes'
#
# for gene in Yg_better_genes:
#     print(gene)
#     scimpute.scatterplot2(X.loc[:, gene],  Yg.loc[:,gene],
#                           range='same',
#                           title = gene + ' (Gene-feature))',
#                           xlabel='Ground Truth, SD(Y)/SD(X):' + str(round(std_df.loc[gene,'Yg_X_ratio'], 3)),
#                           ylabel='Imputation (Gene-feature)',
#                           dir=gene_plt_dir)
#     scimpute.scatterplot2(X.loc[:, gene],  Yc.loc[:,gene],
#                           range='same',
#                           title = gene + ' (Cell-feature))',
#                           xlabel='Ground Truth, SD(Y)/SD(X):' + str(round(std_df.loc[gene,'Yc_X_ratio'], 3)),
#                           ylabel='Imputation (Cell-feature)',
#                           dir=gene_plt_dir)
#
# ## PLT TOP GENES
# gene_plt_dir = 'cell_feature_better_genes'
#
# for gene in Yc_better_genes:
#     print(gene)
#     scimpute.scatterplot2(X.loc[:, gene],  Yg.loc[:,gene],
#                           range='same',
#                           title = gene + ' (Gene-feature))',
#                           xlabel='Ground Truth, SD(Y)/SD(X):' + str(round(std_df.loc[gene,'Yg_X_ratio'], 3)),
#                           ylabel='Imputation (Gene-feature)',
#                           dir=gene_plt_dir)
#     scimpute.scatterplot2(X.loc[:, gene],  Yc.loc[:,gene],
#                           range='same',
#                           title = gene + ' (Cell-feature))',
#                           xlabel='Ground Truth, SD(Y)/SD(X):' + str(round(std_df.loc[gene,'Yc_X_ratio'], 3)),
#                           ylabel='Imputation (Cell-feature)',
#                           dir=gene_plt_dir)