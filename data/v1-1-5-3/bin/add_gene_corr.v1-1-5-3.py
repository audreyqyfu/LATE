#!/usr/bin/python
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

sys.path.append('/Users/rui/AUDREY_LAB/imputation/bin/py')
import scimpute

# read data
# df = scimpute.read_csv('../splat.OneGroup.csv')
# df = df.astype('float64')
# scimpute.save_hd5(df, '../splat.OneGroup.hd5')
# df = scimpute.read_hd5('../splat.OneGroup.hd5')
df = scimpute.read_hd5('../normalization/splat.OneGroup.norm.log.hd5')  # before
df2 = df.copy()  # after

# summary input
[nrow, ncol] = df.shape  # genes, cells

# SD
sd_log = []
for i in np.arange(nrow):
    sd_temp = np.std(df.ix[i])
    sd_log.append(sd_temp)

scimpute.hist_list(sd_log, xlab='std', title='sd_of_genes_before')
scimpute.hist_list(df.ix[999], title='gene1000 in different cells (before)')
scimpute.hist_list(df.ix[:, 1], title='cell2 in different genes (before)')

# generate new data with linear model
J = [200, 400, 600, 800]
BETA = [0.46, 0.85, 1.3, 1.7]
np.random.seed(810)
for j, beta in zip(J, BETA):
    ref_arr = np.copy(df.ix[j:j + 1].values)  # np.copy very important
    sd = np.median(sd_log)

    print("parameters: ",
          'j:', j,
          "beta: ", beta,
          "sd: ", sd)

    for i in np.arange(j, j+200):
        rand_row_arr = np.random.normal(0, sd, [1, ncol])
        df2.ix[i:i + 1] = ref_arr * beta + rand_row_arr  # only fast when same dtypes

# replace negative values with zero
df2[df2 < 0] = 0

# output
scimpute.save_hd5(df2, 'gene_corr_added' + str(beta) + '.hd5')

# summary 1
# df
print('np.corrcoef_gene_wise_before: \n', np.round(np.corrcoef(df.ix[497:504]), 3))
print('np.corrcoef_gene_wise_after: \n', np.round(np.corrcoef(df2.ix[497:504]), 3))
scimpute.heatmap_vis(df.values.T, title='df.before', vmin=0, vmax=6.5, xlab='genes', ylab='cells')
scimpute.heatmap_vis(df2.values.T, title='df.after', vmin=0, vmax=6.5, xlab='genes', ylab='cells')
# corr gene-wise
corrcoef_matrix_gene_wise = np.corrcoef(df, rowvar=True)
scimpute.heatmap_vis(corrcoef_matrix_gene_wise, title='corr_gene_wise.before.vis.png')
corrcoef_matrix_gene_wise2 = np.corrcoef(df2, rowvar=True)
scimpute.heatmap_vis(corrcoef_matrix_gene_wise2, title='corr_gene_wise.after.vis.png')
# corr cell-wise
corrcoef_matrix_cell_wise = np.corrcoef(df.ix[:, 0:400], rowvar=False)
median_corr = round(np.median(corrcoef_matrix_cell_wise), 3)
scimpute.heatmap_vis(corrcoef_matrix_cell_wise, title='corr_cell_wise.before.vis.png',
                     vmin=-1, vmax=1, cmap="PiYG",
                     xlab='top 200 cells, median corr:' + str(median_corr))

corrcoef_matrix_cell_wise2 = np.corrcoef(df2.ix[:, 0:400], rowvar=False)
median_corr2 = round(np.median(corrcoef_matrix_cell_wise2), 3)
scimpute.heatmap_vis(corrcoef_matrix_cell_wise2, title='corr_cell_wise.after.vis.png',
                     vmin=-1, vmax=1, cmap="PiYG",
                     xlab='top 200 cells, median corr:' + str(median_corr2))

# hist
scimpute.hist_arr_flat(corrcoef_matrix_gene_wise, title="corr_gene_wise_before")
scimpute.hist_arr_flat(corrcoef_matrix_gene_wise2, title='corr_gene_wise_after')
scimpute.hist_arr_flat(corrcoef_matrix_cell_wise, title='corr_cell_wise_before')
scimpute.hist_arr_flat(corrcoef_matrix_cell_wise2, title='corr_cell_wise_after')

# hist
# SD
sd_log2 = []
for i in np.arange(nrow):
    sd_temp = np.std(df2.ix[i])
    sd_log2.append(sd_temp)

scimpute.hist_list(sd_log2, xlab='std', title='sd_of_genes_after')
scimpute.hist_list(df2.ix[999], title='gene1000 in different cells (after)')
scimpute.hist_list(df2.ix[:, 1], title='cell2 in different genes (after)')

# boxplot
