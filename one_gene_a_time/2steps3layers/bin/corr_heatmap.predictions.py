import pandas as pd
import numpy as np
import scimpute

# input data
df = scimpute.read_hd5('pre_train/imputation.step1.hd5')  # [cells, genes]

# calculate corr matrix
corrcoef_matrix_gene_wise = np.corrcoef(df, rowvar=False)

# histogram
scimpute.hist_arr_flat(corrcoef_matrix_gene_wise, title="step1.imputation.corr_gene_wise")

# heatmap of corr matrix
scimpute.heatmap_vis(corrcoef_matrix_gene_wise, title="step1.imputation.corr_gene_wise.heatmap.png")

# scatterplot of pairs of genes
scimpute.scatterplot2(df.ix[:,1], df.ix[:,2],  title='Gene2 vs Gene3(step1, prediction)')


## For input data ##

# input original data
DF = scimpute.read_hd5('../data/v1-1-5-3/v1-1-5-3.E3.hd5')
DF = DF.transpose()
scimpute.scatterplot2(DF.ix[:,1], DF.ix[:,2],  title='Gene2 vs Gene3 (input)')
corrcoef_matrix_gene_wise0 = np.corrcoef(DF, rowvar=False)
scimpute.hist_arr_flat(corrcoef_matrix_gene_wise0, title="input.corr_gene_wise")
scimpute.heatmap_vis(corrcoef_matrix_gene_wise0, title="input.corr_gene_wise.heatmap.png")
