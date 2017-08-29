import pandas as pd
import numpy as np
import scimpute

## For input data ##

# input original data
DF = scimpute.read_hd5('../data/v1-1-5-3/v1-1-5-3.E3.hd5')
DF = DF.transpose()
# corr
corrcoef_matrix_gene_wise0 = np.corrcoef(DF, rowvar=False)
scimpute.hist_arr_flat(corrcoef_matrix_gene_wise0, title="input.corr_gene_wise")
scimpute.heatmap_vis(corrcoef_matrix_gene_wise0, title="input.corr_gene_wise.heatmap.png", vmin=-1, vmax=1)
# scatterplot
scimpute.scatterplot2(DF.ix[:,1], DF.ix[:,2],  title='Gene2 vs Gene3 (input)')
scimpute.scatterplot2(DF.ix[:,4], DF.ix[:,998],  title='Gene5 vs Gene998 (input)')
scimpute.scatterplot2(DF.ix[:,5], DF.ix[:,8],  title='Gene6 vs Gene9 (input)')
scimpute.scatterplot2(DF.ix[:,201], DF.ix[:,203],  title='Gene202 vs Gene204 (input)')
scimpute.scatterplot2(DF.ix[:,204], DF.ix[:,205],  title='Gene205 vs Gene206 (input)')
scimpute.scatterplot2(DF.ix[:,401], DF.ix[:,403],  title='Gene402 vs Gene404 (input)')
scimpute.scatterplot2(DF.ix[:,601], DF.ix[:,603],  title='Gene602 vs Gene604 (input)')
scimpute.scatterplot2(DF.ix[:,801], DF.ix[:,803],  title='Gene802 vs Gene804 (input)')



## for prediction ##

# input data
df = scimpute.read_hd5('pre_train/imputation.step1.hd5')  # [cells, genes]

# corr
corrcoef_matrix_gene_wise = np.corrcoef(df, rowvar=False)
scimpute.hist_arr_flat(corrcoef_matrix_gene_wise, title="step1.imputation.corr_gene_wise")
scimpute.heatmap_vis(corrcoef_matrix_gene_wise, title="step1.imputation.corr_gene_wise.heatmap.png", vmin=-1, vmax=1)

# scatterplot of pairs of genes
scimpute.scatterplot2(df.ix[:,1], df.ix[:,2],  title='Gene2 vs Gene3(step1, prediction)')
scimpute.scatterplot2(df.ix[:,4], df.ix[:,998],  title='Gene5 vs Gene998(step1, prediction)')
scimpute.scatterplot2(df.ix[:,5], df.ix[:,8],  title='Gene6 vs Gene9 (step1, prediction)')
scimpute.scatterplot2(df.ix[:,201], df.ix[:,203],  title='Gene202 vs Gene204 (step1, prediction)')
scimpute.scatterplot2(df.ix[:,204], df.ix[:,205],  title='Gene205 vs Gene206 (step1, prediction)')
scimpute.scatterplot2(df.ix[:,401], df.ix[:,403],  title='Gene402 vs Gene404 (step1, prediction)')
scimpute.scatterplot2(df.ix[:,601], df.ix[:,603],  title='Gene602 vs Gene604 (step1, prediction)')
scimpute.scatterplot2(df.ix[:,801], df.ix[:,803],  title='Gene802 vs Gene804 (step1, prediction)')






