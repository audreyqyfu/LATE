import pandas as pd
import numpy as np
import scimpute

# parameters #
list = [[1, 2], [100, 104], [101, 160], [107, 198],
        [201, 203], [204, 205], [210, 220],
        [401, 403], [440, 450],
        [601, 603], [660, 670],
        [801, 803], [880, 890],
        [4, 997], [101, 987], [109, 963],
        [600, 800], [204, 604], [205, 805]]


# For input data #
DF = scimpute.read_hd5('../data/v1-1-5-3/v1-1-5-3.E3.hd5')
DF = DF.transpose()
# corr
corrcoef_matrix_gene_wise0 = np.corrcoef(DF, rowvar=False)
scimpute.hist_arr_flat(corrcoef_matrix_gene_wise0, title="input.corr_gene_wise")
scimpute.heatmap_vis(corrcoef_matrix_gene_wise0, title="input.corr_gene_wise.heatmap.png", vmin=-1, vmax=1)
# scatterplot
for i,j in list:
    scimpute.scatterplot2(DF.ix[:,i], DF.ix[:,j], title="Gene"+str(i+1)+'vs Gene'+str(j+1)+' (input)',
                          xlabel='Gene'+str(i+1), ylabel='Gene'+str(j+1))



# for prediction #
df = scimpute.read_hd5('pre_train/imputation.step1.hd5')  # [cells, genes]
# corr
corrcoef_matrix_gene_wise = np.corrcoef(df, rowvar=False)
scimpute.hist_arr_flat(corrcoef_matrix_gene_wise, title="step1.imputation.corr_gene_wise")
scimpute.heatmap_vis(corrcoef_matrix_gene_wise, title="step1.imputation.corr_gene_wise.heatmap.png", vmin=-1, vmax=1)
# scatterplot of pairs of genes
for i,j in list:
    scimpute.scatterplot2(df.ix[:,i], df.ix[:,j], title="Gene"+str(i+1)+'vs Gene'+str(j+1)+' (step1, prediction)',
                          xlabel='Gene'+str(i+1), ylabel='Gene'+str(j+1))
