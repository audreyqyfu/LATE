#!/usr/bin/python
# read gene expression matrix
# into format [gene, cell]
# 1. RPM transformation: divide by lib-size, scale back to 1M
# 2. log(rpm+1) transformation

import numpy as np
import pandas as pd
import scimpute
import sys
import matplotlib
matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt

# get argv #
print('usage: <data_RpmNorm_LogTrans.py> <file.csv/hd5> <cell_row/gene_row> <out_name(x.rpm.log.hd5)>')
print('cmd typed:', sys.argv)
if len(sys.argv) != 4:
    raise Exception('num args err')

file = str(sys.argv[1])
matrix_mode = str(sys.argv[2])
outname = str(sys.argv[3])

# Read data into [gene, cell]
if matrix_mode == 'cell_row':
    if file.endswith('.csv'):
        df = scimpute.read_csv(file).transpose()
    elif file.endswith('.hd5'):
        df = scimpute.read_hd5(file).transpose()
    else:
        raise Exception('file extension error: not hd5/csv')
elif matrix_mode == 'gene_row':
    if file.endswith('.csv'):
        df = scimpute.read_csv(file)
    elif file.endswith('.hd5'):
        df = scimpute.read_hd5(file)
    else:
        raise Exception('file extension error: not hd5/csv')
else:
    raise Exception('cmd err in the argv[2]')

# summary
nz_rate_df = scimpute.nnzero_rate_df(df)
print('df.shape, [gene, cell]:', df.shape)
print('nz_rate: {}'.format(round(nz_rate_df, 3)))
print(df.ix[0:3, 0:3])

# lib-size per million normalization
df = scimpute.df_normalization(df)
print('after normalization')
print(df.ix[0:3, 0:3])
read_per_gene = df.sum(axis=1)
read_per_cell = df.sum(axis=0)
print('sum_reads_per_gene:', read_per_gene[0:3])
print('sum_reads_per_cell:', read_per_cell[0:3])

# log(tpm+1) transformation
df = scimpute.df_log_transformation(df)
print('after log transformation')
print(df.ix[0:3, 0:3])

# save
print('saving output to:', outname)
scimpute.save_hd5(df, outname)
print('finished')