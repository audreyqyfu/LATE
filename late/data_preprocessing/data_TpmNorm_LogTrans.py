#!/usr/bin/python
# read gene expression matrix
# into format [gene, cell]
# 1. TPM transformation: divide by length, then by lib-size, scale back to 1M
# 2. log(tpm+1) transformation
# 3. Histogram of resulting matrix


import numpy as np
import pandas as pd
import scimpute
import sys
import matplotlib
matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt


# get argv #
print('usage: <data_TpmNorm_LogTrans.py> <file.csv/hd5> <cell_row/gene_row> <gene_length.txt> <out_name>')
print('gene_length.txt:\n\t')
print('warning: length not used yet, just rpm')
print('cmd typed:', sys.argv)
if len(sys.argv) != 5:
    raise Exception('num args err')

file = str(sys.argv[1])
matrix_mode = str(sys.argv[2])
length_file = str(sys.argv[3])
outname = str(sys.argv[4])

# Read data
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
print('input matrix.shape:', df.shape)
print('nz_rate: {}'.format(round(nz_rate_df, 3)))
print(df.ix[0:3, 0:3])

# Read length.txt
len_df = pd.read_csv(length_file, index_col=0, sep='\t')
len_df = len_df.ix[:, 1] - len_df.ix[:, 0]

# divide by length
for gene_id in df.index:
    df.ix[gene_id] = df.ix[gene_id]/len_df[gene_id]
print('after dividing *gene* length')
print(df.ix[0:3, 0:3])


# lib-size per million normalization
df = scimpute.df_normalization(df)
print('after normalization')
print(df.ix[0:3, 0:3])


# log(tpm+1) transformation
df = scimpute.df_log_transformation(df)
print('after log transformation')
print(df.ix[0:3, 0:3])


# histogram of filtered data
read_per_gene = df.sum(axis=1)
read_per_cell = df.sum(axis=0)

scimpute.save_hd5(df, outname)
