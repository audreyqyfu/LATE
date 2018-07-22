#!/usr/bin/python
# read gene expression matrix
# into format [gene, cell]
# 1. Filter
# 2. Stat: Histogram of reads/cell, reads/gene
# example usage:
# python data_filter_stat.py GSE72857.umitab.csv.gz row_gene 0 0 EMT.Raw.Gene0.Cell0

import numpy as np
import pandas as pd
import scimpute
import sys
import matplotlib
matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt


# get argv #
print('usage: <data_filter_stat.py> <file.csv/hd5> <cell_row/gene_row> <reads/gene min> <reads/cell min> <out-tag>')
print('min included')
print('cmd typed:', sys.argv)
if len(sys.argv) != 6:
    raise Exception('num args err')

file = str(sys.argv[1])
matrix_mode = str(sys.argv[2])

gene_min = int(sys.argv[3])
cell_min = int(sys.argv[4])
tag0 = str(sys.argv[5])
tag = '(' + str(sys.argv[5]) +')'

# read data
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


# filter #
read_per_gene = df.sum(axis=1)
read_per_cell = df.sum(axis=0)
df_filtered = df.loc[(read_per_gene >= gene_min), (read_per_cell >= cell_min)]
nz_rate_filtered = scimpute.nnzero_rate_df(df_filtered)
print('filtered matrix : ', df_filtered.shape)
print('nz_rate:', nz_rate_filtered)
print(df_filtered.ix[0:3, 0:3])
scimpute.save_hd5(df_filtered, tag0+'.hd5')


# histogram of filtered data
read_per_gene_filtered = df_filtered.sum(axis=1)
read_per_cell_filtered = df_filtered.sum(axis=0)
scimpute.hist_list(read_per_cell_filtered.values, xlab='counts/cell',
                   title='Histogram of counts per cell' + tag)
scimpute.hist_list(read_per_gene_filtered.values, xlab='counts/gene',
                   title='Histogram of counts per gene' + tag)
scimpute.hist_df(df_filtered, xlab='counts',
                 title='Histogram of counts in expression matrix' + tag)


# histogram of log transformed filtered data
scimpute.hist_list(scimpute.df_log_transformation(read_per_cell_filtered).values, xlab='log10(count+1)/cell',
                   title='Histogram of log10(count+1) per cell' + tag)
scimpute.hist_list(scimpute.df_log_transformation(read_per_gene_filtered).values, xlab='log10(count+1)/gene',
                   title='Histogram of log10(count+1) per gene' + tag)
scimpute.hist_df(scimpute.df_log_transformation(df_filtered), xlab='log10(count+1)',
                 title='Histogram of log10(count+1) in expression matrix' + tag)