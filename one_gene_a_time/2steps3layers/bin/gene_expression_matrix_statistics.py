#!/usr/bin/python
# read gene expression matrix
# into format [gene, cell]
# 1. Filter
# 2. Histogram of reads/cell, reads/gene
# example usage:
# python gene_expression_matrix_statistics.py GSE72857.umitab.csv.gz row_gene 0 0 EMT.Raw

import numpy as np
import pandas as pd
import scimpute
import sys
import matplotlib
matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt


# get argv #
print('usage: <gene_expression_matrix_statistics.py> <file> <row_cell/row_gene> <reads/gene min> <reads/cell min> <out-tag>')
print('min included')
print('cmd typed:', sys.argv)
if len(sys.argv) != 6:
    raise Exception('num args err')

# read data
file = str(sys.argv[1])
matrix_mode = str(sys.argv[2])
if matrix_mode == 'row_cell':
    df = scimpute.read_csv(file).transpose()
elif matrix_mode == 'row_gene':
    df = scimpute.read_csv(file)
else:
    raise Exception('cmd err in the argv[2]')
print('matrix.shape:', df.shape)
print(df.ix[0:3, 0:3])

# get argv
gene_min = int(sys.argv[3])
cell_min = int(sys.argv[4])
tag = '(' + str(sys.argv[5]) +')'

# filter #
read_per_gene = df.sum(axis=1)
read_per_cell = df.sum(axis=0)
df_filtered = df.loc[(read_per_gene >= gene_min), (read_per_cell >= cell_min)]
print('filtered matrix: ', df_filtered.shape)
read_per_cell_filtered = df_filtered.sum(axis=1)
read_per_gene_filtered = df_filtered.sum(axis=0)

# histogram of filtered data
scimpute.hist_list(read_per_cell_filtered.values, xlab='counts/cell',
                   title='Histogram of counts per cell' + tag)
scimpute.hist_list(read_per_gene_filtered.values, xlab='counts/gene',
                   title='Histogram of counts per gene' + tag)
scimpute.hist_df(df_filtered, xlab='counts',
                 title='Histogram of counts in expression matrix' + tag)

# normalization and log(x+1) transformation
df_filtered_norm = scimpute.df_normalization(df_filtered)
df_filtered_norm_log = scimpute.df_log_transformation(df_filtered_norm)
read_per_cell_filtered_norm_log = df_filtered_norm_log.sum(axis=1)
read_per_gene_filtered_norm_log = df_filtered_norm_log.sum(axis=0)
# histogram of filtered_norm_log data
scimpute.hist_list(read_per_cell_filtered_norm_log.values, xlab='log10(normed-counts+1)/cell',
                   title='Histogram of log10(normed_counts+1) per cell' + tag)
scimpute.hist_list(read_per_gene_filtered_norm_log.values, xlab='log10(normed-counts+1)/gene',
                   title='Histogram of log10(normed_counts+1) per gene' + tag)
scimpute.hist_df(df_filtered_norm_log, xlab='log10(normed-counts+1)',
                 title='Histogram of expression matrix' + tag)


