#!/usr/bin/python
# read gene expression matrix
# into format [cell, gene]
# 1. Filter
# 2. Histogram of reads/cell, reads/gene
# example usage:
# python gene_expression_matrix_statistics.py GSE72857.umitab.csv.gz row_gene 0 0 EMT.Raw

import numpy as np
import pandas as pd
import scimpute
import sys

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
    df = scimpute.read_csv(file)
elif matrix_mode == 'row_gene':
    df = scimpute.read_csv(file).transpose()
else:
    raise Exception('cmd err in the argv[2]')
print('matrix.shape:', df.shape)
print(df.ix[0:3, 0:3])

# get argv
gene_min = int(sys.argv[3])
cell_min = int(sys.argv[4])
tag = '(' + str(sys.argv[5]) +')'

# filter #
read_per_cell = df.sum(axis=1)
read_per_gene = df.sum(axis=0)
df_filtered = df.loc[(read_per_cell >= cell_min), (read_per_gene >= gene_min)]
print('filtered matrix: ', df_filtered.shape)
read_per_gene_filtered = df_filtered.sum(axis=1)
read_per_cell_filtered = df_filtered.sum(axis=0)

# histogram
scimpute.hist_list(read_per_cell_filtered.values, xlab='reads/cell',
                   title='Histogram of reads per cell' + tag)
scimpute.hist_list(read_per_gene_filtered.values, xlab='reads/gene',
                   title='Histogram of reads per gene' + tag)
