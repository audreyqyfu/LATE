#!/usr/bin/python
# read gene expression matrix (data)
# read cell id list (info)
# filter/keep cells in the id list

import numpy as np
import pandas as pd
import sys
import scimpute


# instructions
print('usage: data_sample_selection.py <file_name> <cell_row/gene_row> <cell_list_name> <out_prefix>',
      'file_name: name of the gene expression hd5 file',
      'cell_row/gene_row: indicates the matrix direction in input file',
      'cell_list_name: name of the list of cells to be filtered or excluded',
      'cell_list requirement: no headers, tab de-limited, first column cell_id',
      'out_prefix: the output name prefix',
      'output: [cell_row, gene_column].hd5',
      'two outputs, one with cells in list(yes), one excluding cells in list(no)',
      '', sep='\n')

# check num arguments
if len(sys.argv) is not 5:
    raise Exception("error: the num of arguments not correct")
else:
    print('running:')
    file_name = str(sys.argv[1])  # hd5
    matrix_type = str(sys.argv[2])  # cell_row, gene_row
    list_name = str(sys.argv[3])  # txt, first column as id
    out_prefix = str(sys.argv[4])  # gtex.xxx.muscle

print("> command: ", sys.argv)

# read data (so that output matrix is [sample, gene])
if matrix_type == 'cell_row':
    df = pd.read_hdf(file_name)
elif matrix_type == 'gene_row':
    df = pd.read_hdf(file_name).transpose()

# summary
print('input shape [samples, genes]:', df.shape,  df.ix[0:3, 0:2])

nz_rate_in = scimpute.nnzero_rate_df(df)
print('nz_rate_in: {}'.format(nz_rate_in))

# read list
list_df = pd.read_csv(list_name, index_col=0, sep='\t', header=None)
print('list:', list_df.shape, list_df.index)

# filter
df_yes = df.ix[list_df.index]
overlap = df.index.isin(list_df.index)
df_no = df.ix[~overlap]

print('matrix yes: ', df_yes.shape, df_yes.ix[0:3, 0:2])
print('matrix no: ', df_no.shape, df_no.ix[0:3, 0:2])

# output result dataframe
scimpute.save_hd5(df_yes, out_prefix+'_yes.hd5')
scimpute.save_hd5(df_no, out_prefix+'_no.hd5')