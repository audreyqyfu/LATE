#!/usr/bin/python
import pandas as pd
import time
import sys
import scimpute

print('usage: python data_select_genes.py df_big.hd5  df_small.hd5 outname')
print('assume gene(row) cell(columns), also gene_row inside code')

if len(sys.argv) is not 4:
    raise Exception("error: the num of arguments not correct")
else:
    print('running:')
    print(sys.argv)
    big_name = str(sys.argv[1])  # big.hd5
    small_name = str(sys.argv[2])  # small.hd5
    out_name = str(sys.argv[3])  # big.small.hd5

# read
print('read big-df..')
df_big = pd.read_hdf(big_name)
nz_big = scimpute.nnzero_rate_df(df_big)
print('nz_rate big-df: ', nz_big)

print('read small-df..')
df_small = scimpute.read_hd5(small_name)
nz_small = scimpute.nnzero_rate_df(df_small)
print('nz_rate small_df: ', nz_small)


# Remove .x from ID
# df_big.index = df_big.index.to_series().astype(str).str.replace(r'\.[0-9]*','').astype(str)
# print('because the index is different, remove the appendix')
# print('big df after changing index', df_big.ix[0:5, 0:5])


print('df_big index is unique? {}'.format(df_big.index.is_unique))
print('df_small index is unique? {}'.format(df_small.index.is_unique))


# SELECT
print('selecting..')
df_selected = df_big.ix[df_small.index]
# Check null, fill zeros
null_gene_num = df_selected.ix[:, 1].isnull().sum()
print('there are {} genes from small-df not found in big-df'.format(null_gene_num))
df_selected = df_selected.fillna(value=0)
print('those N.A. in selected-df has been filled with zeros')
null_gene_num2 = df_selected.ix[:, :].isnull().sum().sum()
print('Now, there are {} null values in selected-df'.format(null_gene_num2))

nz_selected = scimpute.nnzero_rate_df(df_selected)
print('nz_rate output: ', nz_selected)


# Finish
print('selected big df from small df, after fillna:\n', df_selected.ix[0:5, 0:5])
print('shape: ', df_selected.shape)

scimpute.save_hd5(df_selected, out_name)
print('Finished')

