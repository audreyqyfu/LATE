#!/usr/bin/python
import pandas as pd
import time
import sys

def save_hd5(df, out_name):
    tic = time.time()
    df.to_hdf(out_name, key='null', mode='w', complevel=9, complib='blosc')
    toc = time.time()
    print("saving" + out_name + " took {:.1f} seconds".format(toc - tic))

print('usage: python data_select_genes_cells.py df_big.hd5  df_small.hd5 outname')
print('assume gene(row) cell(columns), also gene_row inside code')

if len(sys.argv) is not 4:
    raise Exception("error: the num of arguments not correct")
else:
    print('running:')
    big_name = str(sys.argv[1])  # big.hd5
    small_name = str(sys.argv[2])  # small.hd5
    out_name = str(sys.argv[3])  # big.small.hd5

print("> command: ", sys.argv)

df_small = pd.read_hdf(small_name)
print('small df:\n', df_small.ix[0:5, 0:5])
print(df_small.shape)

df_big = pd.read_hdf(big_name)
print('big df\n', df_big.ix[0:5, 0:5])
print(df_big.shape)

df_selected = df_big.loc[df_small.index, df_small.columns]
print('selected big df from small df:', df_selected.ix[0:5, 0:5])
print(df_selected.shape)

print('saving ', out_name)
save_hd5(df_selected, out_name)


