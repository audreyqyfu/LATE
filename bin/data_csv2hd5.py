#!/usr/bin/python
# read csv and tranform into hd5 format

import pandas as pd
import sys
import time

print("usage: python data_csv2hd5.py inname.csv outname.hd5")
if len(sys.argv) == 3:
    in_name = sys.argv[1]
    out_name = sys.argv[2]
    print('usage: {}'.format(sys.argv))
else:
    raise Exception("cmd error")

def read_csv(fname):
    '''read_csv into pd.df, assuming index_col=0, and header=True'''
    print('reading ', fname)
    tic = time.time()
    df = pd.read_csv(fname, index_col=0)
    # print("read matrix: [genes, cells]")
    print(df.shape)
    # print(df.axes)
    if df.shape[0] > 1 and df.shape[1] > 1:
        print(df.ix[0:2, 0:2])
    toc = time.time()
    print("reading took {:.1f} seconds".format(toc - tic))
    return (df)

df = read_csv(in_name)

def save_hd5(df, out_name):
    tic = time.time()
    df.to_hdf(out_name, key='null', mode='w', complevel=9, complib='blosc')
    toc = time.time()
    print("saving" + out_name + " took {:.1f} seconds".format(toc - tic))

save_hd5(df, out_name)
