import pandas as pd
import os
import numpy as np
import time
import sys

def save_hd5 (df, out_name):
    tic = time.time()
    df.to_hdf(out_name, key='null', mode='w', complevel=9, complib='blosc')
    toc = time.time()
    print("saving" + out_name + " took {:.1f} seconds".format(toc-tic))

if len(sys.argv) is not 3:
    raise Exception ('usage: log_transformation <in.hd5> <out.hd5>')
else:
    print(sys.argv)
    print('this is log10(x+1) transformation')

file_in = sys.argv[1] #training set
file_out = sys.argv[2]
df = pd.read_hdf(file_in)
print(df.ix[1:5, 1:5])
df = np.log10(df+1)
print (df.ix[1:5, 1:5])
save_hd5(df, file_out)

