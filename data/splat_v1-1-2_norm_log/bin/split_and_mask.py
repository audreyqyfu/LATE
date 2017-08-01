# split simulated dataset into two parts (A and B)
# mask 90% of the B part to mimic single cell data

import pandas as pd
import os
import numpy as np
import time

def save_hd5 (df, out_name):
    tic = time.time()
    df.to_hdf(out_name, key='null', mode='w', complevel=9, complib='blosc')
    toc = time.time()
    print("saving" + out_name + " took {:.1f} seconds".format(toc-tic))

file_in = 'splat.OneGroup.norm.log.hd5'
file_A = "splat.OneGroup.norm.log.A.hd5" #training set
file_B = "splat.OneGroup.norm.log.B.hd5" #ground truth for scRNA-seq data
file_B_sub = 'splat.OneGroup.norm.log.B.mask90.hd5' #mimic scRNA_seq data with drop-outs

# read data #
# df=pd.read_csv(file, sep=',', index_col=0)
df = pd.read_hdf(file_in).transpose() #[cells,genes]

# split randomly
np.random.seed(1) # for splitting consistency
inx_A = np.random.rand(len(df)) < 0.5
df_A = df[inx_A].transpose()
df_B = df[~inx_A].transpose()

# mask df_B
np.random.seed(2)
df_B_msked = df_B.where(np.random.uniform(size=df_B.shape)>0.9, 0)

# print
print("input.shape: ", df.transpose().shape) 
print("A.shape: ", df_A.shape)
print("B.shape: ", df_B.shape)
print("B_sub.shape", df_B_msked.shape)

# save files
save_hd5(df_A, file_A)
save_hd5(df_B, file_B)
save_hd5(df_B_msked, file_B_sub)





