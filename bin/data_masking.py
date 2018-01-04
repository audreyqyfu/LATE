# split simulated dataset into two parts (A and B)
# mask 90% of the B part to mimic single cell data
print('needs polishing')

import pandas as pd
import os
import numpy as np
import time

def save_hd5 (df, out_name):
    tic = time.time()
    df.to_hdf(out_name, key='null', mode='w', complevel=9, complib='blosc')
    toc = time.time()
    print("saving" + out_name + " took {:.1f} seconds".format(toc-tic))

file_B = "EMT.MAGIC.9k.B.hd5"  # ground truth for scRNA-seq data
file_B_sub = 'EMT.MAGIC.9k.B.msk50.hd5'  # mimic scRNA_seq data with drop-outs

# read data #
df_B = pd.read_hdf(file_B)

# mask df_B
np.random.seed(2)
df_B_msked = df_B.where(np.random.uniform(size=df_B.shape)>0.5, 0)

# print
print("B.", df_B.ix[1:6, 1:6])
print("B.msk.shape", df_B_msked.ix[1:6, 1:6])

# save files
save_hd5(df_B_msked, file_B_sub)





