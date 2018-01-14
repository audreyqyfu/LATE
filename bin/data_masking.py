import pandas as pd
import os
import numpy as np
import time
import sys
import scimpute


print('usage: python data_masking.py df.hd5  zero_percentage_goal(90, 95) outname')
print('example: python data_masking.py test.hd5 90 test.msk90.hd5')
print('matrix direction does not matter, only values masked, shape kept same')

if len(sys.argv) is not 4:
    raise Exception("error: the num of arguments not correct")
else:
    print('running:')
    fname = str(sys.argv[1])  # df.hd5
    zero_goal = float(sys.argv[2])/100  # small.hd5
    nz_goal = 1 - zero_goal
    out_name = str(sys.argv[3])  # big.small.hd5
    print('cmd: ', sys.argv)

# read data #
df = pd.read_hdf(fname)
print('input shape:', df.shape)
nz_rate = scimpute.nnzero_rate_df(df)
print('nz_rate: {}\n'.format(nz_rate))
print(df.iloc[0:6, 0:2])

# Masking
np.random.seed(2)
# further masking (zero_inflation)
if nz_rate > nz_goal:
    df = scimpute.mask_df(df, nz_goal)
else:
    df = df

# print
print('msked shape:', df.shape)
nz_rate = scimpute.nnzero_rate_df(df)
print('nz_rate_msked: {}\n'.format(nz_rate))
print(df.iloc[0:6, 0:2])


# save files
scimpute.save_hd5(df, out_name)





