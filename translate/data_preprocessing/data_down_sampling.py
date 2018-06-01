#!/usr/bin/python
# first down-sample to typical single cell RNA-seq lib-size
# then add random zeros so that desired zero percentage reached
# do log10(x+1) transformation
import numpy as np
import pandas as pd
import sys
import scimpute

np.random.seed(1120)


# instructions
print('usage: data_data_down_sampling.py <file> <libsize-resampled> <zero-inflation-goal> <outname>',
      '<file>: count or tpm data matrix in hd5 format ([row = gene, column = cell]) (e.g. MAGIC.B.hd5)',
      '<lib-size-resampled>: the lib-size for each sample in the output (e.g. 60000)',
      '<zero-inflation goal>: a percentage (e.g. 10 represents 10%),',
      'if the goal is not met, random zeros will be added to meet the goal',
      'the output will be log10(x+1) transformed, and transposed [cells, genes]',
      'outname example: inname + ds_70k_10p.log.hd5',
      'the random seed has been set to 1120 in the beginning of this script',
      '', sep='\n')


# check num arguments
if len(sys.argv) is not 5:
    raise Exception("error: the num of arguments not correct")
else:
    print('running:')
    in_name = str(sys.argv[1])  # e.g. "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.hd5"
    libsize_resampled = int(sys.argv[2])  # e.g. 6e4
    percentage_goal = float(sys.argv[3])/100  # e.g.: 0.1
    out_name = str(sys.argv[4])

print("> command: ", sys.argv)


# read data
df = pd.read_hdf(in_name).transpose()  # so that output matrix is [sample, gene]
# summary
print('> input shape [samples, genes]:', df.shape)
nz_rate = scimpute.nnzero_rate_df(df)
print('nz_rate: {}\n'.format(nz_rate))
print(df.ix[0:3, 0:2])


# down-sampling with Multinomial distribution
df = scimpute.multinormial_downsampling(df, libsize_resampled)
# summary
nz_rate = scimpute.nnzero_rate_df(df)
print('> after down-sampling to {} reads/cell\nnz_rate is {}'.
      format(libsize_resampled, nz_rate))
print(df.ix[0:3, 0:2])


# further masking (zero_inflation)
if nz_rate > percentage_goal:
    df = scimpute.mask_df(df, percentage_goal)
else:
    df = df
# summary
nz_rate = scimpute.nnzero_rate_df(df)
print('> after masking(random zero injection)\nnz_rate is: {}'.format(nz_rate))
print(df.ix[0:3, 0:2])


# # lib-size normalization
# df = df.transpose()
# df = scimpute.df_normalization(df)
# df = df.transpose()
# print('> after RPM normalization: ', df.ix[0:3, 0:2])

# log-transformation
df = scimpute.df_log_transformation(df, 1)
print('> after log10(x+1) transformation: ', df.ix[0:3, 0:2])

# output result dataframe
scimpute.save_hd5(df, out_name)