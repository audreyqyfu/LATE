#!/usr/bin/python
# first downsample to typical single cell RNA-seq lib-size
# then add random zeros so that desired zero percentage reached
# do log10(x+1) transformation
import numpy as np
import pandas as pd
import sys
import scimpute


# instructions
print('usage: down_sampling.py <file> <libsize-resampled> <zero-inflation-goal> <outname>',
      'file: count or tpm data matrix in hd5 format ([row = gene, column = cell]) (e.g. MAGIC.B.hd5)',
      'lib-size-resampled: the lib-size for each sample in the output (e.g. 60000)',
      'zero-inflation goal: a percentage (e.g. 10 represents 10%),',
      ' if the goal is not met, random zeros will be added to meet the goal',
      'the output will be log10(x+1) transformed',
      'outname example: inname + ds_70k_10p.log.hd5',
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
input_df = pd.read_hdf(in_name).transpose()  # so that output matrix is [sample, gene]
# summary
print('> input shape [samples, genes]:', input_df.shape)
nz_rate_in = scimpute.nnzero_rate_df(input_df)
print('nz_rate: {}\n'.format(nz_rate_in))
print(input_df.ix[0:3, 0:3])


# down-sampling with Multinomial distribution
resampled_df = scimpute.multinormial_downsampling(input_df, libsize_resampled)
del input_df
# summary
nz_rate_resampled = scimpute.nnzero_rate_df(resampled_df)
print('> after down-sampling to {} reads/cell\nnz_rate is {}'.
      format(libsize_resampled, nz_rate_resampled))
print(resampled_df.ix[0:3, 0:3])


# further masking (zero_inflation)
if nz_rate_resampled > percentage_goal:
    masked_df = scimpute.mask_df(resampled_df, percentage_goal)
else:
    masked_df = resampled_df
# summary
del resampled_df
nz_rate_masked = scimpute.nnzero_rate_df(masked_df)
print('after masking(random zero injection), nz_rate is: {}'.format(nz_rate_masked))
print(masked_df.ix[0:3, 0:3])

# log-transformation
log_transformed_df = scimpute.df_log_transformation(masked_df, 1)
del masked_df
print('after log_transformation: ', log_transformed_df.ix[0:3, 0:3])

# output result dataframe
scimpute.save_hd5(log_transformed_df, out_name)