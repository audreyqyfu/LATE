#!/usr/bin/python
# first downsample to typical single cell RNA-seq lib-size
# then add random zeros so that desired zero percentage reached
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
      'outname example: inname + ds_60000_10.hd5',
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


# read data
input_df = pd.read_hdf(in_name).transpose()  # so that output matrix is [sample, gene]
# summary
print('input shape [samples, genes]:', input_df.shape)
nz_rate_in = scimpute.nnzero_rate_df(input_df)
print('nz_rate_in: {}'.format(nz_rate_in))


# down-sampling with Multinomial distribution
resampled_df = scimpute.multinormial_downsampling(input_df, libsize_resampled)
del input_df
# summary
nz_rate_resampled = scimpute.nnzero_rate_df(resampled_df)
print('nz_rate after downsampling to {} libsize: {}'.
      format(libsize_resampled, nz_rate_resampled))


# further masking (zero_inflation)
if nz_rate_resampled > percentage_goal:
    masked_df = scimpute.mask_df(resampled_df, percentage_goal)
else:
    masked_df = resampled_df
# summary
del resampled_df
nz_rate_masked = scimpute.nnzero_rate_df(masked_df)
print('nz_rate after further zero_inflation(masking) is: {}'.format(nz_rate_masked))


# output result dataframe
scimpute.save_hd5(masked_df, out_name)