#!/usr/bin/python
print('10/17/2017, reads h.hd5 and data.hd5, then analysis the result')

import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import math
import os
import time
import scimpute

# set filename
# file1 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # data need imputation
# file2 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # data need imputation
file1 = "../../../../data/gtex/gtex_v7.norm.log.hd5"  # input
file2 = "../../../../data/gtex/gtex_v7.norm.log.hd5"  # ground truth (same as input in step1)
file_pred = 'pre_train/imputation.step1.hd5'

# read data
h = scimpute.read_hd5(file_pred)
df1 = scimpute.read_hd5(file1).transpose()
# df1 = scimpute.subset_df(df1, h)
print('df1.shape', df1.shape)
df2 = scimpute.read_hd5(file2).transpose()
# df2 = scimpute.subset_df(df2, h)
print('df2.shape', df2.shape)
print('h:', h.ix[0:4, 0:4])
print('input:', df1.ix[0:4, 0:4])
print('groundTruth:', df2.ix[0:4, 0:4])

# read index
train_idx = scimpute.read_csv('pre_train/df_train.index.csv').index
valid_idx = scimpute.read_csv('pre_train/df_valid.index.csv').index
# split
h_train = h.ix[train_idx]
h_valid = h.ix[valid_idx]
df_train = df1.ix[train_idx]
df_valid = df1.ix[valid_idx]
df2_train = df2.ix[train_idx]
df2_valid = df2.ix[valid_idx]

# truth vs pred
def groundTruth_vs_prediction():
    print("> Ground truth vs prediction")
    for j in [1, 2, 3, 4, 205, 206, 4058, 7496, 8495, 12871]:  # Cd34, Gypa, Klf1, Sfpi1
            scimpute.scatterplot2(df2_valid.values[:, j], h_valid.values[:, j], range='same',
                                  title=str('scatterplot1, gene-' + str(j) + ', valid, step1'),
                                  xlabel='Ground Truth ',
                                  ylabel='Prediction '
                                  )
            scimpute.scatterplot2(df2_valid.values[:, j], h_valid.values[:, j], range='flexible',
                                      title=str('scatterplot2, gene-' + str(j) + ', valid, step1'),
                                      xlabel='Ground Truth ',
                                      ylabel='Prediction '
                                      )

groundTruth_vs_prediction()

# gene MSE
j = 0
input_j = df1.ix[:, j:j+1].values
pred_j = h.ix[:, j:j+1].values
groundTruth_j = df2.ix[:, j:j+1].values

mse_j_input = ((pred_j - input_j) ** 2).mean()
mse_j_groundTruth = ((pred_j - groundTruth_j) ** 2).mean()

# matrix MSE
matrix_mse_input = ((h.values - df1.values) ** 2).mean()
matrix_mse_groundTruth = ((h.values - df2.values) ** 2).mean()