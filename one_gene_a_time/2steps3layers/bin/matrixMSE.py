#!/usr/bin/python
print('10/17/2017, reads h.hd5 and data.hd5, then calculate the matrix MSE')

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

# read data
file_input = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # data need imputation
file_groundTruth = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"

df_pred = scimpute.read_hd5('pre_train/imputation.step1.hd5')
df_input = scimpute.read_hd5(file_input)
df_input = scimpute.subset_df(df_input, df_pred)

df_groundTruth = scimpute.read_hd5(file_groundTruth)
df_groundTruth = scimpute.subset_df(df_groundTruth, df_pred)

print(df_pred.ix[0:4, 0:4])
print(df_input.ix[0:4, 0:4])
print(df_groundTruth.ix[0:4, 0:4])

# gene MSE
j = 0
input_j = df_input.ix[:, j:j+1].values
pred_j = df_pred.ix[:, j:j+1].values
groundTruth_j = df_groundTruth.ix[:, j:j+1].values

mse_j_input = ((pred_j - input_j) ** 2).mean()
mse_j_groundTruth = ((pred_j - groundTruth_j) ** 2).mean()

# matrix MSE
matrix_mse_input = ((df_pred.values - df_input.values) ** 2).mean()
matrix_mse_groundTruth = ((df_pred.values - df_groundTruth.values) ** 2).mean()