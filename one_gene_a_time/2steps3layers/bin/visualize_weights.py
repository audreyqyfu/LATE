#!/usr/bin/python
# load pre-trained NN, transfer learning
# https://www.tensorflow.org/get_started/summaries_and_tensorboard
# 07/25/2017
from __future__ import division #fix division // get float bug
from __future__ import print_function #fix printing \n

import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import math
import os
import time
import scimpute
print ('python version:', sys.version)
print('tf.__version__', tf.__version__)

# Input to get n
data = 'EMT9k_log_A'  # EMT2730 or splatter

if data is 'splatter':  # only this mode creates gene-gene plot
    file = "../data/v1-1-5-3/v1-1-5-3.F3.msk.hd5" #data need imputation
    file_benchmark = "../data/v1-1-5-3/v1-1-5-3.F3.hd5"
    Aname = '(F3.msk)'
    Bname = '(F3)'
    df = pd.read_hdf(file).transpose() #[cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT2730':
    file = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5" #data need imputation
    file_benchmark = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5"
    Aname = '(EMT2730)'
    Bname = '(EMT2730)'
    df = pd.read_hdf(file).transpose() #[cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.hd5"
    Aname = '(EMT9k.B.msk)'
    Bname = '(EMT9k.B)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k_log_msk50':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk50.log.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"
    Aname = '(EMT9kLog_Bmsk50)'
    Bname = '(EMT9kLog_B)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k_log_msk90':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk90.log.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"
    Aname = '(EMT9kLog_Bmsk90)'
    Bname = '(EMT9kLog_B)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k_log_A':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # data need imputation
    Aname = '(EMT9kLog_A)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    m, n = df.shape  # m: n_cells; n: n_genes
else:
    raise Warning("data name not recognized!")

# parameters
sd = 0.0001 #stddev for random init
n_input = n
n_hidden_1 = 400
n_hidden_2 = 200


# Define model #
X = tf.placeholder(tf.float32, [None, n_input])  # input
M = tf.placeholder(tf.float32, [None, n_input])  # benchmark
encoder_params = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=sd), name='encoder_w1'),
    'b1': tf.Variable(tf.random_normal([n_hidden_1], mean=100 * sd, stddev=sd), name='encoder_b1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=sd), name='encoder_w2'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], mean=100 * sd, stddev=sd), name='encoder_b2')
}
decoder_params = {
    'w1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1], stddev=sd), name='decoder_w1'),
    'b1': tf.Variable(tf.random_normal([n_hidden_1], mean=100 * sd, stddev=sd), name='decoder_b1'),
    # 'b1': tf.Variable(tf.ones([n_input]), name='decoder_b1') #fast, but maybe linear model
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_input], stddev=sd), name='decoder_w2'),
    'b2': tf.Variable(tf.random_normal([n_input], mean=100 * sd, stddev=sd), name='decoder_b2')
}
parameters = {**encoder_params, **decoder_params}


# Launch Session #
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "./pre_train/step1.ckpt")


# get weight arrays
encoder_w1 = sess.run(encoder_params['w1'])  #1000, 500
encoder_b1 = sess.run(encoder_params['b1'])  #500, (do T)
encoder_b1 = encoder_b1.reshape(len(encoder_b1), 1)
encoder_b1_T = encoder_b1.T

encoder_w2 = sess.run(encoder_params['w2'])
encoder_b2 = sess.run(encoder_params['b2'])
encoder_b2 = encoder_b2.reshape(len(encoder_b2), 1)
encoder_b2_T = encoder_b2.T

decoder_w1 = sess.run(decoder_params['w1'])
decoder_b1 = sess.run(decoder_params['b1'])
decoder_b1 = decoder_b1.reshape(len(decoder_b1), 1)
decoder_b1_T = decoder_b1.T

decoder_w2 = sess.run(decoder_params['w2'])  # (400, 16114)
decoder_b2 = sess.run(decoder_params['b2'])
decoder_b2 = decoder_b2.reshape(len(decoder_b2), 1)
decoder_b2_T = decoder_b2.T


# small abs_sum
decoder_w2_abs = np.absolute(decoder_w2)  # (400, 16114)
decoder_w2_abs_colsum = np.sum(decoder_w2_abs, axis=0)  # (16114,)
# hist of decoder_w2_abs_colsum
plt.hist(decoder_w2_abs_colsum, bins=100)
plt.title('decoder_w2_abs_colsum')
plt.savefig('decoder_w2_abs_colsum.hist.png', bbox_inches='tight')
plt.close()
# filter
small_absSum_idx = decoder_w2_abs_colsum < 1  # seem from hist
print("small_absSum_idx:", sum(small_absSum_idx))
df_small_absSum = df.ix[:, small_absSum_idx]





# small var
decoder_w2_colVar = np.var(decoder_w2, axis=0)  # (16114,)
# hist
plt.hist(decoder_w2_colVar, bins=100)
plt.title('decoder_w2_colVar')
plt.savefig('decoder_w2_colVar.hist.png', bbox_inches='tight')
plt.close()
# filter
small_var_idx = decoder_w2_colVar < 0.00001  # to get 25% genes
print('small_var_idx', sum(small_var_idx))
df_small_var = df.ix[:, small_var_idx]

# visualization
df.ix[0:4, 0:4]
df_small_absSum.ix[0:4, 0:4]
df_small_var.ix[0:4, 0:4]

# show mean expression, hopefully find low expression
def mean_df(df):
    Sum = sum(sum(df.values))
    Mean = Sum/df.size
    return(Mean)

print('mean expression of original df: ', Aname, mean_df(df))
print('mean expression of data with small weights: ',  mean_df(df_small_absSum))
print('mean expression of data with small variation in weights: ',  mean_df(df_small_var))

# show nnzero percentage
def nnzero_rate_df(df):
    idx = df != 0
    nnzero_rate = round(sum(sum(idx.values))/df.size,3)
    return(nnzero_rate)

nnzero_rate_df(df)
print('non-zero rate of original df: ', Aname, nnzero_rate_df(df))
print('non-zero rate of data with small weights: ',  nnzero_rate_df(df_small_absSum))
print('non-zero rate of data with small variation in weights: ',  nnzero_rate_df(df_small_var))

# histogram
def hist_df(df, title="hist of df"):
    df_flat = df.values.reshape(df.size, 1)
    plt.hist(df_flat, bins=200)
    plt.title(title)
    plt.savefig(title+'.png', bbox_inches='tight')
    plt.close()
    print('hist of ', title, 'is done')

hist_df(df, title='hist_of_df')
hist_df(df_small_absSum, title='hist_of_df_small_absSum')
hist_df(df_small_var, title='hist_of_df_small_var')


# heatmap(fail)
def visualization_of_dfs():
    # max, min = scimpute.max_min_element_in_arrs([df.values, df_small_absSum.values, df_small_var.values])
    max, min = scimpute.max_min_element_in_arrs([df_small_absSum.values, df_small_var.values])
    # scimpute.heatmap_vis(df.values, title='df', xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(df_small_absSum.values, title=Aname+'.df.subset: decoder_w2, small absSum', xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(df_small_var, title=Aname+'.df.subset: decoder_w2, small var', xlab='genes', ylab='cells', vmax=max, vmin=min)

visualization_of_dfs()

# # visualization
# scimpute.visualize_weights_biases(encoder_w1, encoder_b1_T, 'encoder_w1, b1')
# scimpute.visualize_weights_biases(encoder_w2, encoder_b2_T, 'encoder_w2, b2')
# scimpute.visualize_weights_biases(decoder_w1, decoder_b1_T, 'decoder_w1, b1')
# scimpute.visualize_weights_biases(decoder_w2, decoder_b2_T, 'decoder_w2, b2')
#
# # save weights for further work
# scimpute.save_csv(encoder_w1, 'encoder_w1.csv')
# scimpute.save_csv(encoder_w2, 'encoder_w2.csv')
# scimpute.save_csv(decoder_w1, 'decoder_w1.csv')
# scimpute.save_csv(decoder_w2, 'decoder_w2.csv')
