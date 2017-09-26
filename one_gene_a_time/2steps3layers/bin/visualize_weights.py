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
data = 'EMT9k_log_msk90'  # EMT2730 or splatter

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

decoder_w2 = sess.run(decoder_params['w2'])
decoder_b2 = sess.run(decoder_params['b2'])
decoder_b2 = decoder_b2.reshape(len(decoder_b2), 1)
decoder_b2_T = decoder_b2.T

# visualization
scimpute.visualize_weights_biases(encoder_w1, encoder_b1_T, 'encoder_w1, b1')
scimpute.visualize_weights_biases(encoder_w2, encoder_b2_T, 'encoder_w2, b2')
scimpute.visualize_weights_biases(decoder_w1, decoder_b1_T, 'decoder_w1, b1')
scimpute.visualize_weights_biases(decoder_w2, decoder_b2_T, 'decoder_w2, b2')

# save weights for further work
scimpute.save_csv(fname='encoder_w1.csv', arr=encoder_w1)