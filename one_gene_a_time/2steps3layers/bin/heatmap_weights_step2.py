#!/usr/bin/python
# load weights and plot them
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

print ('python version:', sys.version)
print('tf.__version__', tf.__version__)
raise Exception("not finished")

# read data #
file = "../../data/v1-1-5-2/v1-1-5-2.F2.msk.hd5" #data need imputation
file_benchmark = "../../data/v1-1-5-2/v1-1-5-2.F2.hd5"
df = pd.read_hdf(file).transpose() #[cells,genes]
df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
m, n = df.shape  # m: n_cells; n: n_genes



# Parameters #
j = 3
learning_rate = 0.0001
training_epochs = 100
batch_size = 256
sd = 0.0001 #stddev for random init
n_input = n
n_hidden_1 = 500


# define
encoder_params = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=sd), name='encoder_w1'),
    'b1': tf.Variable(tf.random_normal([n_hidden_1], mean=30*sd, stddev=sd), name='encoder_b1')
}
decoder_params = {
    'w1': tf.Variable(tf.random_normal([n_hidden_1, n_input], stddev=sd), name='decoder_w1'),
    'b1': tf.Variable(tf.random_normal([n_input], mean=100 * sd, stddev=sd), name='decoder_b1')
    # 'b1': tf.Variable(tf.ones([n_input]), name='decoder_b1') #fast, but maybe linear model
}
focusFnn_params = {
    'w1': tf.Variable(tf.random_normal([n_hidden_1, 1], stddev=sd), name='focusFnn_w1'),
    'b1': tf.Variable(tf.random_normal([1], mean=30*sd, stddev=sd), name='focusFnn_b1')
}

# session
sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "./re_train_j999/step2.ckpt")

encoder_w1 = sess.run(encoder_params['w1'])
encoder_b1 = sess.run(encoder_params['b1'])
focusFnn_w1 = sess.run(focusFnn_params['w1'])
focusFnn_b1 = sess.run(focusFnn_params['b1'])

encoder_b1 = encoder_b1.reshape(len(encoder_b1), 1)
focusFnn_b1 = focusFnn_b1.reshape(len(focusFnn_b1), 1)

print(encoder_w1.shape)
print(encoder_b1.shape)
print(focusFnn_w1.shape)
print(focusFnn_b1.shape)

# heatmap
import scimpute

scimpute.heatmap_vis(encoder_w1, title='encoder_w1')
scimpute.heatmap_vis2(encoder_b1, title='encoder_b1')

scimpute.heatmap_vis(focusFnn_w1.T, title='focusFnn_w1.T')
scimpute.heatmap_vis2(focusFnn_b1, title='focusFnn_b1')