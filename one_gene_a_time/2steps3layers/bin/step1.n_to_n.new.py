#!/usr/bin/python
from __future__ import division  # fix division // get float bug
from __future__ import print_function  # fix printing \n

import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import math
import os
import time
import seaborn as sns

print('python version:', sys.version)
print('tf.__version__', tf.__version__)

sys.path.append('./bin')
print('sys.path', sys.path)
import scimpute




# refresh pre_train folder
log_dir = './pre_train'
scimpute.refresh_logfolder(log_dir)

# read data and save indexes
df, df2, Aname, Bname, m, n = scimpute.read_data('EMT9k_log')
max = max(df.values.max(), df2.values.max())
df_train, df_valid, df_test = scimpute.split_df(df, a=0.7, b=0.15, c=0.15)
df2_train, df2_valid, df2_test = df2.ix[df_train.index], df2.ix[df_valid.index], df2.ix[df_test.index]
df_train.to_csv('pre_train/df_train.index.csv', columns=[], header=False)  # save index for future use
df_valid.to_csv('pre_train/df_valid.index.csv', columns=[], header=False)
df_test.to_csv('pre_train/df_test.index.csv', columns=[], header=False)

# Parameters #
# todo: update for different depth
n_input = n
n_hidden_1 = 800
n_hidden_2 = 600
n_hidden_3 = 400
n_hidden_4 = 200

# todo: adjust to optimized hyper-parameters when different layers used
pIn = 0.8
pHidden = 0.5
learning_rate = 0.00003  # 0.0003 for 3-7L, 0.00003 for 9L
sd = 0.00001  # 3-7L:1e-3, 9L:1e-4
batch_size = 256
training_epochs = 3  #3L:100, 5L:1000, 7L:1000, 9L:3000
display_step = 20
snapshot_step = 1000
print_parameters()  # todo: use logger

# Define model #
tf.reset_default_graph()

# placeholders
X = tf.placeholder(tf.float32, [None, n_input], name='X_input')  # input
M = tf.placeholder(tf.float32, [None, n_input], name='M_ground_truth')  # benchmark

pIn_holder = tf.placeholder(tf.float32, name='pIn')
pHidden_holder = tf.placeholder(tf.float32, name='pHidden')

# init variables and build graph
tf.set_random_seed(3)  # seed
# todo: adjust based on depth
with tf.name_scope('Encoder_L1'):
    e_w1, e_b1 = scimpute.weight_bias_variable('encoder1', n, n_hidden_1, sd)
    e_a1 = scimpute.dense_layer('encoder1', X, e_w1, e_b1, pIn_holder)

with tf.name_scope('Encoder_L2'):
    e_w2, e_b2 = scimpute.weight_bias_variable('encoder2', n_hidden_1, n_hidden_2, sd)
    e_a2 = scimpute.dense_layer('encoder2', e_a1, e_w2, e_b2, pHidden_holder)

with tf.name_scope('Encoder_L3'):
    e_w3, e_b3 = scimpute.weight_bias_variable('encoder3', n_hidden_2, n_hidden_3, sd)
    e_a3 = scimpute.dense_layer('encoder3', e_a2, e_w3, e_b3, pHidden_holder)

with tf.name_scope('Encoder_L4'):
    e_w4, e_b4 = scimpute.weight_bias_variable('encoder4', n_hidden_3, n_hidden_4, sd)
    e_a4 = scimpute.dense_layer('encoder4', e_a3, e_w4, e_b4, pHidden_holder)

with tf.name_scope('Decoder_L4'):
    d_w4, d_b4 = scimpute.weight_bias_variable('decoder4', n_hidden_4, n_hidden_3, sd)
    d_a4 = scimpute.dense_layer('decoder4', e_a4, d_w4, d_b4, pHidden_holder)

with tf.name_scope('Decoder_L3'):
    d_w3, d_b3 = scimpute.weight_bias_variable('decoder3', n_hidden_3, n_hidden_2, sd)
    d_a3 = scimpute.dense_layer('decoder3', d_a4, d_w3, d_b3, pHidden_holder)

with tf.name_scope('Decoder_L2'):
    d_w2, d_b2 = scimpute.weight_bias_variable('decoder2', n_hidden_2, n_hidden_1, sd)
    d_a2 = scimpute.dense_layer('decoder2', d_a3, d_w2, d_b2, pHidden_holder)

with tf.name_scope('Decoder_L1'):
    d_w1, d_b1 = scimpute.weight_bias_variable('decoder1', n_hidden_1, n, sd)
    d_a1 = scimpute.dense_layer('decoder1', d_a2, d_w1, d_b1, pHidden_holder)  # todo: change input activations if model changed

# define input/output
y_input = X
y_groundTruth = M
h = d_a1

# define loss
with tf.name_scope("Metrics"):
    mse_input = tf.reduce_mean(tf.pow(y_input - h, 2))
    mse_groundTruth = tf.reduce_mean(tf.pow(y_groundTruth - h, 2))
    tf.summary.scalar('mse_input', mse_input)
    tf.summary.scalar('mse_groundTruth', mse_groundTruth)

trainer = tf.train.AdamOptimizer(learning_rate).minimize(mse_input)

# Launch Session #
sess = tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)
batch_writer = tf.summary.FileWriter(log_dir + '/batch', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir + '/valid', sess.graph)
epoch = 0
num_batch = int(math.floor(len(df_train) // batch_size))  # floor
epoch_log = []
mse_log_batch, mse_log_valid, mse_log_train = [], [], []
cell_corr_log_batch, cell_corr_log_valid, cell_corr_log_train = [], [], []

evaluate_epoch0()

# training
for epoch in range(1, training_epochs+1):
    # training model
    tic_cpu, tic_wall = time.clock(), time.time()
    ridx_full = np.random.choice(len(df_train), len(df_train), replace=False)
    for i in range(num_batch):
        indices = np.arange(batch_size * i, batch_size*(i+1))
        ridx_batch = ridx_full[indices]
        x_batch = df_train.values[ridx_batch, :]
        sess.run(trainer, feed_dict={X: x_batch, pIn_holder: pIn, pHidden_holder: pHidden})

    toc_cpu, toc_wall = time.clock(), time.time()

     # Log per epoch
    if (epoch == 1) or (epoch % display_step == 0):
        # print training time
        print("\n#Epoch ", epoch, " took: ",
              round(toc_cpu - tic_cpu, 2), " CPU seconds; ",
              round(toc_wall - tic_wall, 2), "Wall seconds")

        tic_log = time.time()

        # Ad hoc summaries
        mse_batch, h_batch = sess.run([mse_input, h], feed_dict={X: x_batch, pIn_holder: 1.0, pHidden_holder: 1.0})
        # mse_train, h_train = sess.run([mse_input, h], feed_dict={X: df_train, pIn_holder:1.0, pHidden_holder:1.0})
        mse_valid, h_valid = sess.run([mse_input, h], feed_dict={X: df_valid, pIn_holder: 1.0, pHidden_holder: 1.0})
        mse_log_batch.append(mse_batch)
        # mse_log_train.append(mse_train)
        mse_log_valid.append(mse_valid)
        print('mse_batch, valid:', mse_batch, mse_valid)
        # print('mse_batch, train, valid:', mse_batch, mse_train, mse_valid)

        corr_batch = scimpute.medium_corr(x_batch, h_batch)
        # corr_train = scimpute.medium_corr(df_train.values, h_train)
        corr_valid = scimpute.medium_corr(df_valid.values, h_valid)
        cell_corr_log_batch.append(corr_batch)
        # cell_corr_log_train.append(corr_train)
        cell_corr_log_valid.append(corr_valid)
        print("cell-pearsonr in batch, valid:", corr_batch, corr_valid)
        # print("cell-pearsonr in batch, train, valid:", corr_batch, corr_train, corr_valid)

        toc_log = time.time()
        epoch_log.append(epoch)
        print('log time for each epoch:', round(toc_log - tic_log, 1))

        # tb summary
        tb_summary()

        # temp: show weights in layer1, and see if it updates in deep network
        print('encoder_w1: ', sess.run(e_w1)[0, 0:2])

    # Log per observation interval
    if (epoch % snapshot_step == 0) or (epoch == training_epochs):
        tic_log2 = time.time()
        h_train, h_valid, h_input = snapshot()
        learning_curve()
        hist = scimpute.gene_corr_hist(h_valid, df2_valid.values,
                                       title="gene-corr (prediction vs ground-truth)"
                                       )
        hist = scimpute.cell_corr_hist(h_valid, df2_valid.values,
                                       title="cell-corr (prediction vs ground-truth)"
                                       )
        visualization_of_dfs()
        gene_gene_relationship()
        groundTruth_vs_prediction()
        save_bottle_neck_representation()
        visualize_weights()
        save_weights()
        toc_log2 = time.time()
        print('log2 time for observation intervals:', round(toc_log2 - tic_log2, 1))

batch_writer.close()
valid_writer.close()
sess.close()
print("Finished!")
