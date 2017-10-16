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

print('python version:', sys.version)
print('tf.__version__', tf.__version__)

sys.path.append('./bin')
print('sys.path', sys.path)
import scimpute


def print_parameters():
    print(os.getcwd(), "\n",
          "\n# Parameters:",
          "\nn_features: ", n,
          "\nn_hidden1: ", n_hidden_1,
          "\nn_hidden2: ", n_hidden_2,
          "\nlearning_rate :", learning_rate,
          "\nbatch_size: ", batch_size,
          "\nepoches: ", training_epochs, "\n",
          "\npIn_holder: ", pIn,
          "\npHidden_holder: ", pHidden, "\n",
          "\ndf_train.values.shape", df_train.values.shape,
          "\ndf_valid.values.shape", df_valid.values.shape,
          "\ndf2_train.shape", df2_train.shape,
          "\ndf2_valid.values.shape", df2_valid.values.shape,
          "\n")
    print("input_array:\n", df.values[0:4, 0:4], "\n")


def evaluate_epoch0():
    print("\nEpoch 0:")
    epoch_log.append(epoch)
    mse_train = sess.run(mse_input, feed_dict={X: df_train.values, pIn_holder: 1, pHidden_holder: 1})
    mse_valid = sess.run(mse_input, feed_dict={X: df_valid.values, pIn_holder: 1, pHidden_holder: 1})
    mse_log_batch.append(mse_train)  # approximation
    mse_log_train.append(mse_train)
    mse_log_valid.append(mse_valid)
    print("mse_train=", round(mse_train, 3), "mse_valid=", round(mse_valid, 3))

    h_train = sess.run(h, feed_dict={X: df_train.values, pIn_holder: 1, pHidden_holder: 1})
    h_valid = sess.run(h, feed_dict={X: df_valid.values, pIn_holder: 1, pHidden_holder: 1})
    corr_train = scimpute.medium_corr(df_train.values, h_train)
    corr_valid = scimpute.medium_corr(df_valid.values, h_valid)
    cell_corr_log_batch.append(corr_train)
    cell_corr_log_train.append(corr_train)
    cell_corr_log_valid.append(corr_valid)
    print("Cell-pearsonr train, valid:", corr_train, corr_valid)


def snapshot():
    print("#Snapshot: ")
    # inference
    h_train = sess.run(h, feed_dict={X: df_train.values, pIn_holder: 1, pHidden_holder: 1})
    h_valid = sess.run(h, feed_dict={X: df_valid.values, pIn_holder: 1, pHidden_holder: 1})
    h_input = sess.run(h, feed_dict={X: df.values, pIn_holder: 1, pHidden_holder: 1})
    # print whole dataset pearsonr
    print("medium cell-pearsonr(all train): ",
          scimpute.medium_corr(df2_train.values, h_train, num=len(df_train)))
    print("medium cell-pearsonr(all valid): ",
          scimpute.medium_corr(df2_valid.values, h_valid, num=len(df_valid)))
    print("medium cell-pearsonr in all imputation cells: ",
          scimpute.medium_corr(df2.values, h_input, num=m))
    # save pred
    df_h_input = pd.DataFrame(data=h_input, columns=df.columns, index=df.index)
    scimpute.save_hd5(df_h_input, log_dir + "/imputation.step1.hd5")
    # save model
    save_path = saver.save(sess, log_dir + "/step1.ckpt")
    print("Model saved in: %s" % save_path)


def visualization_of_dfs():
    max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid, h, df.values])
    scimpute.heatmap_vis(df_valid.values, title='df.valid'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(h_valid, title='h.valid'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(df.values, title='df'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(h, title='h'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)


# read data #
df, df2, Aname, Bname, m, n = scimpute.read_data('EMT9k_log')
max = max(df.values.max(), df2.values.max())
df_train, df_valid, df_test = scimpute.split_df(df, a=0.9, b=0.1, c=0.0)
df2_train, df2_valid, df2_test = df2.ix[df_train.index], df2.ix[df_valid.index], df2.ix[df_test.index]

# Parameters #
n_input = n
n_hidden_1 = 20
n_hidden_2 = 400
n_hidden_3 = 200
pIn = 0.8
pHidden = 1  # todo: change back to 0.5 after test
learning_rate = 0.01  # todo: was 0.0003
sd = 0.0001  # stddev for random init
batch_size = 256
training_epochs = 10
display_step = 1
snapshot_step = 5
log_dir = './pre_train'
scimpute.refresh_logfolder(log_dir)
print_parameters()


# Define model #
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_input])  # input
M = tf.placeholder(tf.float32, [None, n_input])  # benchmark

pIn_holder = tf.placeholder(tf.float32)
pHidden_holder = tf.placeholder(tf.float32)

w_e1, b_e1 = scimpute.weight_bias_variable('encoder1', n, n_hidden_1, sd)
a_e1 = scimpute.dense_layer('encoder1', X, w_e1, b_e1, pIn_holder)

w_d1, b_d1 = scimpute.weight_bias_variable('decoder1', n_hidden_1, n, sd)
a_d1 = scimpute.dense_layer('decoder1', a_e1, w_d1, b_d1, pHidden_holder)

y_input = X
y_groundTruth = M
h = a_d1

with tf.name_scope("Metrics"):
    mse_input = tf.reduce_mean(tf.pow(y_input - h, 2))
    mse_groundTruth = tf.reduce_mean(tf.pow(y_groundTruth - h, 2))
    tf.summary.scalar('mse_input', mse_input)
    tf.summary.scalar('mse_groundTruth', mse_groundTruth)

trainer = tf.train.AdamOptimizer(learning_rate). \
    minimize(mse_input, var_list=[w_e1, b_e1,
                                  w_d1, b_d1])

# Launch Session #
sess = tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir + '/valid', sess.graph)
epoch = 0
total_batch = int(math.floor(len(df_train) // batch_size))  # floor
epoch_log = []
mse_log_batch, mse_log_valid, mse_log_train = [], [], []
cell_corr_log_batch, cell_corr_log_valid, cell_corr_log_train = [], [], []

evaluate_epoch0()

# training
for epoch in range(1, training_epochs+1):
    tic_cpu, tic_wall = time.clock(), time.time()

    random_indices = np.random.choice(len(df_train), batch_size, replace=False)
    for i in range(total_batch):
        indices = np.arange(batch_size * i, batch_size*(i+1))
        x_batch = df_train.values[indices, :]
        sess.run(trainer, feed_dict={X: x_batch, pIn_holder: pIn, pHidden_holder: pHidden})

    toc_cpu, toc_wall = time.clock(), time.time()

     # Log per epoch
    if (epoch == 1) or (epoch % display_step == 0):
        tic_log = time.time()
        print("\n#Epoch ", epoch, " took: ",
              round(toc_cpu - tic_cpu, 2), " CPU seconds; ",
              round(toc_wall - tic_wall, 2), "Wall seconds")
        # epoch_summary()
        mse_batch, h_batch = sess.run([mse_input, h], feed_dict={X: x_batch, pIn_holder:1.0, pHidden_holder:1.0})
        # mse_train, h_train = sess.run([mse_input, h], feed_dict={X: df_train, pIn_holder:1.0, pHidden_holder:1.0})
        mse_valid, h_valid = sess.run([mse_input, h], feed_dict={X: df_valid, pIn_holder:1.0, pHidden_holder:1.0})
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

    # Log per observation interval
    if (epoch == 1) or (epoch % snapshot_step == 0) or (epoch == training_epochs):
        tic_log2 = time.time()
        snapshot()
        toc_log2 = time.time()
        print('log2 time for observation intervals:', round(toc_log2 - tic_log2, 1))

train_writer.close()
valid_writer.close()


# learning curve
scimpute.learning_curve_mse(epoch_log, mse_log_batch, mse_log_valid)
scimpute.learning_curve_corr(epoch_log, cell_corr_log_batch, cell_corr_log_valid)

# gene-corr hist
hist = scimpute.gene_corr_hist(h_valid, df2_valid.values,
                                  fprefix='hist gene-corr, valid, step1',
                                  title="gene-corr (prediction vs ground-truth)"
                                  )
