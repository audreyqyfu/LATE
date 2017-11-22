#!/usr/bin/python
# <step1>: pre-training w/b on reference RNA-seq data

import tensorflow as tf
import numpy as np
import math
import seaborn as sns
import pandas as pd
from scipy.stats.stats import pearsonr
import sys
import os
import time
import matplotlib; matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt
import scimpute

sys.path.append('./bin')
print('Sys.path:\n', sys.path, '\n')
print('python version:', sys.version)
print('tf.__version__', tf.__version__, '\n')

import step1_params as p  #import parameters


# Define functions #
def evaluate_epoch0():
    print("> Evaluate epoch 0:")
    epoch_log.append(epoch)
    mse_train = sess.run(mse1,
                         feed_dict={
                             X: df1_train.values,
                             pIn_holder: 1,
                             pHidden_holder: 1}
                         )
    mse_valid = sess.run(mse1,
                         feed_dict={
                             X: df1_valid.values,
                             pIn_holder: 1,
                             pHidden_holder: 1}
                         )
    mse_log_batch.append(mse_train)  # approximation
    mse_log_train.append(mse_train)
    mse_log_valid.append(mse_valid)
    print("mse_train=", round(mse_train, 3), "mse_valid=", round(mse_valid, 3))

    h_train = sess.run(h,
                       feed_dict={
                           X: df1_train.values,
                           pIn_holder: 1,
                           pHidden_holder: 1}
                       )
    h_valid = sess.run(h,
                       feed_dict={
                           X: df1_valid.values,
                           pIn_holder: 1,
                           pHidden_holder: 1}
                       )
    corr_train = scimpute.median_corr(df1_train.values, h_train)
    corr_valid = scimpute.median_corr(df1_valid.values, h_valid)
    cell_corr_log_batch.append(corr_train)
    cell_corr_log_train.append(corr_train)
    cell_corr_log_valid.append(corr_valid)
    print("Cell-pearsonr train, valid:", corr_train, corr_valid)
    # tb
    merged_summary = tf.summary.merge_all()
    summary_batch = sess.run(merged_summary, feed_dict={X: df1_train,
                                                        pIn_holder: 1.0,
                                                        pHidden_holder: 1.0})
    summary_valid = sess.run(merged_summary, feed_dict={X: df1_valid.values,
                                                        pIn_holder: 1.0,
                                                        pHidden_holder: 1.0})
    batch_writer.add_summary(summary_batch, epoch)
    valid_writer.add_summary(summary_valid, epoch)


def tb_summary():
    tic = time.time()
    # run_metadata = tf.RunMetadata()
    # batch_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)
    merged_summary = tf.summary.merge_all()
    summary_batch = sess.run(merged_summary, feed_dict={X: x_batch,
                                                        pIn_holder: 1.0,
                                                        pHidden_holder: 1.0})
    summary_valid = sess.run(merged_summary, feed_dict={X: df1_valid.values,
                                                        pIn_holder: 1.0,
                                                        pHidden_holder: 1.0})
    batch_writer.add_summary(summary_batch, epoch)
    valid_writer.add_summary(summary_valid, epoch)
    toc = time.time()
    print('tb_summary time:', round(toc-tic, 2))


def learning_curve():
    print('> plotting learning curves')
    scimpute.learning_curve_mse(epoch_log, mse_log_batch, mse_log_valid)
    scimpute.learning_curve_corr(epoch_log, cell_corr_log_batch, cell_corr_log_valid)


def snapshot():
    print("> Snapshot (save inference, save session, calculate whole dataset cell-pearsonr ): ")
    # inference
    h_train = sess.run(h, feed_dict={X: df1_train.values, pIn_holder: 1, pHidden_holder: 1})
    h_valid = sess.run(h, feed_dict={X: df1_valid.values, pIn_holder: 1, pHidden_holder: 1})
    h_input = sess.run(h, feed_dict={X: df1.values, pIn_holder: 1, pHidden_holder: 1})
    # print whole dataset pearsonr
    print("median cell-pearsonr(all train): ",
          scimpute.median_corr(df1_train.values, h_train, num=len(df1_train)))
    print("median cell-pearsonr(all valid): ",
          scimpute.median_corr(df1_valid.values, h_valid, num=len(df1_valid)))
    print("median cell-pearsonr in all imputation cells: ",
          scimpute.median_corr(df1.values, h_input, num=m))
    # save pred
    df1_h_input = pd.DataFrame(data=h_input, columns=df1.columns, index=df1.index)
    scimpute.save_hd5(df1_h_input, log_dir + "/imputation.step1.hd5")
    # save model
    save_path = saver.save(sess, log_dir + "/step1.ckpt")
    print("Model saved in: %s" % save_path)
    return (h_train, h_valid, h_input)


def save_bottle_neck_representation():
    print("> save bottle-neck_representation")
    # todo: change variable name for each model
    code_bottle_neck_input = sess.run(a_bottle_neck, feed_dict={X: df1.values, pIn_holder: 1, pHidden_holder: 1})
    np.save('pre_train/code_neck_valid.npy', code_bottle_neck_input)
    # clustermap = sns.clustermap(code_bottle_neck_input)
    # clustermap.savefig('./plots/bottle_neck.hclust.png')


def visualize_weight(w_name, b_name):
    w = eval(w_name)
    b = eval(b_name)
    w_arr = sess.run(w)
    b_arr = sess.run(b)
    b_arr = b_arr.reshape(len(b_arr), 1)
    b_arr_T = b_arr.T
    scimpute.visualize_weights_biases(w_arr, b_arr_T, w_name + ',' + b_name)  # todo: update name (low priority)


def visualize_weights():
    # todo: update when model changes depth
    for l1 in range(1, p.l+1):
        encoder_weight = 'e_w'+str(l1)
        encoder_bias = 'e_b'+str(l1)
        visualize_weight(encoder_weight, encoder_bias)
        decoder_bias = 'd_b'+str(l1)
        decoder_weight = 'd_w'+str(l1)
        visualize_weight(decoder_weight, decoder_bias)


def save_weights():
    # todo: update when model changes depth
    print('save weights in npy')
    for l1 in range(1, p.l+1):
        encoder_weight_name = 'e_w'+str(l1)
        encoder_bias_name = 'e_b'+str(l1)
        decoder_bias_name = 'd_b'+str(l1)
        decoder_weight_name = 'd_w'+str(l1)
        np.save('pre_train/'+encoder_weight_name, sess.run(eval(encoder_weight_name)))
        np.save('pre_train/'+decoder_weight_name, sess.run(eval(decoder_weight_name)))
        np.save('pre_train/'+encoder_bias_name, sess.run(eval(encoder_bias_name)))
        np.save('pre_train/'+decoder_bias_name, sess.run(eval(decoder_bias_name)))


# Start
print('> cmd: ', sys.argv)

# refresh pre_train folder
log_dir = './pre_train'
scimpute.refresh_logfolder(log_dir)

# read data
if p.file_orientation == 'gene_row':
    df1 = pd.read_hdf(p.file1).transpose()
elif p.file_orientation == 'cell_row':
    df1 = pd.read_hdf(p.file1)  # [cell, gene] in our model
else:
    raise Exception('parameter err: file_orientation not correctly spelled')

# Test or not
if p.test_flag > 0:
    print('* in test mode')
    df1 = df1.ix[0:p.m, 0:p.n]

# Summary of data
print("input_df:\n", df1.ix[0:3, 0:2], "\n")
m, n = df1.shape  # m: n_cells; n: n_genes
print('{} genes, {} samples\n'.format(n, m))

# split data, save index
df1_train, df1_valid, df1_test = scimpute.split_df(df1,
                                                   a=p.a, b=p.b, c=p.c,
                                                   seed_var=1)
df1_train.to_csv('pre_train/df1_train.index.csv',
                 columns=[], header=False)
df1_valid.to_csv('pre_train/df1_valid.index.csv',
                 columns=[], header=False)
df1_test.to_csv('pre_train/df1_test.index.csv',
                columns=[], header=False)

# Define model #
tf.reset_default_graph()

# placeholders
X = tf.placeholder(tf.float32, [None, n], name='X_input')  # input
pIn_holder = tf.placeholder(tf.float32, name='pIn')
pHidden_holder = tf.placeholder(tf.float32, name='pHidden')

# init variables and build graph
tf.set_random_seed(p.seed_tf)  # seed
# update for different depth
with tf.name_scope('Encoder_L1'):
    e_w1, e_b1 = scimpute.weight_bias_variable(
        'encoder1', n, p.n_hidden_1, p.sd)
    e_a1 = scimpute.dense_layer(
        'encoder1', X, e_w1, e_b1, pIn_holder)

with tf.name_scope('Encoder_L2'):
    e_w2, e_b2 = scimpute.weight_bias_variable(
        'encoder2', p.n_hidden_1, p.n_hidden_2, p.sd)
    e_a2 = scimpute.dense_layer(
        'encoder2', e_a1, e_w2, e_b2, pHidden_holder)

with tf.name_scope('Encoder_L3'):
    e_w3, e_b3 = scimpute.weight_bias_variable(
        'encoder3', p.n_hidden_2, p.n_hidden_3, p.sd)
    e_a3 = scimpute.dense_layer(
        'encoder3', e_a2, e_w3, e_b3, pHidden_holder)

# with tf.name_scope('Encoder_L4'):
#     e_w4, e_b4 = scimpute.weight_bias_variable('encoder4', p.n_hidden_3, p.n_hidden_4, p.sd)
#     e_a4 = scimpute.dense_layer('encoder4', e_a3, e_w4, e_b4, pHidden_holder)
#
# with tf.name_scope('Decoder_L4'):
#     d_w4, d_b4 = scimpute.weight_bias_variable('decoder4', p.n_hidden_4, p.n_hidden_3, p.sd)
#     d_a4 = scimpute.dense_layer('decoder4', e_a4, d_w4, d_b4, pHidden_holder)

with tf.name_scope('Decoder_L3'):
    d_w3, d_b3 = scimpute.weight_bias_variable(
        'decoder3', p.n_hidden_3, p.n_hidden_2, p.sd)
    d_a3 = scimpute.dense_layer(
        'decoder3', e_a3, d_w3, d_b3, pHidden_holder)

with tf.name_scope('Decoder_L2'):
    d_w2, d_b2 = scimpute.weight_bias_variable(
        'decoder2', p.n_hidden_2, p.n_hidden_1, p.sd)
    d_a2 = scimpute.dense_layer(
        'decoder2', d_a3, d_w2, d_b2, pHidden_holder)

with tf.name_scope('Decoder_L1'):
    d_w1, d_b1 = scimpute.weight_bias_variable(
        'decoder1', p.n_hidden_1, n, p.sd)
    d_a1 = scimpute.dense_layer(
        'decoder1', d_a2, d_w1, d_b1, pHidden_holder)


# define input/output
a_bottle_neck = e_a3
h = d_a1

# define loss
with tf.name_scope("Metrics"):
    mse1 = tf.reduce_mean(tf.pow(X - h, 2))
    tf.summary.scalar('mse1 (H vs X)', mse1)

# trainer
optimizer = tf.train.AdamOptimizer(p.learning_rate)
trainer = optimizer.minimize(mse1)

# Launch Session #
sess = tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

batch_writer = tf.summary.FileWriter(log_dir + '/batch', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir + '/valid', sess.graph)

epoch = 0
num_batch = int(math.floor(len(df1_train) // p.batch_size))  # floor
epoch_log = []
mse_log_batch, mse_log_valid, mse_log_train = [], [], []
cell_corr_log_batch, cell_corr_log_valid, cell_corr_log_train = [], [], []

evaluate_epoch0()

# training
for epoch in range(1, p.max_training_epochs+1):
    # training model #
    tic_cpu, tic_wall = time.clock(), time.time()
    ridx_full = np.random.choice(len(df1_train), len(df1_train), replace=False)
    for i in range(num_batch):
        indices = np.arange(p.batch_size * i, p.batch_size*(i+1))
        ridx_batch = ridx_full[indices]
        x_batch = df1_train.values[ridx_batch, :]
        sess.run(trainer, feed_dict={X: x_batch, pIn_holder: p.pIn, pHidden_holder: p.pHidden})
    toc_cpu, toc_wall = time.clock(), time.time()

    # Log per epoch #
    if (epoch == 1) or (epoch % p.display_step == 0):
        tic_log = time.time()
        # print training time
        cpu_time = round(toc_cpu - tic_cpu, 2)
        wall_time = round(toc_wall - tic_wall, 2)
        print('\n> Epoch {}: {}s cpu-time, {}s wall-time'.format(
            epoch, cpu_time, wall_time
        ))

        # mse
        mse_batch, h_batch = sess.run([mse1, h],
                                      feed_dict={
                                          X: x_batch,
                                          pIn_holder: 1.0,
                                          pHidden_holder: 1.0}
                                      )
        mse_log_batch.append(mse_batch)
        mse_valid, h_valid = sess.run([mse1, h],
                                      feed_dict={
                                          X: df1_valid,
                                          pIn_holder: 1.0,
                                          pHidden_holder: 1.0}
                                      )
        mse_log_valid.append(mse_valid)
        print('MSE1: batch: {}, valid: {}'.
              format(mse_batch, mse_valid))

        # cell-corr
        corr_batch = scimpute.median_corr(x_batch, h_batch)
        cell_corr_log_batch.append(corr_batch)
        corr_valid = scimpute.median_corr(df1_valid.values, h_valid)
        cell_corr_log_valid.append(corr_valid)
        print("Cell-corr: batch: {}, valid: {}".
              format(corr_batch, corr_valid))

        epoch_log.append(epoch)
        tb_summary()

        # todo: see if w1 updates
        print('encoder_w1[0, 0:2]: ', sess.run(e_w1)[0, 0:2])

        # todo: print RAM usage

        toc_log = time.time()
        log_time = round(toc_log - tic_log, 1)
        print('display_step: {}s'.format(log_time))

    # Log per observation interval #
    if (epoch % p.snapshot_step == 0) or (epoch == p.max_training_epochs):
        tic_log2 = time.time()
        h_train, h_valid, h_input = snapshot()  # save
        learning_curve()
        scimpute.gene_corr_hist(h_valid, df1_valid.values,
                                title="Gene-corr (H vs X) (valid)")
        scimpute.cell_corr_hist(h_valid, df1_valid.values,
                                title="Cell-corr (H vs X) (valid)")
        save_bottle_neck_representation()
        save_weights()
        visualize_weights()
        toc_log2 = time.time()
        log2_time = round(toc_log2 - tic_log2, 1)
        print('snapshot_step: {}s'.format(log2_time))

batch_writer.close()
valid_writer.close()
sess.close()
print("Finished!")