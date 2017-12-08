#!/usr/bin/python
# <step1>: pre-training w/b on reference RNA-seq data
# ignore zeros in cost function, by OMEGA

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
    print("> Epoch 0:")
    epoch_log.append(epoch)

    # mse_omega, h
    mse_omega_train, mse1_train, h_train = sess.run([mse_omega, mse1, h],
                         feed_dict={
                             X: df1_train.values,
                             pIn_holder: 1,
                             pHidden_holder: 1}
                         )
    mse_omega_valid, mse1_valid, h_valid = sess.run([mse_omega, mse1, h],
                         feed_dict={
                             X: df1_valid.values,
                             pIn_holder: 1,
                             pHidden_holder: 1}
                         )
    print('mse_omega:   train: {}, valid: {}'.
          format(mse_omega_train, mse_omega_valid))
    print('mse1:    train: {}, valid: {}'.
          format(mse1_train, mse1_valid))
    mse_omega_log_batch.append(mse_omega_train)  # approximation
    mse_omega_log_train.append(mse_omega_train)
    mse_omega_log_valid.append(mse_omega_valid)
    mse1_log_batch.append(mse1_train)  # approximation
    mse1_log_train.append(mse1_train)
    mse1_log_valid.append(mse1_valid)

    # cell-corr
    cell_corr_train = scimpute.median_corr(df1_train.values, h_train)
    cell_corr_valid = scimpute.median_corr(df1_valid.values, h_valid)
    print("Cell-corr(full): train: {}, valid: {}".
          format(cell_corr_train, cell_corr_valid))
    cell_corr_log_batch.append(cell_corr_train)
    cell_corr_log_valid.append(cell_corr_valid)

    # gene-corr
    gene_corr_batch = scimpute.median_corr(
        df1_train.values.transpose(), h_train.transpose(), num=2000)
    gene_corr_valid = scimpute.median_corr(
        df1_valid.values.transpose(), h_valid.transpose(), num=2000)
    print("Gene-corr(full): batch: {}, valid: {}".
          format(gene_corr_batch, gene_corr_valid))
    gene_corr_log_batch.append(gene_corr_batch)
    gene_corr_log_valid.append(gene_corr_valid)

    # tensorboard (tb)
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


def learning_curves():
    print('plotting learning curves')
    scimpute.learning_curve(
        epoch_log, mse_omega_log_batch, mse_omega_log_valid,
        title='Learning curve (mse_omega).{}'.format(p.stage),
        ylabel='mse_omega',
        dir=p.stage
    )
    scimpute.learning_curve(
        epoch_log, mse1_log_batch, mse1_log_valid,
        title='Learning curve (mse1).{}'.format(p.stage),
        ylabel='mse1',
        dir=p.stage
    )
    scimpute.learning_curve(
        epoch_log, cell_corr_log_batch, cell_corr_log_valid,
        title='Learning curve (cell_corr).{}'.format(p.stage),
        ylabel='Cell-corr',
        dir=p.stage
    )
    scimpute.learning_curve(
        epoch_log, gene_corr_log_batch, gene_corr_log_valid,
        title='Learning curve (gene_corr).{}'.format(p.stage),
        ylabel='Gene-corr',
        dir=p.stage
    )


def snapshot():
    '''save inference, 
    save session'''
    print("\n> Snapshot: ")
    # inference
    h_train = sess.run(h,
                       feed_dict={
                           X: df1_train.values,
                           pIn_holder: 1,
                           pHidden_holder: 1})
    h_valid = sess.run(h,
                       feed_dict={
                           X: df1_valid.values,
                           pIn_holder: 1,
                           pHidden_holder: 1})
    h_input = sess.run(h,
                       feed_dict={
                           X: df1.values,
                           pIn_holder: 1,
                           pHidden_holder: 1})

    # save pred
    df1_h_input = pd.DataFrame(data=h_input,
                               columns=df1.columns,
                               index=df1.index)
    scimpute.save_hd5(df1_h_input,
                      "{}/imputation.{}.hd5".format(p.stage, p.stage))
    # save model
    save_path = saver.save(sess, p.stage + "/step1.ckpt")
    print("Model saved in: %s" % save_path)
    return (h_train, h_valid, h_input)


def save_bottle_neck_representation():
    print("save bottle-neck_representation")
    code_bottle_neck_input = sess.run(a_bottle_neck,
                                      feed_dict={
                                          X: df1.values,
                                          pIn_holder: 1,
                                          pHidden_holder: 1})
    np.save('{}/code_neck_valid.{}.npy'.format(p.stage, p.stage), code_bottle_neck_input)


def visualize_weight(w_name, b_name):
    w = eval(w_name)
    b = eval(b_name)
    w_arr = sess.run(w)
    b_arr = sess.run(b)
    b_arr = b_arr.reshape(len(b_arr), 1)
    b_arr_T = b_arr.T
    scimpute.visualize_weights_biases(w_arr, b_arr_T,
                                      '{},{}.{}'.format(w_name, b_name, p.stage),
                                      dir=p.stage)


def visualize_weights():
    for l1 in range(1, p.l+1):
        encoder_weight = 'e_w'+str(l1)
        encoder_bias = 'e_b'+str(l1)
        visualize_weight(encoder_weight, encoder_bias)
        decoder_bias = 'd_b'+str(l1)
        decoder_weight = 'd_w'+str(l1)
        visualize_weight(decoder_weight, decoder_bias)


def save_weights():
    print('save weights in npy')
    for l1 in range(1, p.l+1):
        encoder_weight_name = 'e_w'+str(l1)
        encoder_bias_name = 'e_b'+str(l1)
        decoder_bias_name = 'd_b'+str(l1)
        decoder_weight_name = 'd_w'+str(l1)
        np.save('{}/{}.{}'.format(p.stage, encoder_weight_name, p.stage),
                sess.run(eval(encoder_weight_name)))
        np.save('{}/{}.{}'.format(p.stage, decoder_weight_name, p.stage),
                sess.run(eval(decoder_weight_name)))
        np.save('{}/{}.{}'.format(p.stage, encoder_bias_name, p.stage),
                sess.run(eval(encoder_bias_name)))
        np.save('{}/{}.{}'.format(p.stage, decoder_bias_name, p.stage),
                sess.run(eval(decoder_bias_name)))


# Start
print('Cmd: ', sys.argv)

# refresh pre_train folder
scimpute.refresh_logfolder(p.stage)

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
df1_train.to_csv('{}/df1_train.{}_index.csv'.format(p.stage, p.stage),
                 columns=[], header=False)
df1_valid.to_csv('{}/df1_valid.{}_index.csv'.format(p.stage, p.stage),
                 columns=[], header=False)
df1_test.to_csv('{}/df1_test.{}_index.csv'.format(p.stage, p.stage),
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

# with tf.name_scope('Encoder_L3'):
#     e_w3, e_b3 = scimpute.weight_bias_variable(
#         'encoder3', p.n_hidden_2, p.n_hidden_3, p.sd)
#     e_a3 = scimpute.dense_layer(
#         'encoder3', e_a2, e_w3, e_b3, pHidden_holder)
#
# # with tf.name_scope('Encoder_L4'):
# #     e_w4, e_b4 = scimpute.weight_bias_variable('encoder4', p.n_hidden_3, p.n_hidden_4, p.sd)
# #     e_a4 = scimpute.dense_layer('encoder4', e_a3, e_w4, e_b4, pHidden_holder)
# #
# # with tf.name_scope('Decoder_L4'):
# #     d_w4, d_b4 = scimpute.weight_bias_variable('decoder4', p.n_hidden_4, p.n_hidden_3, p.sd)
# #     d_a4 = scimpute.dense_layer('decoder4', e_a4, d_w4, d_b4, pHidden_holder)
#
# with tf.name_scope('Decoder_L3'):
#     d_w3, d_b3 = scimpute.weight_bias_variable(
#         'decoder3', p.n_hidden_3, p.n_hidden_2, p.sd)
#     d_a3 = scimpute.dense_layer(
#         'decoder3', e_a3, d_w3, d_b3, pHidden_holder)

with tf.name_scope('Decoder_L2'):
    d_w2, d_b2 = scimpute.weight_bias_variable(
        'decoder2', p.n_hidden_2, p.n_hidden_1, p.sd)
    d_a2 = scimpute.dense_layer(
        'decoder2', e_a2, d_w2, d_b2, pHidden_holder)

with tf.name_scope('Decoder_L1'):
    d_w1, d_b1 = scimpute.weight_bias_variable(
        'decoder1', p.n_hidden_1, n, p.sd)
    d_a1 = scimpute.dense_layer(
        'decoder1', d_a2, d_w1, d_b1, pHidden_holder)


# define input/output
a_bottle_neck = e_a2
h = d_a1

# define loss
with tf.name_scope("Metrics"):
    # todo: omega
    omega = tf.sign(X)  # 0 if 0, 1 if > 0; not possibly < 0 in our data
    print(X)
    print(omega)
    mse1 = tf.reduce_mean(tf.pow(X-h, 2))
    mse_omega = tf.reduce_mean(
                    tf.multiply(
                        tf.pow(X-h, 2),
                        omega
                        )
                )

    tf.summary.scalar('mse_omega (H vs X)', mse_omega)

# trainer
optimizer = tf.train.AdamOptimizer(p.learning_rate)
trainer = optimizer.minimize(mse_omega)

# Launch Session #
sess = tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

batch_writer = tf.summary.FileWriter(p.stage + '/batch', sess.graph)
valid_writer = tf.summary.FileWriter(p.stage + '/valid', sess.graph)

epoch = 0
num_batch = int(math.floor(len(df1_train) // p.batch_size))  # floor
epoch_log = []
mse_omega_log_batch, mse_omega_log_valid, mse_omega_log_train = [], [], []
mse1_log_batch, mse1_log_valid, mse1_log_train = [], [], []
cell_corr_log_batch, cell_corr_log_valid, cell_corr_log_train = [], [], []
gene_corr_log_batch, gene_corr_log_valid, gene_corr_log_train = [], [], []
increasing_epochs = 0
previous_mse_omega = -1  # neg is impossible for true mse_omega

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
        epoch_log.append(epoch)
        # print training time
        cpu_time = round(toc_cpu - tic_cpu, 2)
        wall_time = round(toc_wall - tic_wall, 2)
        print('\n> Epoch {}: {}s cpu-time, {}s wall-time'.format(
            epoch, cpu_time, wall_time
        ))

        # mse_omega
        mse_omega_batch, mse1_batch, h_batch = sess.run([mse_omega, mse1, h],
                                      feed_dict={
                                          X: x_batch,
                                          pIn_holder: 1.0,
                                          pHidden_holder: 1.0}
                                      )
        mse_omega_log_batch.append(mse_omega_batch)
        mse1_log_batch.append(mse1_batch)
        mse_omega_valid, mse1_valid, h_valid = sess.run([mse_omega, mse1, h],
                                      feed_dict={
                                          X: df1_valid,
                                          pIn_holder: 1.0,
                                          pHidden_holder: 1.0}
                                      )
        mse_omega_log_valid.append(mse_omega_valid)
        mse1_log_valid.append(mse1_valid)
        print('mse_omega:   batch: {}, valid: {}'.
              format(mse_omega_batch, mse_omega_valid))
        print('mse1:    batch: {}, valid: {}'.
              format(mse1_batch, mse1_valid))


        # cell-corr
        cell_corr_batch = scimpute.median_corr(x_batch, h_batch)
        cell_corr_valid = scimpute.median_corr(df1_valid.values, h_valid)
        print("Cell-corr(fast): batch: {}, valid: {}".
              format(cell_corr_batch, cell_corr_valid))
        cell_corr_log_batch.append(cell_corr_batch)
        cell_corr_log_valid.append(cell_corr_valid)

        # gene-corr
        gene_corr_batch = scimpute.median_corr(
            x_batch.transpose(), h_batch.transpose(), num=2000)
        gene_corr_valid = scimpute.median_corr(
            x_batch.transpose(), h_batch.transpose(), num=2000)
        print("Gene-corr(fast): batch: {}, valid: {}".
              format(gene_corr_batch, gene_corr_valid))
        gene_corr_log_batch.append(gene_corr_batch)
        gene_corr_log_valid.append(gene_corr_valid)

        # tensor-board
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
        learning_curves()
        scimpute.gene_corr_hist(
            h_valid, df1_valid.values,
            title="Gene-corr(H vs X)(valid).{}".format(p.stage),
            dir=p.stage
        )
        scimpute.cell_corr_hist(
            h_valid, df1_valid.values,
            title="Cell-corr(H vs X)(valid).{}".format(p.stage),
            dir=p.stage
        )
        save_bottle_neck_representation()
        save_weights()
        visualize_weights()
        toc_log2 = time.time()
        log2_time = round(toc_log2 - tic_log2, 1)
        min_mse_omega_valid = min(mse_omega_log_valid)
        print('min_mse_omega_valid till now: {}'.format(min_mse_omega_valid))
        print('snapshot_step: {}s'.format(log2_time))

    # early stop (not used yet)
    if previous_mse_omega > 0:  # skip if mse_omega == -1, no previous mse_omega yet
        if mse_omega_valid > previous_mse_omega:
            increasing_epochs += 1
        else:
            increasing_epochs = 0
    previous_mse_omega = mse_omega_valid

    if increasing_epochs >= p.patience:
        print('* Warning: {} epochs with increasing mse_omega_valid'.
              format(increasing_epochs))


batch_writer.close()
valid_writer.close()
sess.close()
print("Finished!\n\n\n")