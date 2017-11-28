#!/usr/bin/python

#  todo:
# 1. [x] restore (TL)
# 2. [x] use functions in model
# 3. [x] mtask (one gene a time with one network)
# 4. [x] exclude zeros or not
# 5. [ ] search for 'change with layer' after changing layers
# 6. [ ] split analysis of final result from imputation; only plot learning curve here

# import
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
# sys.path.append('./bin')
import scimpute
import step2_params as p


def evaluate_epoch_step2():
    print("> Evaluation:")
    epoch_log.append(epoch)
    # mse2 (h vs M)
    mse2_train = sess.run(mse2, feed_dict={X: df1_train, M: df2_train,
                                           pIn_holder: 1, pHidden_holder: 1})
    mse2_valid = sess.run(mse2, feed_dict={X: df1_valid, M: df2_valid,
                                           pIn_holder: 1, pHidden_holder: 1})
    mse2_batch_vec.append(mse2_train)  # approximation
    mse2_valid_vec.append(mse2_valid)
    print("mse2_train=", round(mse2_train, 3), "mse2_valid=", round(mse2_valid, 3))

    # mse1 (h vs X)
    mse1_train = sess.run(mse1, feed_dict={X: df1_train, M: df2_train,
                                           pHidden_holder: 1.0, pIn_holder: 1.0})
    mse1_batch_vec.append(mse1_train)

    mse1_valid = sess.run(mse1, feed_dict={X: df1_valid, M: df2_valid,
                                           pHidden_holder: 1.0, pIn_holder: 1.0})
    mse1_valid_vec.append(mse1_valid)
    print("mse1_train=", round(mse1_train, 3), "mse1_valid=", round(mse1_valid, 3))

    h_train = sess.run(h, feed_dict={X: df1_train.values, pIn_holder: 1, pHidden_holder: 1})
    h_valid = sess.run(h, feed_dict={X: df1_valid.values, pIn_holder: 1, pHidden_holder: 1})
    corr2_train = scimpute.median_corr(df2_train.values, h_train)
    corr2_valid = scimpute.median_corr(df2_valid.values, h_valid)
    median_cell_corr2_batch_vec.append(corr2_train)
    median_cell_corr2_valid_vec.append(corr2_valid)
    print("Medium-cell-corr2, train, valid:", corr2_train, corr2_valid)
    # # tb todo
    # merged_summary = tf.summary.merge_all()
    # summary_batch = sess.run(merged_summary, feed_dict={X: df1_train, M: df2_train,  # M is not used here, just dummy
    #                                                     pIn_holder: 1.0, pHidden_holder: 1.0})
    # summary_valid = sess.run(merged_summary, feed_dict={X: df1_valid.values, M: df2_valid.values,
    #                                                     pIn_holder: 1.0, pHidden_holder: 1.0})
    # batch_writer.add_summary(summary_batch, epoch)
    # valid_writer.add_summary(summary_valid, epoch)


def tb_summary():
    print('> Tensorboard summaries')
    tic = time.time()
    # run_metadata = tf.RunMetadata()
    # batch_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)
    merged_summary = tf.summary.merge_all()
    summary_batch = sess.run(merged_summary, feed_dict={X: x_batch, M: x_batch,  # M is not used here, just dummy
                                                        pIn_holder: 1.0, pHidden_holder: 1.0})
    summary_valid = sess.run(merged_summary, feed_dict={X: df1_valid.values, M: df2_valid.values,
                                                        pIn_holder: 1.0, pHidden_holder: 1.0})
    batch_writer.add_summary(summary_batch, epoch)
    valid_writer.add_summary(summary_valid, epoch)
    toc = time.time()
    print('tb_summary time:', round(toc-tic,2))


def learning_curve_step2():
    print('> plotting learning curves')
    scimpute.learning_curve_mse(epoch_log, mse2_batch_vec, mse2_valid_vec,
                                title="Learning Curve MSE2.{}".format(p.stage),
                                ylabel='MSE2')
    scimpute.learning_curve_mse(epoch_log, mse1_batch_vec, mse1_valid_vec,
                                title="Learning Curve MSE1.{}".format(p.stage),
                                ylabel="MSE1")
    # scimpute.learning_curve_corr(epoch_log, median_cell_corr2_batch_vec, median_cell_corr2_valid_vec)


def snapshot():
    print("> Snapshot (save inference, save session, calculate whole dataset cell-pearsonr ): ")
    # inference
    h_train = sess.run(h, feed_dict={X: df1_train.values, pIn_holder: 1, pHidden_holder: 1})
    h_valid = sess.run(h, feed_dict={X: df1_valid.values, pIn_holder: 1, pHidden_holder: 1})
    h_input = sess.run(h, feed_dict={X: df1.values, pIn_holder: 1, pHidden_holder: 1})
    # print whole dataset pearsonr
    print("median cell-pearsonr(all train): ",
          scimpute.median_corr(df2_train.values, h_train, num=len(df1_train)))
    print("median cell-pearsonr(all valid): ",
          scimpute.median_corr(df2_valid.values, h_valid, num=len(df1_valid)))
    print("median cell-pearsonr in all imputation cells: ",
          scimpute.median_corr(df2.values, h_input, num=m))
    # save pred
    df_h_input = pd.DataFrame(data=h_input, columns=df1.columns, index=df1.index)
    scimpute.save_hd5(df_h_input, "{}/imputation.{}.hd5".format(p.stage, p.stage))
    # save model
    save_path = saver.save(sess, log_dir + "/step2.ckpt")
    print("Model saved in: %s" % save_path)
    return (h_train, h_valid, h_input)


def save_bottle_neck_representation():
    print("> save bottle-neck_representation")
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
                                      '{},{}.{}'.format(w_name, b_name, p.stage))


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


print("Usage: python -u <step2.py>")

# print versions / sys.path
print('python version:', sys.version)
print('tf.__version__', tf.__version__)
print('sys.path', sys.path)

# refresh folder
log_dir = './{}'.format(p.stage)
scimpute.refresh_logfolder(log_dir)

# read data into df1/2 [cells, genes]
if p.file_orientation == 'gene_row':
    df1 = pd.read_hdf(p.file1).transpose()
    df2 = pd.read_hdf(p.file2).transpose()
elif p.file_orientation == 'cell_row':
    df1 = pd.read_hdf(p.file1)
    df2 = pd.read_hdf(p.file2)
else:
    raise Exception('parameter err: file_orientation not correctly spelled')

# Test or not
if p.test_flag > 0:
    print('in test mode')
    df1 = df1.ix[0:p.m, 0:p.n]
    df2 = df2.ix[0:p.m, 0:p.n]

# Summary of data
print("input_name:", p.name1)
print("input_df:\n", df1.ix[0:3, 0:2], "\n")
print("grouth_truth_name:", p.name2)
print("ground_truth_df:\n", df2.ix[0:3, 0:2], "\n")
m, n = df1.shape  # m: n_cells; n: n_genes
print('{} genes, {} cells\n'.format(n, m))


# split data and save indexes
df1_train, df1_valid, df1_test = scimpute.split_df(df1,
                                                   a=0.7, b=0.15, c=0.15)
df2_train, df2_valid, df2_test = [df2.ix[df1_train.index],
                                  df2.ix[df1_valid.index],
                                  df2.ix[df1_test.index]]

df1_train.to_csv('{}/df1_train.{}_index.csv'.format(p.stage, p.stage),
                 columns=[], header=False)
df1_valid.to_csv('{}/df1_valid.{}_index.csv'.format(p.stage, p.stage),
                 columns=[], header=False)
df1_test.to_csv('{}/df1_test.{}_index.csv'.format(p.stage, p.stage),
                columns=[], header=False)


# parameters
n_input = n
# todo: print_parameters()

# Start model
tf.reset_default_graph()

# define placeholders and variables
X = tf.placeholder(tf.float32, [None, n_input], name='X_input')  # input
M = tf.placeholder(tf.float32, [None, n_input], name='M_ground_truth')  # benchmark
pIn_holder = tf.placeholder(tf.float32, name='p.pIn')
pHidden_holder = tf.placeholder(tf.float32, name='p.pHidden')
J = tf.placeholder(tf.int32, name='j')

# define layers and variables
tf.set_random_seed(3)  # seed
# change with layer
with tf.name_scope('Encoder_L1'):
    e_w1, e_b1 = scimpute.weight_bias_variable('encoder1', n, p.n_hidden_1, p.sd)
    e_a1 = scimpute.dense_layer('encoder1', X, e_w1, e_b1, pIn_holder)
with tf.name_scope('Encoder_L2'):
    e_w2, e_b2 = scimpute.weight_bias_variable('encoder2', p.n_hidden_1, p.n_hidden_2, p.sd)
    e_a2 = scimpute.dense_layer('encoder2', e_a1, e_w2, e_b2, pHidden_holder)
with tf.name_scope('Encoder_L3'):
    e_w3, e_b3 = scimpute.weight_bias_variable('encoder3', p.n_hidden_2, p.n_hidden_3, p.sd)
    e_a3 = scimpute.dense_layer('encoder3', e_a2, e_w3, e_b3, pHidden_holder)
# with tf.name_scope('Encoder_L4'):
#     e_w4, e_b4 = scimpute.weight_bias_variable('encoder4', p.n_hidden_3, p.n_hidden_4, p.sd)
#     e_a4 = scimpute.dense_layer('encoder4', e_a3, e_w4, e_b4, pHidden_holder)
# with tf.name_scope('Decoder_L4'):
#     d_w4, d_b4 = scimpute.weight_bias_variable('decoder4', p.n_hidden_4, p.n_hidden_3, p.sd)
#     d_a4 = scimpute.dense_layer('decoder4', e_a4, d_w4, d_b4, pHidden_holder)
with tf.name_scope('Decoder_L3'):
    d_w3, d_b3 = scimpute.weight_bias_variable('decoder3', p.n_hidden_3, p.n_hidden_2, p.sd)
    d_a3 = scimpute.dense_layer('decoder3', e_a3, d_w3, d_b3, pHidden_holder)
with tf.name_scope('Decoder_L2'):
    d_w2, d_b2 = scimpute.weight_bias_variable('decoder2', p.n_hidden_2, p.n_hidden_1, p.sd)
    d_a2 = scimpute.dense_layer('decoder2', d_a3, d_w2, d_b2, pHidden_holder)
with tf.name_scope('Decoder_L1'):
    d_w1, d_b1 = scimpute.weight_bias_variable('decoder1', p.n_hidden_1, n, p.sd)
    d_a1 = scimpute.dense_layer('decoder1', d_a2, d_w1, d_b1, pHidden_holder)  # todo: change input activations if model changed

# define input/output
a_bottle_neck = e_a3
h = d_a1

# define loss
with tf.name_scope("Metrics"):
    mse1_j = tf.reduce_mean(tf.pow(X[:, J] - h[:, J], 2))  # for training
    mse1 = tf.reduce_mean(tf.pow(X - h, 2))  # for report
    mse2 = tf.reduce_mean(tf.pow(M - h, 2))
    tf.summary.scalar('mse1_j', mse1_j)
    tf.summary.scalar('mse1', mse1)
    tf.summary.scalar('mse2', mse2)

# trainer
optimizer = tf.train.AdamOptimizer(p.learning_rate)
trainer = optimizer.minimize(mse1_j)  # for gene_j

# start session
sess = tf.Session()

# restore variables
saver = tf.train.Saver()
saver.restore(sess, "./pre_train/step1.ckpt")

# define tensor_board writer
batch_writer = tf.summary.FileWriter(log_dir + '/batch', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir + '/valid', sess.graph)

# prep mini-batch, and reporter vectors
epoch = 0
num_batch = int(math.floor(len(df1_train) // p.batch_size))  # floor
epoch_log = []
mse1_batch_vec, mse1_valid_vec = [], []  # mse1 = MSE(X, h)
mse2_batch_vec, mse2_valid_vec = [], []  # mse2 = MSE(M, h)
median_cell_corr1_batch_vec, median_cell_corr1_valid_vec = [], []  # median_cell_corr, for subset of cells
median_cell_corr2_batch_vec, median_cell_corr2_valid_vec = [], []  # 1 ~ (X, h); 2 ~ (M, h)

# evaluate epoch0
evaluate_epoch_step2()

# Outer loop (epochs)
for epoch in range(1, p.max_training_epochs+1):
    # training on non-zero(nz) cells for gene-j #
    tic_cpu, tic_wall = time.clock(), time.time()
    # rand mini-batch  todo: sort randx out
    # random_indices = np.random.choice(len(df1_train), p.batch_size, replace=False)
    # inner loop (mini-batches)
    for i in range(num_batch):
        # x_batch
        # indices = np.arange(p.batch_size * i, p.batch_size*(i+1))
        # x_batch = df1_train.ix[indices, :]  # [bs, n]
        # x_idx = np.random.choice(m, p.batch_size, replace=False)
        x_batch = df1_train  #todo: still batch
        # j_batch
        j = np.random.choice(range(n_input), 1)[0]  # todo: change to jlist for test, range(n_input) for real usage
        # solid data
        nz_indices = (x_batch.ix[:, j:j + 1] > 0).values  #todo: get nz first then choose mini-batch
        x_batch_nz = x_batch[nz_indices]

        sess.run(trainer, feed_dict={X: x_batch_nz,
                                     pIn_holder: p.pIn, pHidden_holder: p.pHidden,
                                     J: j})
    toc_cpu, toc_wall = time.clock(), time.time()


    # report per epoch #
    if (epoch == 1) or (epoch % p.display_step == 0):
        tic_log = time.time()

        # overview
        print('epoch: ', epoch, '; num mini-batch:', i+1)
        print('for gene', j, ', num_nz: ', len(x_batch_nz))

        # print training time
        print("\n#Epoch ", epoch, " took: ",
              round(toc_cpu - tic_cpu, 2), " CPU seconds; ",
              round(toc_wall - tic_wall, 2), "Wall seconds")

        # debug
        # print('d_w1', sess.run(d_w1[1, 0:4]))  # verified when GradDescent used

        # log mse2 (h vs M)
        mse2_batch = sess.run(mse2, feed_dict={X: x_batch, M: df2.ix[x_batch.index],
                                               pHidden_holder: 1.0, pIn_holder: 1.0})
        mse2_batch_vec.append(mse2_batch)
        print('mse2_batch:', mse2_batch)

        mse2_valid = sess.run(mse2, feed_dict={X: df1_valid, M: df2_valid,
                                               pHidden_holder: 1.0, pIn_holder: 1.0})
        mse2_valid_vec.append(mse2_valid)
        print('mse2_valid:', mse2_valid)

        # log mse1 (h vs X)
        mse1_batch = sess.run(mse1, feed_dict={X: x_batch, M: df2.ix[x_batch.index],
                                               pHidden_holder: 1.0, pIn_holder: 1.0})
        mse1_batch_vec.append(mse1_batch)
        print('mse1_batch:', mse1_batch)

        mse1_valid = sess.run(mse1, feed_dict={X: df1_valid, M: df2_valid,
                                               pHidden_holder: 1.0, pIn_holder: 1.0})
        mse1_valid_vec.append(mse1_valid)
        print('mse1_valid:', mse1_valid)

        toc_log = time.time()
        epoch_log.append(epoch)
        print('log time for each epoch:', round(toc_log - tic_log, 1))

    # report and save sess per observation interval
    if (epoch % p.snapshot_step == 0) or (epoch == p.max_training_epochs):
        tic_log2 = time.time()
        h_train, h_valid, h_input = snapshot()  # save
        learning_curve_step2()
        scimpute.gene_corr_hist(
            h_valid, df1_valid.values,
            title="Gene-corr(H vs X)(valid).{}".format(p.stage))
        scimpute.cell_corr_hist(
            h_valid, df1_valid.values,
            title="Cell-corr(H vs X)(valid).{}".format(p.stage))
        save_bottle_neck_representation()
        save_weights()
        visualize_weights()
        toc_log2 = time.time()
        log2_time = round(toc_log2 - tic_log2, 1)
        min_mse2_valid = min(mse2_log_valid)
        print('min_MSE2_valid till now: {}'.format(min_mse2_valid))
        print('snapshot_step: {}s'.format(log2_time))


evaluate_epoch_step2()  # careful, this append new mse to mse_vec


batch_writer.close()
valid_writer.close()
sess.close()
print("Finished!")

