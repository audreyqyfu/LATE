#!/usr/bin/python
# this is the wrapper for step1

# import modules and print versions
from __future__ import division  # fix division // get float bug
from __future__ import print_function  # fix printing \n

import tensorflow as tf
import numpy as np
import math
import seaborn as sns
import pandas as pd
from scipy.stats.stats import pearsonr
import sys
import os
import time
import matplotlib
matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt

sys.path.append('./bin')
print('sys.path', sys.path)
import scimpute
# import hl_func
# import importlib  # for development: reload modules in pycharm
# importlib.reload(hl_func)

print('python version:', sys.version)
print('tf.__version__', tf.__version__)


def print_parameters():
    print(os.getcwd(), "\n",
          "\n# Parameters: {}p.L".format(p.L),
          "\nn_features: ", n)
    for l1 in range(1, p.l+1):
      print("n_hidden{}: {}".format(l1, eval('p.n_hidden_'+str(l1))))
    print(
          "\np.learning_rate :", p.learning_rate,
          "\np.batch_size: ", p.batch_size,
          "\nepoches: ", p.training_epochs,
          "\np.display_step (interval on learning curve): {}epochs".format(p.display_step),
          "\np.snapshot_step (interval of saving session, imputation): {}epochs".format(p.snapshot_step),
          "\n",
          "\npIn_holder: ", p.pIn,
          "\npHidden_holder: ", p.pHidden,
          "\n",
          "\ndf_train.values.shape", df_train.values.shape,
          "\ndf_valid.values.shape", df_valid.values.shape,
          "\ndf2_train.shape", df2_train.shape,
          "\ndf2_valid.values.shape", df2_valid.values.shape,
          "\n")
    print("input_array:\n", df.values[0:4, 0:4], "\n")


def evaluate_epoch0():
    print("> Evaluate epoch 0:")
    epoch_log.append(epoch)
    mse_train = sess.run(mse_input, feed_dict={X: df_train.values, pIn_holder: 1, pHidden_holder: 1})
    mse_valid = sess.run(mse_input, feed_dict={X: df_valid.values, pIn_holder: 1, pHidden_holder: 1})
    mse_log_batch.append(mse_train)  # approximation
    mse_log_train.append(mse_train)
    mse_log_valid.append(mse_valid)
    print("mse_train=", round(mse_train, 3), "mse_valid=", round(mse_valid, 3))

    h_train = sess.run(h, feed_dict={X: df_train.values, pIn_holder: 1, pHidden_holder: 1})
    h_valid = sess.run(h, feed_dict={X: df_valid.values, pIn_holder: 1, pHidden_holder: 1})
    corr_train = scimpute.median_corr(df_train.values, h_train)
    corr_valid = scimpute.median_corr(df_valid.values, h_valid)
    cell_corr_log_batch.append(corr_train)
    cell_corr_log_train.append(corr_train)
    cell_corr_log_valid.append(corr_valid)
    print("Cell-pearsonr train, valid:", corr_train, corr_valid)
    # tb
    merged_summary = tf.summary.merge_all()
    summary_batch = sess.run(merged_summary, feed_dict={X: df_train, M: df2_train,  # M is not used here, just dummy
                                                        pIn_holder: 1.0, pHidden_holder: 1.0})
    summary_valid = sess.run(merged_summary, feed_dict={X: df_valid.values, M: df2_valid.values,
                                                        pIn_holder: 1.0, pHidden_holder: 1.0})
    batch_writer.add_summary(summary_batch, epoch)
    valid_writer.add_summary(summary_valid, epoch)


def tb_summary():
    print('> Tensorboard summaries')
    tic = time.time()
    # run_metadata = tf.RunMetadata()
    # batch_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)
    merged_summary = tf.summary.merge_all()
    summary_batch = sess.run(merged_summary, feed_dict={X: x_batch, M: x_batch,  # M is not used here, just dummy
                                                        pIn_holder: 1.0, pHidden_holder: 1.0})
    summary_valid = sess.run(merged_summary, feed_dict={X: df_valid.values, M: df2_valid.values,
                                                        pIn_holder: 1.0, pHidden_holder: 1.0})
    batch_writer.add_summary(summary_batch, epoch)
    valid_writer.add_summary(summary_valid, epoch)
    toc = time.time()
    print('tb_summary time:', round(toc-tic,2))


def learning_curve():
    print('> plotting learning curves')
    scimpute.learning_curve_mse(epoch_log, mse_log_batch, mse_log_valid)
    scimpute.learning_curve_corr(epoch_log, cell_corr_log_batch, cell_corr_log_valid)


def snapshot():
    print("> Snapshot (save inference, save session, calculate whole dataset cell-pearsonr ): ")
    # inference
    h_train = sess.run(h, feed_dict={X: df_train.values, pIn_holder: 1, pHidden_holder: 1})
    h_valid = sess.run(h, feed_dict={X: df_valid.values, pIn_holder: 1, pHidden_holder: 1})
    h_input = sess.run(h, feed_dict={X: df.values, pIn_holder: 1, pHidden_holder: 1})
    # print whole dataset pearsonr
    print("median cell-pearsonr(all train): ",
          scimpute.median_corr(df2_train.values, h_train, num=len(df_train)))
    print("median cell-pearsonr(all valid): ",
          scimpute.median_corr(df2_valid.values, h_valid, num=len(df_valid)))
    print("median cell-pearsonr in all imputation cells: ",
          scimpute.median_corr(df2.values, h_input, num=m))
    # save pred
    df_h_input = pd.DataFrame(data=h_input, columns=df.columns, index=df.index)
    scimpute.save_hd5(df_h_input, log_dir + "/imputation.step1.hd5")
    # save model
    save_path = saver.save(sess, log_dir + "/step1.ckpt")
    print("Model saved in: %s" % save_path)
    return (h_train, h_valid, h_input)


def save_bottle_neck_representation():
    print("> save bottle-neck_representation")
    # todo: change variable name for each model
    code_bottle_neck_input = sess.run(a_bottle_neck, feed_dict={X: df.values, pIn_holder: 1, pHidden_holder: 1})
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


def visualization_of_dfs():
    print('visualization of dfs')
    max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid])
    # max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid, h, df.values])
    scimpute.heatmap_vis(df_valid.values, title='df.valid'+name1, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(h_valid, title='h.valid'+name1, xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(df.values, title='df'+name1, xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(h, title='h'+name1, xlab='genes', ylab='cells', vmax=max, vmin=min)


# refresh pre_train folder
log_dir = './pre_train'
scimpute.refresh_logfolder(log_dir)


# read data and save indexes #

# EMT.MAGIC
# file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # input
# file2 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # ground truth (same as input in step1)
# name1 = '(EMT_MAGIC_A)'
# name2 = '(EMT_MAGIC_A)'
# gtex_gene
file = "../../../../data/gtex/gtex_v7.norm.log.hd5"  # input
file2 = "../../../../data/gtex/gtex_v7.norm.log.hd5"  # ground truth (same as input in step1)
name1 = '(gtex_gene)'  # todo: uses 20GB of RAM
name2 = '(gtex_gene)'
# read
df = pd.read_hdf(file).transpose()  # [cells,genes]
df2 = df  # same for step1

m, n = df.shape  # m: n_cells; n: n_genes
print("\ninput df: ", name1, " ", file, "\n", df.values[0:4, 0:4], "\n")
print("ground-truth df: ", name2, " ", file2, "\n", df2.values[0:4, 0:4], "\n")
# df, df2, name1, name2, m, n = scimpute.read_data('EMT9k_log')  # used during development

max = max(df.values.max(), df2.values.max())
df_train, df_valid, df_test = scimpute.split_df(df, a=0.7, b=0.15, c=0.15)
df2_train, df2_valid, df2_test = df2.ix[df_train.index], df2.ix[df_valid.index], df2.ix[df_test.index]
df_train.to_csv('pre_train/df_train.index.csv', columns=[], header=False)  # save index for future use
df_valid.to_csv('pre_train/df_valid.index.csv', columns=[], header=False)
df_test.to_csv('pre_train/df_test.index.csv', columns=[], header=False)

# Parameters #
import step1_params as p
n_input = n
print_parameters()  # todo: use logger, dict

# Define model #
tf.reset_default_graph()

# placeholders
X = tf.placeholder(tf.float32, [None, n_input], name='X_input')  # input
M = tf.placeholder(tf.float32, [None, n_input], name='M_ground_truth')  # benchmark

pIn_holder = tf.placeholder(tf.float32, name='pIn')
pHidden_holder = tf.placeholder(tf.float32, name='pHidden')

# init variables and build graph
tf.set_random_seed(3)  # seed
# update for different depth
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
#
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
    d_a1 = scimpute.dense_layer('decoder1', d_a2, d_w1, d_b1, pHidden_holder)

a_bottle_neck = e_a3

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

trainer = tf.train.AdamOptimizer(p.learning_rate).minimize(mse_input)

# Launch Session #
sess = tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

batch_writer = tf.summary.FileWriter(log_dir + '/batch', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir + '/valid', sess.graph)

epoch = 0
num_batch = int(math.floor(len(df_train) // p.batch_size))  # floor
epoch_log = []
mse_log_batch, mse_log_valid, mse_log_train = [], [], []
cell_corr_log_batch, cell_corr_log_valid, cell_corr_log_train = [], [], []

evaluate_epoch0()

# training
for epoch in range(1, p.training_epochs+1):
    # training model
    tic_cpu, tic_wall = time.clock(), time.time()
    ridx_full = np.random.choice(len(df_train), len(df_train), replace=False)
    for i in range(num_batch):
        indices = np.arange(p.batch_size * i, p.batch_size*(i+1))
        ridx_batch = ridx_full[indices]
        x_batch = df_train.values[ridx_batch, :]
        sess.run(trainer, feed_dict={X: x_batch, pIn_holder: p.pIn, pHidden_holder: p.pHidden})
    toc_cpu, toc_wall = time.clock(), time.time()

     # Log per epoch
    if (epoch == 1) or (epoch % p.display_step == 0):
        tic_log = time.time()
        # print training time
        print("\n#Epoch ", epoch, " took: ",
              round(toc_cpu - tic_cpu, 2), " CPU seconds; ",
              round(toc_wall - tic_wall, 2), "Wall seconds")

        # Ad hoc summaries
        mse_batch, h_batch = sess.run([mse_input, h], feed_dict={X: x_batch, pIn_holder: 1.0, pHidden_holder: 1.0})
        mse_log_batch.append(mse_batch)
        mse_valid, h_valid = sess.run([mse_input, h], feed_dict={X: df_valid, pIn_holder: 1.0, pHidden_holder: 1.0})
        mse_log_valid.append(mse_valid)
        print('mse_batch, valid:', mse_batch, mse_valid)
        # mse_train, h_train = sess.run([mse_input, h], feed_dict={X: df_train, pIn_holder:1.0, pHidden_holder:1.0})
        # mse_log_train.append(mse_train)
        # print('mse_batch, train, valid:', mse_batch, mse_train, mse_valid)

        corr_batch = scimpute.median_corr(x_batch, h_batch)
        cell_corr_log_batch.append(corr_batch)
        corr_valid = scimpute.median_corr(df_valid.values, h_valid)
        cell_corr_log_valid.append(corr_valid)
        print("cell-pearsonr in batch, valid:", corr_batch, corr_valid)
        # corr_train = scimpute.median_corr(df_train.values, h_train)
        # cell_corr_log_train.append(corr_train)
        # print("cell-pearsonr in batch, train, valid:", corr_batch, corr_train, corr_valid)

        epoch_log.append(epoch)

        # tb summary
        tb_summary()

        # temp: show weights in layer1, and see if it updates in deep network
        print('encoder_w1[0, 0:2]: ', sess.run(e_w1)[0, 0:2])

        toc_log = time.time()
        print('log time for each epoch:', round(toc_log - tic_log, 1))

    # Log per observation interval
    if (epoch % p.snapshot_step == 0) or (epoch == p.training_epochs):
        tic_log2 = time.time()
        h_train, h_valid, h_input = snapshot()  # save session and imputation
        learning_curve()
        scimpute.gene_corr_hist(h_valid, df2_valid.values,
                                title="gene-corr (prediction vs ground-truth) (valid)")
        scimpute.cell_corr_hist(h_valid, df2_valid.values,
                                title="cell-corr (prediction vs ground-truth) (valid)")
        save_bottle_neck_representation()
        save_weights()
        visualize_weights()
        toc_log2 = time.time()
        print('log2 time for observation intervals:', round(toc_log2 - tic_log2, 1))

batch_writer.close()
valid_writer.close()
sess.close()
print("Finished!")
