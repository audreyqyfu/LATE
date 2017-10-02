#!/usr/bin/python
# 07/25/2017
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
          "\n# Hyper parameters:",
          "\nn_features: ", n,
          "\nn_hidden1: ", n_hidden_1,
          "\nn_hidden2: ", n_hidden_2,
          "\nlearning_rate :", learning_rate,
          "\nbatch_size: ", batch_size,
          "\nepoches: ", training_epochs, "\n",
          "\nkeep_prob_input: ", pIn,
          "\nkeep_prob_hidden: ", pHidden, "\n",
          "\nfile: ", file, "\n",
          "\nfile_benchmark: ", file_benchmark, "\n",
          "\ndf_train.values.shape", df_train.values.shape,
          "\ndf_valid.values.shape", df_valid.values.shape,
          "\ndf2_train.shape", df2_train.shape,
          "\ndf2_valid.values.shape", df2_valid.values.shape,
          "\n")
    print("input_array:\n", df.values[0:4, 0:4], "\n")


def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def evaluate_epoch0():
    cost_train = sess.run(cost, feed_dict={X: df_train.values, keep_prob_input: 1, keep_prob_hidden: 1})
    cost_valid = sess.run(cost, feed_dict={X: df_valid.values, keep_prob_input: 1, keep_prob_hidden: 1})
    print("\nEpoch 0: cost_train=", round(cost_train, 3), "cost_valid=", round(cost_valid, 3))
    h_train = sess.run(y_pred, feed_dict={X: df_train.values[:100], keep_prob_input: 1, keep_prob_hidden: 1})
    h_valid = sess.run(y_pred, feed_dict={X: df_valid.values[:100], keep_prob_input: 1, keep_prob_hidden: 1})
    print("medium benchmark-cell-pearsonr in first 100 train cells, between prediction and ground truth: ",
          scimpute.medium_corr(df2_train.values, h_train))
    print("medium benchmark-cell-pearsonr in first 100 valid cells: between prediction and ground truth:",
          scimpute.medium_corr(df2_valid.values, h_valid))


def epoch_summary():
    run_metadata = tf.RunMetadata()
    train_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)

    h_train = sess.run(y_pred, feed_dict={X: df_train.values[:100], keep_prob_input: 1, keep_prob_hidden: 1})
    h_valid = sess.run(y_pred, feed_dict={X: df_valid.values[:100], keep_prob_input: 1, keep_prob_hidden: 1})
    corr_train = scimpute.medium_corr(df2_train.values, h_train)
    corr_valid = scimpute.medium_corr(df2_valid.values, h_valid)
    print("medium cell-pearsonr in first 100 train cells: ", corr_train)
    print("medium cell-pearsonr in first 100 valid cells: ", corr_valid)
    corr_log.append(corr_valid)
    epoch_log.append(epoch)

    # Summary
    merged = tf.summary.merge_all()
    [summary_train, cost_train] = sess.run([merged, cost], feed_dict={X: df_train.values, M: df2_train.values,
                                                                      keep_prob_input: 1, keep_prob_hidden: 1})
    [summary_valid, cost_valid] = sess.run([merged, cost], feed_dict={X: df_valid.values, M: df2_valid.values,
                                                                      keep_prob_input: 1, keep_prob_hidden: 1})
    train_writer.add_summary(summary_train, epoch)
    valid_writer.add_summary(summary_valid, epoch)

    print("cost_train=", "{:.6f}".format(cost_train),
          "cost_valid=", "{:.6f}".format(cost_valid))


def snapshot():
    print("#Snapshot: ")
    # show full data-set corr
    h_train = sess.run(y_pred, feed_dict={X: df_train.values, keep_prob_input: 1, keep_prob_hidden: 1})  # np.array [len(df_train),1]
    h_valid = sess.run(y_pred, feed_dict={X: df_valid.values, keep_prob_input: 1, keep_prob_hidden: 1})
    print("medium cell-pearsonr in all train data: ",
          scimpute.medium_corr(df2_train.values, h_train, num=len(df_train)))
    print("medium cell-pearsonr in all valid data: ",
          scimpute.medium_corr(df2_valid.values, h_valid, num=len(df_valid)))
    # save predictions
    h_input = sess.run(y_pred, feed_dict={X: df.values, keep_prob_input: 1, keep_prob_hidden: 1})
    print("medium cell-pearsonr in all imputation cells: ", scimpute.medium_corr(df2.values, h_input, num=m))
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
data = 'EMT9k_log'  # EMT2730 or splatter

if data is 'splatter':  # only this mode creates gene-gene plot
    file = "../data/v1-1-5-3/v1-1-5-3.E3.hd5"  # data need imputation
    file_benchmark = "../data/v1-1-5-3/v1-1-5-3.E3.hd5"
    Aname = '(E3)'
    Bname = '(E3)'  # careful
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT2730':  # 2.7k cells used in magic paper
    file = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5" #data need imputation
    file_benchmark = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5"
    Aname = '(EMT2730)'
    Bname = '(EMT2730)'
    df = pd.read_hdf(file).transpose() #[cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.hd5"
    Aname = '(EMT9k)'
    Bname = '(EMT9k)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k_log':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"
    Aname = '(EMT9kLog)'
    Bname = '(EMT9kLog)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
else:
    raise Warning("data name not recognized!")

max = max(df.values.max(), df2.values.max())

# rand split data
[df_train, df_valid, df_test] = scimpute.split_df(df, a=0.9, b=0.1, c=0.0)

df2_train = df2.ix[df_train.index]
df2_valid = df2.ix[df_valid.index]
df2_test = df2.ix[df_test.index]

# Parameters #
learning_rate = 0.0003
training_epochs = 1000  # todo change epochs
batch_size = 256
pIn = 0.8
pHidden = 0.5
sd = 0.0001  # stddev for random init
n_input = n
n_hidden_1 = 800
n_hidden_2 = 400
n_hidden_3 = 200

log_dir = './pre_train'

display_step = 20
snapshot_step = 1000

scimpute.refresh_logfolder(log_dir)

print_parameters()

corr_log = []
epoch_log = []

# Define model #
X = tf.placeholder(tf.float32, [None, n_input])  # input
M = tf.placeholder(tf.float32, [None, n_input])  # benchmark

keep_prob_input = tf.placeholder(tf.float32)
keep_prob_hidden = tf.placeholder(tf.float32)

tf.set_random_seed(3)  # seed
encoder_params = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=sd), name='encoder_w1'),
    'b1': tf.Variable(tf.random_normal([n_hidden_1], mean=100 * sd, stddev=sd), name='encoder_b1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=sd), name='encoder_w2'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], mean=100 * sd, stddev=sd), name='encoder_b2'),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=sd), name='encoder_w3'),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], mean=100 * sd, stddev=sd), name='encoder_b3')
}
decoder_params = {
    'w1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2], stddev=sd), name='decoder_w1'),
    'b1': tf.Variable(tf.random_normal([n_hidden_2], mean=100 * sd, stddev=sd), name='decoder_b1'),
    # 'b1': tf.Variable(tf.ones([n_input]), name='decoder_b1') #fast, but maybe linear model
    'w2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1], stddev=sd), name='decoder_w2'),
    'b2': tf.Variable(tf.random_normal([n_hidden_1], mean=100 * sd, stddev=sd), name='decoder_b2'),
    'w3': tf.Variable(tf.random_normal([n_hidden_1, n_input], stddev=sd), name='decoder_w3'),
    'b3': tf.Variable(tf.random_normal([n_input], mean=100 * sd, stddev=sd), name='decoder_b3')
}
parameters = {**encoder_params, **decoder_params}


def encoder(x):
    with tf.name_scope("Encoder"):
        # Encoder Hidden layer with sigmoid activation #1
        x_drop = tf.nn.dropout(x, keep_prob_input)
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x_drop, encoder_params['w1']),
                                    encoder_params['b1']))
        layer_1_drop = tf.nn.dropout(layer_1, keep_prob_hidden)
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_drop, encoder_params['w2']),
                                    encoder_params['b2']))
        layer_2_drop = tf.nn.dropout(layer_2, keep_prob_hidden)
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_drop, encoder_params['w3']),
                                    encoder_params['b3']))

        variable_summaries('encoder_w1', encoder_params['w1'])
        variable_summaries('encoder_w2', encoder_params['w2'])
        variable_summaries('encoder_w3', encoder_params['w3'])
        variable_summaries('encoder_b1', encoder_params['b1'])
        variable_summaries('encoder_b2', encoder_params['b2'])
        variable_summaries('encoder_b3', encoder_params['b3'])
        variable_summaries('encoder_a1', layer_1)
        variable_summaries('encoder_a2', layer_2)
        variable_summaries('encoder_a3', layer_3)
    return layer_3


def decoder(x):
    with tf.name_scope("Decoder"):
        # Encoder Hidden layer with sigmoid activation #1
        x_drop = tf.nn.dropout(x, keep_prob_hidden)
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x_drop, decoder_params['w1']),
                                    decoder_params['b1']))
        layer_1_drop = tf.nn.dropout(layer_1, keep_prob_hidden)
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_drop, decoder_params['w2']),
                                       decoder_params['b2']))
        layer_2_drop = tf.nn.dropout(layer_2, keep_prob_hidden)
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_drop, decoder_params['w3']),
                                       decoder_params['b3']))

        variable_summaries('decoder_w1', decoder_params['w1'])
        variable_summaries('decoder_w2', decoder_params['w2'])
        variable_summaries('decoder_w3', decoder_params['w3'])
        variable_summaries('decoder_b1', decoder_params['b1'])
        variable_summaries('decoder_b2', decoder_params['b2'])
        variable_summaries('decoder_b3', decoder_params['b3'])
        variable_summaries('decoder_a1', layer_1)
        variable_summaries('decoder_a2', layer_2)
        variable_summaries('decoder_a3', layer_3)
    return layer_3


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_true = X
y_benchmark = M
y_pred = decoder_op

with tf.name_scope("Metrics"):
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    cost_benchmark = tf.reduce_mean(tf.pow(y_pred - y_benchmark, 2))
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('cost_benchmark', cost_benchmark)

train_op = tf.train.AdamOptimizer(learning_rate). \
    minimize(cost, var_list=[list(decoder_params.values()), list(encoder_params.values())])

# Launch Session #
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()

train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir + '/valid', sess.graph)

evaluate_epoch0()

total_batch = int(math.floor(len(df_train) // batch_size))  # floor

epoch = 0
epoch_summary()

# Training cycle
for epoch in range(1, training_epochs + 1):
    tic_cpu = time.clock()
    tic_wall = time.time()
    random_indices = np.random.choice(len(df_train), batch_size)
    for i in range(total_batch):
        indices = np.arange(batch_size * i, batch_size * (i + 1))
        batch_xs = df_train.values[indices, :]
        _, cost_batch = sess.run([train_op, cost], feed_dict={X: batch_xs,
                                                              keep_prob_input: pIn, keep_prob_hidden: pHidden})
    toc_cpu = time.clock()
    toc_wall = time.time()

    # Log per epoch
    if (epoch == 1) or (epoch % display_step == 0):
        tic_log = time.time()
        print("\n#Epoch ", epoch, " took: ",
              round(toc_cpu - tic_cpu, 2), " CPU seconds; ",
              round(toc_wall - tic_wall, 2), "Wall seconds")
        epoch_summary()
        toc_log = time.time()
        print('log time for each epoch:', round(toc_log - tic_log, 1))

    # Log per observation interval
    if (epoch == 1) or (epoch % snapshot_step == 0) or (epoch == training_epochs):
        tic_log2 = time.time()
        snapshot()
        toc_log2 = time.time()
        print('log2 time for observation intervals:', round(toc_log2 - tic_log2, 1))

train_writer.close()
valid_writer.close()


# summaries and plots #
# calculation
h_valid = sess.run(y_pred, feed_dict={X: df_valid.values, keep_prob_input: 1, keep_prob_hidden: 1})
h = sess.run(y_pred, feed_dict={X: df.values, keep_prob_input: 1, keep_prob_hidden: 1})
code_neck_valid = sess.run(encoder_op, feed_dict={X: df_valid.values, keep_prob_input: 1, keep_prob_hidden: 1})

# learning curve
scimpute.curveplot(epoch_log, corr_log,
                     title='learning_curve_pearsonr.step1',
                     xlabel='epoch'+' (final corr = ' + str(corr_log[-1]) + ')',
                     ylabel='Medium Cell-Pearsonr for first 100 cells\n(predction vs ground truth, valid)')

# gene-corr hist
hist = scimpute.gene_corr_hist(h_valid, df2_valid.values,
                                  fprefix='hist gene-corr, valid, step1',
                                  title="gene-corr, prediction vs ground-truth"
                                  )

# gene-correlation for gene-j
for j in [0, 1, 200, 201, 400, 401, 600, 601, 800, 801, 998, 999]:
    scimpute.scatterplot2(df2_valid.values[:, j], h_valid[:, j],
                          title=str('scatterplot, gene-' + str(j) + ', valid, step1'),
                          xlabel='Ground Truth ' + Bname,
                          ylabel='Prediction ' + Aname
                          )

# visualization of weights (new way), get weights
encoder_w1 = sess.run(encoder_params['w1'])  #1000, 500
encoder_b1 = sess.run(encoder_params['b1'])  #500, (do T)
encoder_b1 = encoder_b1.reshape(len(encoder_b1), 1)
encoder_b1_T = encoder_b1.T

encoder_w2 = sess.run(encoder_params['w2'])
encoder_b2 = sess.run(encoder_params['b2'])
encoder_b2 = encoder_b2.reshape(len(encoder_b2), 1)
encoder_b2_T = encoder_b2.T

encoder_w3 = sess.run(encoder_params['w3'])
encoder_b3 = sess.run(encoder_params['b3'])
encoder_b3 = encoder_b3.reshape(len(encoder_b3), 1)
encoder_b3_T = encoder_b3.T

decoder_w1 = sess.run(decoder_params['w1'])
decoder_b1 = sess.run(decoder_params['b1'])
decoder_b1 = decoder_b1.reshape(len(decoder_b1), 1)
decoder_b1_T = decoder_b1.T

decoder_w2 = sess.run(decoder_params['w2'])
decoder_b2 = sess.run(decoder_params['b2'])
decoder_b2 = decoder_b2.reshape(len(decoder_b2), 1)
decoder_b2_T = decoder_b2.T

decoder_w3 = sess.run(decoder_params['w3'])
decoder_b3 = sess.run(decoder_params['b3'])
decoder_b3 = decoder_b3.reshape(len(decoder_b3), 1)
decoder_b3_T = decoder_b3.T
# visualize weights/biases
scimpute.visualize_weights_biases(encoder_w1, encoder_b1_T, 'encoder_w1, b1')
scimpute.visualize_weights_biases(encoder_w2, encoder_b2_T, 'encoder_w2, b2')
scimpute.visualize_weights_biases(encoder_w3, encoder_b3_T, 'encoder_w3, b3')
scimpute.visualize_weights_biases(decoder_w1, decoder_b1_T, 'decoder_w1, b1')
scimpute.visualize_weights_biases(decoder_w2, decoder_b2_T, 'decoder_w2, b2')
scimpute.visualize_weights_biases(decoder_w3, decoder_b3_T, 'decoder_w3, b3')
# save weights
scimpute.save_csv(encoder_w1, 'encoder_w1.csv')
scimpute.save_csv(encoder_w2, 'encoder_w2.csv')
scimpute.save_csv(encoder_w3, 'encoder_w3.csv')
scimpute.save_csv(decoder_w1, 'decoder_w1.csv')
scimpute.save_csv(decoder_w2, 'decoder_w2.csv')
scimpute.save_csv(decoder_w3, 'decoder_w3.csv')

# visualizing dfs
visualization_of_dfs()

# visualizing activations
scimpute.heatmap_vis(code_neck_valid, title='code bottle-neck, valid', xlab='nodes', ylab='cells')
# save activations
np.save(log_dir + '/code_neck_valid', code_neck_valid)

sess.close()
print("Finished!")

