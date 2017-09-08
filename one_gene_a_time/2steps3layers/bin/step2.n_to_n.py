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

print ('python version:', sys.version)
print('tf.__version__', tf.__version__)

def split_arr(arr, a=0.8, b=0.1, c=0.1):
    """input array, output rand split arrays
    a: train, b: valid, c: test
    e.g.: [arr_train, arr_valid, arr_test] = split(df.values)"""
    np.random.seed(1) # for splitting consistency
    train_indices = np.random.choice(arr.shape[0], int(round(arr.shape[0] * a//(a+b+c))), replace=False)
    remain_indices = np.array(list(set(range(arr.shape[0])) - set(train_indices)))
    valid_indices = np.random.choice(remain_indices, int(round(len(remain_indices) * b//(b+c))), replace=False)
    test_indices = np.array(list( set(remain_indices) - set(valid_indices) ))
    np.random.seed() # cancel seed effect
    print("total samples being split: ", len(train_indices) + len(valid_indices) + len(test_indices))
    print('train:', len(train_indices), ' valid:', len(valid_indices), 'test:', len(test_indices))

    arr_train = arr[train_indices]
    arr_valid = arr[valid_indices]
    arr_test = arr[test_indices]

    return(arr_train, arr_valid, arr_test)


def split_df(df, a=0.8, b=0.1, c=0.1):
    """input df, output rand split dfs
    a: train, b: valid, c: test
    e.g.: [df_train, df2, df_test] = split(df, a=0.7, b=0.15, c=0.15)"""
    np.random.seed(1) # for splitting consistency
    train_indices = np.random.choice(df.shape[0], int(df.shape[0] * a//(a+b+c)), replace=False)
    remain_indices = np.array(list(set(range(df.shape[0])) - set(train_indices)))
    valid_indices = np.random.choice(remain_indices, int(len(remain_indices) * b//(b+c)), replace=False)
    test_indices = np.array(list( set(remain_indices) - set(valid_indices) ))
    np.random.seed() # cancel seed effect
    print("total samples being split: ", len(train_indices) + len(valid_indices) + len(test_indices))
    print('train:', len(train_indices), ' valid:', len(valid_indices), 'test:', len(test_indices))

    df_train = df.ix[train_indices, :]
    df_valid = df.ix[valid_indices, :]
    df_test = df.ix[test_indices, :]

    return(df_train, df_valid, df_test)


def medium_corr(arr1, arr2, num=100, accuracy = 3):
    """arr1 & arr2 must have same shape
    will calculate correlation between corresponding columns"""
    # from scipy.stats.stats import pearsonr
    pearsonrlog = []
    for i in range(num - 1):
        pearsonrlog.append(pearsonr(arr1[i], arr2[i]))
    pearsonrlog.sort()
    result = round(pearsonrlog[int(num//2)][0], accuracy)
    return(result)


def save_hd5 (df, out_name):
    """save blosc compressed hd5"""
    tic = time.time()
    df.to_hdf(out_name, key='null', mode='w', complevel=9, complib='blosc')
    toc = time.time()
    print("saving" + out_name + " took {:.1f} seconds".format(toc-tic))


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


def scatterplot(x, y, title, xlabel, ylabel):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(title + '.png', bbox_inches='tight')


def evaluate_epoch0():
    cost_train = sess.run(cost, feed_dict={X: df_train.values})
    cost_valid = sess.run(cost, feed_dict={X: df_valid.values})
    print("\nEpoch 0: cost_train=", round(cost_train,3), "cost_valid=", round(cost_valid,3))
    h_train = sess.run(y_pred, feed_dict={X: df_train.values[:100]})
    h_valid = sess.run(y_pred, feed_dict={X: df_valid.values[:100]})
    print("medium benchmark_pearsonr in first 100 train cells: ", medium_corr(df2_train.values, h_train))
    print("medium benchmark_pearsonr in first 100 valid cells: ", medium_corr(df2_valid.values, h_valid))


def snapshot():
    print("#Snapshot: ")
    # show full data-set corr
    h_train = sess.run(y_pred, feed_dict={X: df_train.values})  # np.array [len(df_train),1]
    h_valid = sess.run(y_pred, feed_dict={X: df_valid.values})
    print("medium pearsonr in all train data: ", medium_corr(df2_train.values, h_train, num=len(df_train)))
    print("medium pearsonr in all valid data: ", medium_corr(df2_valid.values, h_valid, num=len(df_valid)))
    # save predictions
    h_input = sess.run(y_pred, feed_dict={X: df.values})
    print("medium pearsonr in all imputation cells: ", medium_corr(df2.values, h_input, num=m))
    df_h_input = pd.DataFrame(data=h_input, columns=df.columns, index=df.index)
    save_hd5(df_h_input, log_dir + "/imputation.step1.hd5")
    # save model
    save_path = saver.save(sess, log_dir + "/step1.ckpt")
    print("Model saved in: %s" % save_path)


def print_parameters():
    print(os.getcwd(),"\n",
        "\n# Hyper parameters:",
        "\nn_features: ", n,
        "\nn_hidden1: ", n_hidden_1,
        "\nlearning_rate :", learning_rate,
        "\nbatch_size: ", batch_size,
        "\nepoches: ", training_epochs, "\n",
        "\ndf_train.values.shape", df_train.values.shape,
        "\ndf_valid.values.shape", df_valid.values.shape,
        "\ndf2_train.shape", df2_train.shape,
        "\ndf2_valid.values.shape", df2_valid.values.shape,
        "\n")
    print("input_array:\n", df.values[0:4,0:4], "\n")


def refresh_logfolder():
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
        print (log_dir, "deleted")
    tf.gfile.MakeDirs(log_dir)
    print(log_dir, 'created')


def epoch_summary():
    run_metadata = tf.RunMetadata()
    train_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)

    h_train = sess.run(y_pred, feed_dict={X: df_train.values[:100]})
    h_valid = sess.run(y_pred, feed_dict={X: df_valid.values[:100]})
    corr_train = medium_corr(df2_train.values, h_train)
    corr_valid = medium_corr(df2_valid.values, h_valid)
    print("medium pearsonr in first 100 train cells: ", corr_train)
    print("medium pearsonr in first 100 valid cells: ", corr_valid)
    corr_log.append(corr_valid)
    epoch_log.append(epoch)

    # Summary
    merged = tf.summary.merge_all()

    [summary_train, cost_train] = sess.run([merged, cost], feed_dict={X: df_train.values, M: df2_train.values})
    [summary_valid, cost_valid] = sess.run([merged, cost], feed_dict={X: df_valid.values, M: df2_valid.values})
    train_writer.add_summary(summary_train, epoch)
    valid_writer.add_summary(summary_valid, epoch)

    print("cost_batch=", "{:.6f}".format(cost_batch),
          "cost_train=", "{:.6f}".format(cost_train),
          "cost_valid=", "{:.6f}".format(cost_valid))

# read data #
file = "../../../data/v1-1-5-3/v1-1-5-3.F3.msk.hd5" #data need imputation
file_benchmark = "../../../data/v1-1-5-3/v1-1-5-3.F3.hd5"
df = pd.read_hdf(file).transpose() #[cells,genes]
df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
m, n = df.shape  # m: n_cells; n: n_genes

# rand split data
[df_train, df_valid, df_test] = split_df(df)

df2_train = df2.ix[df_train.index]
df2_valid = df2.ix[df_valid.index]
df2_test = df2.ix[df_test.index]

# Parameters #
print ("this is just testing version, superfast and bad")
learning_rate = 0.0001
training_epochs = 1
batch_size = 256
sd = 0.0001 #stddev for random init
n_input = n
n_hidden_1 = 500
log_dir = './re_train'
display_step = 1
snapshot_step = 2500

print_parameters()

corr_log = []
epoch_log = []

# Define model #
X = tf.placeholder(tf.float32, [None, n_input])  # input
M = tf.placeholder(tf.float32, [None, n_input])  # benchmark

encoder_params = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=sd), name='encoder_w1'),
    'b1': tf.Variable(tf.random_normal([n_hidden_1], mean=3*sd, stddev=sd), name='encoder_b1')
}
decoder_params = {
    'w1': tf.Variable(tf.random_normal([n_hidden_1, n_input], stddev=sd), name='decoder_w1'),
    'b1': tf.Variable(tf.random_normal([n_input], mean=3*sd, stddev=sd), name='decoder_b1')
    # 'b1': tf.Variable(tf.ones([n_input]), name='decoder_b1') #fast, but maybe linear model

}
parameters = {**encoder_params, **decoder_params}


def encoder(x) :
    with tf.name_scope("Encoder"):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, encoder_params['w1']),
                                       encoder_params['b1']))
        variable_summaries('encoder_w1', encoder_params['w1'])
        variable_summaries('encoder_b1', encoder_params['b1'])
        variable_summaries('encoder_a1', layer_1)
    return layer_1


def decoder(x):
    with tf.name_scope("Decoder"):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, decoder_params['w1']),
                                       decoder_params['b1']))
        variable_summaries('decoder_w1', decoder_params['w1'])
        variable_summaries('decoder_b1', decoder_params['b1'])
        variable_summaries('decoder_a1', layer_1)
    return layer_1

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_true = X
y_benchmark = M
y_pred = decoder_op

with tf.name_scope("Metrics"):
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    cost_benchmark = tf.reduce_mean(tf.pow(y_pred- y_benchmark, 2))
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('cost_benchmark', cost_benchmark)

# train_op = tf.train.AdamOptimizer(learning_rate).\
#     minimize(cost, var_list=[list(decoder_params.values())])

train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=[list(decoder_params.values())])

# Launch Session #
sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "./pre_train/step1.ckpt")

train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir+'/valid', sess.graph)

evaluate_epoch0()

total_batch = int(math.floor(len(df_train)//batch_size))  # floor
# Training cycle
for epoch in range(1, training_epochs+1):
    tic_cpu = time.clock()
    tic_wall = time.time()
    random_indices = np.random.choice(len(df_train), batch_size)
    for i in range(total_batch):
        indices = np.arange(batch_size*i, batch_size*(i+1))
        batch_xs = df_train.values[indices,:]
        _, cost_batch = sess.run([train_op, cost], feed_dict={X: batch_xs})
    toc_cpu = time.clock()
    toc_wall = time.time()

    # Log per epoch
    if (epoch == 1) or (epoch % display_step == 0):
        tic_log = time.time()
        print("\n#Epoch ", epoch, " took: ",
              round(toc_cpu - tic_cpu, 2), " CPU seconds; ",
              round(toc_wall - tic_wall, 2), "Wall seconds")
        epoch_summary
        toc_log=time.time()
        print('log time for each epoch:', round(toc_log-tic_log, 1))

    # Log per observation interval
    if (epoch == 1) or (epoch % snapshot_step == 0) or (epoch == training_epochs):
        tic_log2 = time.time()
        snapshot()
        toc_log2 = time.time()
        print ('log2 time for observation intervals:', round(toc_log2 - tic_log2, 1))

train_writer.close()
valid_writer.close()

scatterplot(epoch_log, corr_log, 'correlation_metrics.step1', 'epoch', 'Pearson corr with ground truth')

print("Finished!")

