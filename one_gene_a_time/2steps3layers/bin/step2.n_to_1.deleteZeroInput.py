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
    cost_train = sess.run(cost, feed_dict={X: df_train.values})
    cost_valid = sess.run(cost, feed_dict={X: df_valid.values})
    print("\nEpoch 0: cost_train=", round(cost_train,3), "cost_valid=", round(cost_valid,3))
    h_input = sess.run(y_pred, feed_dict={X: df.values})
    print('corr', pearsonr(h_input, df.values[:,j:j+1]))
    # print("prediction:\n", h_input, "\ntruth:\n", df2.values[:,j:j+1])


def snapshot():
    print("#Snapshot: ")
    h_input = sess.run(y_pred, feed_dict={X: df.values})
    print('corr', pearsonr(h_input, df.values[:,j:j+1]))
    # print("prediction:\n", h_input, "\ntruth:\n", df2.values[:,j:j+1])
    df_h_input = pd.DataFrame(data=h_input, columns=df.columns[j:j+1], index=df.index)
    scimpute.save_hd5(df_h_input, log_dir + "/imputation.step2.hd5")
    # save model
    save_path = saver.save(sess, log_dir + "/step2.ckpt")
    print("Model saved in: %s" % save_path)


def print_parameters():
    print(os.getcwd(),"\n",
        "\n# Hyper parameters:",
        "\nn_features: ", n,
        "\nn_hidden1: ", n_hidden_1,
        "\nlearning_rate :", learning_rate,
        "\nbatch_size: ", batch_size,
        "\nepoches: ", training_epochs, "\n",
        "\ndf_train.shape", df_train.values.shape,
        "\ndf_valid.shape", df_valid.values.shape,
        "\ndf2_train.shape", df2_train.values.shape,
        "\ndf2_valid.shape", df2_valid.values.shape,
        "\ndf2_train_solid.shape", df2_train_solid.shape,
        "\ndf2_valid_solid.values.shape", df2_valid_solid.values.shape,
        "\ndf_train_solid.shape", df_train_solid.values.shape,
        "\ndf_valid_solid.shape", df_valid_solid.values.shape)
    print("input_array:\n", df.values[0:4,0:4], "\n")


# read data #
file = "../data/v1-1-5-3/v1-1-5-3.F3.hd5" #data need imputation
file_benchmark = "../data/v1-1-5-3/v1-1-5-3.F3.hd5"
Aname = '(F3)'
Bname = '(F3)'
df = pd.read_hdf(file).transpose() #[cells,genes]
df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
m, n = df.shape  # m: n_cells; n: n_genes


# Parameters #
print ("this is just testing version, superfast and bad")
j = 400
print("\n\n>>> for gene", j)

learning_rate = 0.002
training_epochs = 10000  # change to 10000
batch_size = 128
sd = 0.0001 #stddev for random init
n_input = n
n_hidden_1 = 500
log_dir = './re_train' + '_j' + str(j)
scimpute.refresh_logfolder(log_dir)
display_step = 100  # change to 100
snapshot_step = 1000

corr_log = []
epoch_log = []

# rand split data
[df_train, df_valid, df_test] = scimpute.split_df(df)
# filter data
solid_row_id_train = (df_train.ix[:, j:j+1] > 0).values
df_train_solid = df_train[solid_row_id_train]
solid_row_id_valid = (df_valid.ix[:, j:j+1] > 0).values
df_valid_solid = df_valid[solid_row_id_valid]
solid_row_id_test = (df_test.ix[:, j:j+1] > 0).values
df_test_solid = df_test[solid_row_id_test]

# df2 benchmark
df2_train_solid = df2.ix[df_train_solid.index]
df2_valid_solid = df2.ix[df_valid_solid.index]
df2_test_solid = df2.ix[df_test_solid.index]
df2_train = df2.ix[df_train.index]
df2_valid = df2.ix[df_valid.index]
df2_test = df2.ix[df_test.index]
print_parameters()


# Define model #
X = tf.placeholder(tf.float32, [None, n_input])  # input
M = tf.placeholder(tf.float32, [None, n_input])  # benchmark

encoder_params = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=sd), name='encoder_w1'),
    'b1': tf.Variable(tf.random_normal([n_hidden_1], mean=30*sd, stddev=sd), name='encoder_b1')
}


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


# Launch Session #
sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "./pre_train/step1.ckpt")

tf.set_random_seed(4)  # seed
focusFnn_params = {
    'w1': tf.Variable(tf.random_normal([n_hidden_1, 1], stddev=sd), name='focusFnn_w1'),
    'b1': tf.Variable(tf.random_normal([1], mean=30*sd, stddev=sd), name='focusFnn_b1')
}

init_new = tf.variables_initializer(list(focusFnn_params.values()))
sess.run(init_new)

def focusFnn(x):
    with tf.name_scope("Decoder"):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, focusFnn_params['w1']),
                                       focusFnn_params['b1']))
        variable_summaries('fnn_w1', focusFnn_params['w1'])
        variable_summaries('fnn_b1', focusFnn_params['b1'])
        variable_summaries('fnn_a1', layer_1)
    return layer_1

encoder_op = encoder(X)
fnn_op = focusFnn(encoder_op)

y_true = X[:, j:j+1]
y_benchmark = M[:, j:j+1]
y_pred = fnn_op

with tf.name_scope("Metrics"):
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    cost_benchmark = tf.reduce_mean(tf.pow(y_benchmark - y_pred, 2))
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('cost_benchmark', cost_benchmark)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=[list(focusFnn_params.values())])

train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir+'/valid', sess.graph)

evaluate_epoch0()

total_batch = int(math.floor(len(df_train_solid)//batch_size))  # floor

# Training cycle
for epoch in range(1, training_epochs+1):
    tic_cpu = time.clock()
    tic_wall = time.time()
    random_indices = np.random.choice(len(df_train_solid), batch_size)
    for i in range(total_batch):
        indices = np.arange(batch_size*i, batch_size*(i+1))
        batch_xs = df_train_solid.values[indices,:]
        _, cost_batch = sess.run([train_op, cost], feed_dict={X: batch_xs})
    toc_cpu = time.clock()
    toc_wall = time.time()

    # Log per epoch
    if (epoch == 1) or (epoch % display_step == 0):
        tic_log = time.time()
        print("\n#Epoch ", epoch, " took: ",
              round(toc_cpu - tic_cpu, 2), " CPU seconds; ",
              round(toc_wall - tic_wall, 2), "Wall seconds")

        def epoch_summary():
            run_metadata = tf.RunMetadata()
            train_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)

            # Summary
            merged = tf.summary.merge_all()

            [summary_train, cost_train] = sess.run([merged, cost],
                                                   feed_dict={X: df_train_solid.values, M: df2_train_solid.values})
            [summary_valid, cost_valid] = sess.run([merged, cost],
                                                   feed_dict={X: df_valid_solid.values, M: df2_valid_solid.values})
            train_writer.add_summary(summary_train, epoch)
            valid_writer.add_summary(summary_valid, epoch)

            print("cost_batch=", "{:.6f}".format(cost_batch),
                  "cost_train=", "{:.6f}".format(cost_train),
                  "cost_valid=", "{:.6f}".format(cost_valid))

        epoch_summary()

        # log corr
        h_train = sess.run(y_pred, feed_dict={X: df_train.values})
        h_valid = sess.run(y_pred, feed_dict={X: df_valid.values})
        # print("prediction_train:\n", h_train[0:5,:], "\ntruth_train:\n", df2_train_solid.values[0:5, j:j + 1])
        # print("prediction_valid:\n", h_valid[0:5,:], "\ntruth_valid:\n", df2_valid_solid.values[0:5, j:j + 1])
        corr_train = scimpute.corr_one_gene(df2_train.values[:,j:j+1], h_train)
        corr_valid = scimpute.corr_one_gene(df2_valid.values[:,j:j+1], h_valid)
        corr_log.append(corr_valid)
        epoch_log.append(epoch)


        toc_log=time.time()
        print('log time for each epoch:', round(toc_log-tic_log, 1))

    # Log per observation interval
    if (epoch == 1) or (epoch % snapshot_step == 0) or (epoch == training_epochs):
        tic_log2 = time.time()
        snapshot()
        # vis
        encoder_w1 = sess.run(encoder_params['w1'])
        encoder_b1 = sess.run(encoder_params['b1'])
        encoder_b1 = encoder_b1.reshape(len(encoder_b1), 1)

        focusFnn_w1 = sess.run(focusFnn_params['w1'])
        focusFnn_w1 = focusFnn_w1.reshape(len(focusFnn_w1), 1)
        focusFnn_b1 = sess.run(focusFnn_params['b1'])
        focusFnn_b1 = focusFnn_b1.reshape(len(focusFnn_b1), 1)

        toc_log2 = time.time()
        print ('log2 time for observation intervals:', round(toc_log2 - tic_log2, 1))

train_writer.close()
valid_writer.close()


# summaries and plots #
h_valid = sess.run(y_pred, feed_dict={X: df_valid.values})
h_valid_solid = sess.run(y_pred, feed_dict={X: df_valid_solid.values})
h = sess.run(y_pred, feed_dict={X: df.values})

# code_neck_valid_solid = sess.run(encoder_op, feed_dict={X: df_valid_solid.values})
# code_neck_valid = sess.run(encoder_op, feed_dict={X: df_valid.values})

# learning curve
scimpute.curveplot(epoch_log, corr_log,
                     title='learning_curve_pearsonr.step2',
                     xlabel='epoch',
                     ylabel='Pearson corr (predction vs ground truth, valid, including cells with zero gene-j)')

# gene-correlation for gene-j
scimpute.scatterplot2(df2_valid.values[:, j], h_valid[:,0],
                      title=str('scatterplot, gene-' + str(j) + ', valid, step2'),
                      xlabel='Ground Truth ' + Aname,
                      ylabel='Prediction ' + Bname
                      )

# visualization of weights (new way)
encoder_w1 = sess.run(encoder_params['w1'])  #1000, 500
encoder_b1 = sess.run(encoder_params['b1'])  #500, (do T)
encoder_b1 = encoder_b1.reshape(len(encoder_b1), 1)
encoder_b1_T = encoder_b1.T

scimpute.visualize_weights_biases(encoder_w1, encoder_b1_T, 'encoder_w1, b1')

# problem
focusFnn_w1 = sess.run(focusFnn_params['w1'])  #500, 1
focusFnn_b1 = sess.run(focusFnn_params['b1'])  #1
focusFnn_b1 = focusFnn_b1.reshape(len(focusFnn_b1), 1)
focusFnn_b1_T = focusFnn_b1.T

scimpute.visualize_weights_biases(focusFnn_w1, focusFnn_b1_T, 'focusFnn_w1, b1')
# scimpute.heatmap_vis(code_neck_valid, title='code_neck_valid, all cells' + Bname, xlab='', ylab='', vmax=max, vmin=min)

# vis df
def visualization_of_dfs():
    max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid_solid, h, df.values, df2.values])
    scimpute.heatmap_vis(df_valid_solid.values, title='df.valid.solid'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(h_valid_solid, title='h.valid.solid'+Aname+'.pred', xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(df_valid.values, title='df.valid.all'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(h_valid, title='h.valid.all'+Aname+'.pred', xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(df.values, title='df.all'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(df2.values, title='df2.all'+Bname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(h, title='h.all'+Aname+'.pred', xlab='genes', ylab='cells', vmax=max, vmin=min)

visualization_of_dfs()
sess.close()
print("Finished gene", j)

