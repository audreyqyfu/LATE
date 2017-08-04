# load pre-trained NN, transfer learning
# https://www.tensorflow.org/get_started/summaries_and_tensorboard
# 07/25/2017
# 07/31/2017 One Gene a time
# from __future__ import division #fix division // get float bug
# from __future__ import print_function #fix printing \n

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

print('tf.__version__', tf.__version__)
print ('python version:', sys.version)

def split_arr(arr, a=0.8, b=0.1, c=0.1):
    """input array, output rand split arrays
    a: train, b: valid, c: test
    e.g.: [arr_train, arr_valid, arr_test] = split(df.values)"""
    np.random.seed(1) # for splitting consistency
    train_indices = np.random.choice(arr.shape[0], int(arr.shape[0] * a/(a+b+c)), replace=False)
    remain_indices = np.array(list(set(range(arr.shape[0])) - set(train_indices)))
    valid_indices = np.random.choice(remain_indices, int(len(remain_indices) * b/(b+c)), replace=False)
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
    train_indices = np.random.choice(df.shape[0], int(df.shape[0] * a/(a+b+c)), replace=False)
    remain_indices = np.array(list(set(range(df.shape[0])) - set(train_indices)))
    valid_indices = np.random.choice(remain_indices, int(len(remain_indices) * b/(b+c)), replace=False)
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
    result = round(pearsonrlog[int(num/2)][0], accuracy)
    return(result)

def corr_one_gene(col1, col2, accuracy = 3):
    """will calculate pearsonr for gene(i)"""
    # from scipy.stats.stats import pearsonr
    result = pearsonr(col1, col2)[0]
    # result = round(result, accuracy)
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

# read data #
file = "../../data/splat_v1-1-2_norm_log/splat.OneGroup.norm.log.B.hd5" #data need imputation
file_benchmark = "../../data/splat_v1-1-2_norm_log/splat.OneGroup.norm.log.B.hd5" #data need imputation
df = pd.read_hdf(file).transpose() #[cells,genes]
df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
m, n = df.shape  # m: n_cells; n: n_genes

# rand split data
[df_train, df_valid, df_test] = split_df(df)

df2_train = df2.ix[df_train.index]
df2_valid = df2.ix[df_valid.index]
df2_test = df2.ix[df_test.index]

# save real data for comparison
# save_hd5(df_train, 'df_train.hd5')

# Parameters #
learning_rate = 0.0001
training_epochs = 1000
batch_size = 256
sd = 0.01 #stddev for random init

display_step = 1
snapshot_step = 2500

# Network Parameters #
n_input = n
n_hidden_1 = 500
n_hidden_2 = 250
log_dir = './re_train'

# refresh tensorboard folder
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
    print (log_dir, "deleted")
tf.gfile.MakeDirs(log_dir)
print(log_dir, 'created')
time.sleep(1)

# LOG
print(os.getcwd(),"\n",
    "\n# Hyper parameters:",
    "\nn_features: ", n,
    "\nn_hidden1: ", n_hidden_1,
    "\nn_hidden2: ", n_hidden_2,
    "\nlearning_rate :", learning_rate,
    "\nbatch_size: ", batch_size,
    "\nepoches: ", training_epochs, "\n",
    "\ndf_train.values.shape", df_train.values.shape,
    "\ndf_valid.values.shape", df_valid.values.shape,
    "\ndf2_train.shape", df2_train.shape,
    "\ndf2_valid.values.shape", df2_valid.values.shape,
    "\n")
print("input_array:\n", df.values[0:4,0:4], "\n")

corr_log = []
epoch_log = []

# regression #
j=1 #can loop in range(n)
print('gene index: ', j)

X = tf.placeholder(tf.float32, [None, n_input])  # input
M = tf.placeholder(tf.float32, [None, n_input])  # benchmark
# Y = tf.placeholder(tf.float32, [None, 1]) # for a gene

weights = {
    'encoder_w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev= sd), name='encoder_w1'),
    'encoder_w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev= sd), name='encoder_w2'),
    'decoder_w1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1], stddev= sd), name='decoder_w1'),
    'decoder_w2': tf.Variable(tf.random_normal([n_hidden_1, n_input], stddev= sd), name='decoder_w2'),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1], stddev= sd), name='encoder_b1'),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2], stddev= sd), name='encoder_b2'),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1], stddev= sd), name='decoder_b1'),
    'decoder_b2': tf.Variable(tf.random_normal([n_input], stddev= sd), name='decoder_b2'),
}

def encoder(x):
    with tf.name_scope("Encoder"):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_w1']),
                                       biases['encoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_w2']),
                                       biases['encoder_b2']))

        variable_summaries('weights_w1', weights['encoder_w1'])
        variable_summaries('weights_w2', weights['encoder_w2'])
        variable_summaries('biases_b1', biases['encoder_b1'])
        variable_summaries('biases_b2', biases['encoder_b2'])
        variable_summaries('activations_a1', layer_1)
        variable_summaries('activations_a2', layer_2)
    return layer_2

def decoder(x):
    with tf.name_scope("Decoder"):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_w1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_w2']),
                                       biases['decoder_b2']))

        variable_summaries('weights_w1', weights['decoder_w1'])
        variable_summaries('weights_w2', weights['decoder_w2'])
        variable_summaries('biases_b1', biases['decoder_b1'])
        variable_summaries('biases_b2', biases['decoder_b2'])
        variable_summaries('activations_a1', layer_1)
        variable_summaries('activations_a2', layer_2)
    return layer_2

def focusFnn(x):
    """output shape is [m, 1]"""
    with tf.name_scope("focusFnn"):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights2['fnn_w1']),
                                       biases2['fnn_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights2['fnn_w2']),
                                       biases2['fnn_b2']))
        variable_summaries('fnn_w1', weights2['fnn_w1'])
        variable_summaries('fnn_w2', weights2['fnn_w2'])
        variable_summaries('fnn_b1', biases2['fnn_b1'])
        variable_summaries('fnn_b2', biases2['fnn_b2'])
    return layer_2


# Session Start
sess = tf.Session()
# restore pre-trained parameters
saver = tf.train.Saver()
saver.restore(sess, "./pre_train/step1.ckpt")
# init new parameters
weights2 = {
    'fnn_w1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1], stddev= sd), name='fnn_w1'),
    'fnn_w2': tf.Variable(tf.random_normal([n_hidden_1, 1], stddev= sd), name='fnn_w2')
}
biases2 = {
    'fnn_b1': tf.Variable(tf.ones([n_hidden_1]), name='fnn_b1'),
    'fnn_b2': tf.Variable(tf.ones([1]), name='fnn_b2')
}
parameters2 = {**weights2, **biases2}
init_params2 = tf.variables_initializer(parameters2.values())
sess.run(init_params2)

# Construct model
encoder_op = encoder(X)
focusFnn_op = focusFnn(encoder_op)  # for one gene a time prediction
decoder_op = decoder(encoder_op)  # for pearson correlation of the whole matrix #bug (8092, 0)

# Prediction and truth
y_pred = focusFnn_op  # [m, 1]
y_true = X[:, j]
y_benchmark = M[:, j]  # benchmark for cost_fnn
M_train = df2_train.values[:, j:j+1]  # benchmark for corr
M_valid = df2_valid.values[:, j:j+1]

# Define loss and optimizer, minimize the squared error
with tf.name_scope("Metrics"):
    cost_fnn = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    cost_fnn_benchmark = tf.reduce_mean(tf.pow(y_pred- y_benchmark, 2))
    cost_decoder = tf.reduce_mean(tf.pow(X - decoder_op, 2))
    cost_decoder_benchmark = tf.reduce_mean(tf.pow(decoder_op - M, 2))
    tf.summary.scalar('cost_fnn', cost_fnn)
    tf.summary.scalar('cost_fnn_benchmark', cost_fnn_benchmark)
    tf.summary.scalar('cost_decoder', cost_decoder)
    tf.summary.scalar('cost_decoder_benchmark', cost_decoder_benchmark)

# optimizer = (
#     tf.train.GradientDescentOptimizer(learning_rate).
#     minimize(cost_fnn, var_list=[list(weights2.values()), list(biases2.values())])
# )# frozen other variables

optimizer = (
    tf.train.GradientDescentOptimizer(learning_rate).
    minimize(cost_fnn)
)# frozen other variables
print("# Updated layers: ", "fnn layers\n")

train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir+'/valid', sess.graph)
# benchmark_writer = tf.summary.FileWriter(log_dir+'/benchmark', sess.graph)

# Evaluate the init network
[cost_train, h_train] = sess.run([cost_fnn, y_pred], feed_dict={X: df_train.values})
[cost_valid, h_valid] = sess.run([cost_fnn, y_pred], feed_dict={X: df_valid.values})
print("\nEpoch 0: cost_fnn_train=", round(cost_train,3), "cost_fnn_valid=", round(cost_valid,3))
print("benchmark_pearsonr for gene ", j, " in training cells :", corr_one_gene(M_train, h_train))
print("benchmark_pearsonr for gene ", j, " in valid cells:", corr_one_gene(M_valid, h_valid))
time.sleep(2)

# Train
total_batch = int(math.floor(len(df_train)/batch_size))  # floor
# Training cycle,step2
for epoch in range(1, training_epochs+1):
    tic_cpu = time.clock(); tic_wall = time.time()
    random_indices = np.arange(len(df_train))
    np.random.shuffle(random_indices)
    for i in range(total_batch):
        indices = np.arange(batch_size*i, batch_size*(i+1))
        batch_xs = df_train.values[indices,:]
        _, cost_batch = sess.run([optimizer, cost_fnn], feed_dict={X: batch_xs})
    if (batch_size*(i+1) < len(df_train)):
        indices = np.arange(batch_size*(i+1), len(df_train))
        batch_xs = df_train.values[indices,:]
        _, cost_batch = sess.run([optimizer, cost_fnn], feed_dict={X: batch_xs})
    toc_cpu = time.clock(); toc_wall = time.time()

    # Log per epoch
    if (epoch == 1) or (epoch % display_step == 0):
        tic_log = time.time()
        print("\n#Epoch ", epoch, " took: ",
              round(toc_cpu - tic_cpu, 2), " CPU seconds; ",
              round(toc_wall - tic_wall, 2), "Wall seconds")

        run_metadata = tf.RunMetadata()
        train_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)

        h_train = sess.run(y_pred, feed_dict={X: df_train.values})
        h_valid = sess.run(y_pred, feed_dict={X: df_valid.values})

        corr_train = corr_one_gene(M_train, h_train)
        corr_valid = corr_one_gene(M_valid, h_valid)

        corr_log.append(corr_valid)
        epoch_log.append(epoch)

        #Summary
        merged = tf.summary.merge_all()

        [summary_train, cost_train] = sess.run([merged, cost_fnn], feed_dict={X: df_train.values, M: df2_train.values})
        [summary_valid, cost_valid] = sess.run([merged, cost_fnn], feed_dict={X: df_valid.values, M: df2_valid.values})
        train_writer.add_summary(summary_train, epoch)
        valid_writer.add_summary(summary_valid, epoch)


        print("cost_batch=", "{:.6f}".format(cost_batch),
              "cost_train=", "{:.6f}".format(cost_train),
              "cost_valid=", "{:.6f}".format(cost_valid))
        print("benchmark_pearsonr for gene ", j, " in training cells :", corr_train)
        print("benchmark_pearsonr for gene ", j, " in valid cells:", corr_valid)

        toc_log=time.time()
        print('log time for each display:', round(toc_log-tic_log, 1))

    # # Log per observation interval
    # if (epoch == 1) or (epoch % snapshot_step == 0) or (epoch == training_epochs):
    #     tic_log2 = time.time()
    #     print("#Snapshot: ")
    #     # save predictions
    #     h_input = sess.run(y_pred, feed_dict={X: df.values})
    #     df_h_input = pd.DataFrame(data=h_input, columns=[j], index=df.index)
    #     save_hd5(df_h_input, log_dir+"/h."+str(j)+".hd5")

    #     # save model
    #     save_path = saver.save(sess, log_dir+"/step2.ckpt")
    #     print("Model saved in: %s" % save_path)
    #     toc_log2 = time.time()
    #     print ('log2 time for observation intervals:', round(toc_log2 - tic_log2, 1))


train_writer.close()
valid_writer.close()
scatterplot(epoch_log, corr_log, 'correlation_metrics.step2', 'epoch', 'Pearson corr with ground truth')

sess.close()

print("Finished!")

