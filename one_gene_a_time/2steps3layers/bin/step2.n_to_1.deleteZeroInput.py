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

# read data #
data = 'EMT9k_log_msk90'  # EMT2730 or splatter

if data is 'splatter':  # only this mode creates gene-gene plot
    file = "../data/v1-1-5-3/v1-1-5-3.F3.msk.hd5" #data need imputation
    file_benchmark = "../data/v1-1-5-3/v1-1-5-3.F3.hd5"
    Aname = '(F3.msk)'
    Bname = '(F3)'
    df = pd.read_hdf(file).transpose() #[cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT2730':
    file = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5" #data need imputation
    file_benchmark = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5"
    Aname = '(EMT2730)'
    Bname = '(EMT2730)'
    df = pd.read_hdf(file).transpose() #[cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose() #[cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.hd5"
    Aname = '(EMT9k.B.msk)'
    Bname = '(EMT9k.B)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k_log_msk50':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk50.log.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"
    Aname = '(EMT9kLog_Bmsk50)'
    Bname = '(EMT9kLog_B)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
elif data is 'EMT9k_log_msk90':  # magic imputation using 8.7k cells > 300 reads/cell
    file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk90.log.hd5"  # data need imputation
    file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"
    Aname = '(EMT9kLog_Bmsk90)'
    Bname = '(EMT9kLog_B)'
    df = pd.read_hdf(file).transpose()  # [cells,genes]
    print("input_array:\n", df.values[0:4, 0:4], "\n")
    df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    m, n = df.shape  # m: n_cells; n: n_genes
else:
    raise Warning("data name not recognized!")


# Parameters #
print("this is just testing version, superfast and bad")
j_lst = [0, 1, 200, 201, 400, 401, 600, 601, 800, 801]  # todo
j_lst = [0, 1, 800]  # todo
j_lst = range(n)
# j = 400
# print("\n\n>>> for gene", j)
learning_rate = 0.01  # todo: was 0.002 for SGD
training_epochs = 1000  # todo: 10000 for show, 1600 for early stop
batch_size = 128  # todo: can be too large if solid cells < 256
sd = 0.0001 #stddev for random init
n_input = n
n_hidden_1 = 200  # for magic data, only < 100 dims in PCA
pIn = 0.8
pHidden = 0.5
# log_dir = './re_train' + '_j' + str(j)
# scimpute.refresh_logfolder(log_dir)
display_step = 10  # todo: change to 100
snapshot_step = 5000

# loop over j_lst, init focusFnn w/b, keep encoder w/b same
for j in j_lst:

    keep_prob_input = tf.placeholder(tf.float32)
    keep_prob_hidden = tf.placeholder(tf.float32)

    def encoder(x):
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


    def focusFnn(x):
        with tf.name_scope("Decoder"):
            # Encoder Hidden layer with sigmoid activation #1
            x_drop = tf.nn.dropout(x, keep_prob_hidden)
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x_drop, focusFnn_params['w1']),
                                        focusFnn_params['b1']))
            variable_summaries('fnn_w1', focusFnn_params['w1'])
            variable_summaries('fnn_b1', focusFnn_params['b1'])
            variable_summaries('fnn_a1', layer_1)
        return layer_1


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
        cost_train = sess.run(cost, feed_dict={X: df_train.values,
                                               keep_prob_input: 1, keep_prob_hidden: 1})
        cost_valid = sess.run(cost, feed_dict={X: df_valid.values,
                                               keep_prob_input: 1, keep_prob_hidden: 1})
        print("\nEpoch 0: cost_train=", round(cost_train, 3), "cost_valid=", round(cost_valid, 3))
        h_input = sess.run(y_pred, feed_dict={X: df.values,
                                              keep_prob_input: 1, keep_prob_hidden: 1})
        print('corr', pearsonr(h_input, df.values[:, j:j + 1]))
        # print("prediction:\n", h_input, "\ntruth:\n", df2.values[:,j:j+1])


    def snapshot():
        print("#Snapshot: ")
        h_input = sess.run(y_pred, feed_dict={X: df.values,
                                              keep_prob_input: 1, keep_prob_hidden: 1})
        print('corr', pearsonr(h_input, df.values[:, j:j + 1]))
        # print("prediction:\n", h_input, "\ntruth:\n", df2.values[:,j:j+1])
        df_h_input = pd.DataFrame(data=h_input, columns=df.columns[j:j + 1], index=df.index)
        scimpute.save_hd5(df_h_input, log_dir + "/imputation.step2.hd5")
        # save model
        save_path = saver.save(sess, log_dir + "/step2.ckpt")
        print("Model saved in: %s" % save_path)


    def print_parameters():
        print(os.getcwd(), "\n",
              "\n# Hyper parameters:",
              "\nn_features: ", n,
              "\nn_hidden1: ", n_hidden_1,
              "\nlearning_rate :", learning_rate,
              "\nbatch_size: ", batch_size,
              "\nepoches: ", training_epochs, "\n",
              "\ndf_train.shape", df_train.values.shape,
              "\ndf_valid.shape", df_valid.values.shape,
              "\ndf_train_solid.shape", df_train_solid.values.shape,
              "\ndf_valid_solid.shape", df_valid_solid.values.shape,
              "\npIn", pIn,
              "\npHidden", pHidden,
              # "\ndf2_train.shape", df2_train.values.shape,
              # "\ndf2_valid.shape", df2_valid.values.shape,
              # "\ndf2_train_solid.shape", df2_train_solid.shape,
              "\ndf2_valid_solid.values.shape", df2_valid_solid.values.shape
              )


    # Define model #
    X = tf.placeholder(tf.float32, [None, n_input])  # input
    M = tf.placeholder(tf.float32, [None, n_input])  # benchmark
    encoder_params = {
        'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=sd), name='encoder_w1'),
        'b1': tf.Variable(tf.random_normal([n_hidden_1], mean=30 * sd, stddev=sd), name='encoder_b1')
    }

    # Launch Session #
    sess = tf.Session()
    # restore encoder w/b, frozen
    saver = tf.train.Saver()
    saver.restore(sess, "./pre_train/step1.ckpt")

    # define fucusFnn_params only after restore, or error, because it's not in ckpt
    tf.set_random_seed(4)  # seed
    focusFnn_params = {
        'w1': tf.Variable(tf.random_normal([n_hidden_1, 1], stddev=sd), name='focusFnn_w1'),
        'b1': tf.Variable(tf.random_normal([1], mean=30 * sd, stddev=sd), name='focusFnn_b1')
    }
    init_focusFnn = tf.variables_initializer(list(focusFnn_params.values()))



    print("\n>>> for gene", j)
    # prep #
    corr_log_train = []
    corr_log_valid = []
    mse_train = []
    mse_valid = []
    mse_bench_train = []
    mse_bench_valid = []
    epoch_log = []
    # log_dir = './re_train' + '_j' + str(j)
    log_dir = './re_train_j'
    scimpute.refresh_logfolder(log_dir)

    # rand split data
    [df_train, df_valid, df_test] = scimpute.split_df(df)  # always same seed
    # filter data
    solid_row_id_train = (df_train.ix[:, j:j + 1] > 0).values
    df_train_solid = df_train[solid_row_id_train]
    solid_row_id_valid = (df_valid.ix[:, j:j + 1] > 0).values
    df_valid_solid = df_valid[solid_row_id_valid]
    solid_row_id_test = (df_test.ix[:, j:j + 1] > 0).values
    df_test_solid = df_test[solid_row_id_test]
    # df2 benchmark
    df2_train_solid = df2.ix[df_train_solid.index]
    df2_valid_solid = df2.ix[df_valid_solid.index]
    df2_test_solid = df2.ix[df_test_solid.index]
    df2_train = df2.ix[df_train.index]
    df2_valid = df2.ix[df_valid.index]
    df2_test = df2.ix[df_test.index]
    print_parameters()

    # work #
    sess.run(init_focusFnn)
    focusFnn_b1 = sess.run(focusFnn_params['b1'])  # same init series each run, diff init each j

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

            def epoch_summary():
                run_metadata = tf.RunMetadata()
                train_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)

                # Summary
                merged = tf.summary.merge_all()

                [summary_train, cost_train, cost_train_m] = sess.run([merged, cost, cost_benchmark],
                                                       feed_dict={X: df_train_solid.values, M: df2_train_solid.values,
                                                                  keep_prob_input: 1, keep_prob_hidden: 1})
                [summary_valid, cost_valid, cost_valid_m] = sess.run([merged, cost, cost_benchmark],
                                                       feed_dict={X: df_valid_solid.values, M: df2_valid_solid.values,
                                                                  keep_prob_input: 1, keep_prob_hidden: 1})
                mse_bench_train.append(cost_train_m)
                mse_bench_valid.append(cost_valid_m)
                mse_train.append(cost_train)
                mse_valid.append(cost_valid)
                train_writer.add_summary(summary_train, epoch)
                valid_writer.add_summary(summary_valid, epoch)

                print("cost_batch=", "{:.6f}".format(cost_batch),
                      "cost_train=", "{:.6f}".format(cost_train),
                      "cost_valid=", "{:.6f}".format(cost_valid))

            epoch_summary()

            # log corr
            h_train_j = sess.run(y_pred, feed_dict={X: df_train.values,
                                                    keep_prob_input: 1, keep_prob_hidden: 1})
            h_valid_j = sess.run(y_pred, feed_dict={X: df_valid.values,
                                                    keep_prob_input: 1, keep_prob_hidden: 1})
            # print("prediction_train:\n", h_train[0:5,:], "\ntruth_train:\n", df2_train_solid.values[0:5, j:j + 1])
            # print("prediction_valid:\n", h_valid[0:5,:], "\ntruth_valid:\n", df2_valid_solid.values[0:5, j:j + 1])
            corr_train = scimpute.corr_one_gene(df2_train.values[:,j:j+1], h_train_j)
            corr_valid = scimpute.corr_one_gene(df2_valid.values[:,j:j+1], h_valid_j)
            corr_log_valid.append(corr_valid)
            corr_log_train.append(corr_train)
            epoch_log.append(epoch)
            print("corr_train: ", corr_train, "\n",
                  "corr_valid: ", corr_valid, "\n"
                  )


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
    h_valid_j = sess.run(y_pred, feed_dict={X: df_valid.values,
                                            keep_prob_input: 1, keep_prob_hidden: 1})
    h_valid_solid_j = sess.run(y_pred, feed_dict={X: df_valid_solid.values,
                                                  keep_prob_input: 1, keep_prob_hidden: 1})
    h_j = sess.run(y_pred, feed_dict={X: df.values,
                                      keep_prob_input: 1, keep_prob_hidden: 1})
    try:
        H
    except NameError:
        # print('H not defined')
        H = h_j
        H_valid = h_valid_j
    else:
        # print('H is defined')
        H = np.column_stack((H, h_j))
        H_valid = np.column_stack((H_valid, h_valid_j))

    print('H:', H.shape, H)
    time.sleep(1)


    # code_neck_valid_solid = sess.run(encoder_op, feed_dict={X: df_valid_solid.values})
    # code_neck_valid = sess.run(encoder_op, feed_dict={X: df_valid.values})

    # learning curve for gene-j
    scimpute.curveplot2(epoch_log, corr_log_train, corr_log_valid,
                         title='learning_curve_pearsonr_bench.step2.gene'+str(j),
                         xlabel='epoch'
                                + "\nvalid:"+str(corr_log_valid[-1]),
                         ylabel='Pearson corr (predction vs ground truth)')
    scimpute.curveplot2(epoch_log, mse_bench_train, mse_bench_valid,
                         title='learning_curve_MSE_bench.step2.gene'+str(j),
                         xlabel='epoch'
                                + "\nvalid:" + str(mse_bench_valid[-1]),
                        ylabel='MSE (predction vs ground truth)')
    scimpute.curveplot2(epoch_log, mse_train, mse_valid,
                         title='learning_curve_MSE_input.step2.gene'+str(j),
                         xlabel='epoch'
                               + "\nvalid:" + str(mse_valid[-1]),
                         ylabel='MSE (predction vs input)')

    # gene-correlation for gene-j
    scimpute.scatterplot2(df2_valid.values[:, j], h_valid_j[:,0],
                          title=str('gene-' + str(j) + ', valid, step2'),
                          xlabel='Ground Truth ' + Bname,
                          ylabel='Prediction ' + Aname
                          )

    # todo: better vis of focusFnn
    focusFnn_w1 = sess.run(focusFnn_params['w1'])  #500, 1
    focusFnn_b1 = sess.run(focusFnn_params['b1'])  #1
    focusFnn_b1 = focusFnn_b1.reshape(len(focusFnn_b1), 1)
    focusFnn_b1_T = focusFnn_b1.T
    scimpute.visualize_weights_biases(focusFnn_w1, focusFnn_b1_T, 'focusFnn_w1, b1, '+'gene-'+str(j))
    # scimpute.heatmap_vis(code_neck_valid, title='code_neck_valid, all cells' + Bname, xlab='', ylab='', vmax=max, vmin=min)

    print("<<< Finished gene", j)

    # visualization of weights
    encoder_w1 = sess.run(encoder_params['w1'])  # 1000, 500
    encoder_b1 = sess.run(encoder_params['b1'])  # 500, (do T)
    encoder_b1 = encoder_b1.reshape(len(encoder_b1), 1)
    encoder_b1_T = encoder_b1.T
    scimpute.visualize_weights_biases(encoder_w1, encoder_b1_T, 'encoder_w1, b1')

    sess.close()
    tf.reset_default_graph()


# out loop #
# only after all genes processed, do that dim match (df, H[:,j_lst])
H_df = pd.DataFrame(data=H, columns=df.ix[:, j_lst].columns, index=df.ix[:, j_lst].index)
scimpute.save_hd5(H_df, "./plots/imputation.step2.hd5")
H_valid_df = pd.DataFrame(data=H_valid, columns=df_valid.ix[:, j_lst].columns, index=df_valid.ix[:, j_lst].index)
scimpute.save_hd5(H_valid_df, "./plots/imputation.step2.valid.hd5")
scimpute.save_hd5(df_valid, "./plots/df_valid.hd5")
scimpute.save_hd5(df2_valid, "./plots/df2_valid.hd5")

# vis df
# Get same subset of genes(j_lst)/cells(valid set)
def subset_df (df_big, df_subset):
    return (df_big.ix[df_subset.index, df_subset.columns])
df_jlst = subset_df(df, H_df)
df2_jlst = subset_df(df2, H_df)
df_valid_jlst = subset_df(df, H_valid_df)
df2_valid_jlst = subset_df(df2, H_valid_df)

# matrix MSE
Matrix_MSE_Valid = np.mean(
    np.power((df_valid_jlst.values - H_valid_df.values), 2)
    )  # todo: not finished
print("Matrix_MSE_valid: ", Matrix_MSE_Valid)

Matrix_MSE_all = np.mean(
    np.power((df_jlst.values - H_df.values), 2)
    )  # todo: not finished
print("Matrix_MSE: ", Matrix_MSE_all)

def visualization_of_dfs():
    max, min = scimpute.max_min_element_in_arrs([df_valid.values, df.values,  # full data frame
                                                 df2_valid.values, df2.values,
                                                 df_valid_jlst.values, df_jlst.values,  # same dim with H
                                                 df2_valid_jlst.values, df2_jlst.values,
                                                 H_df.values, H_valid_df.values])
    # # df
    # scimpute.heatmap_vis(df_valid.values, title='df.valid' + Aname,
    #                      xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(df.values, title='df.all' + Aname,
    #                      xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(df2_valid.values, title='df2.valid' + Bname,
    #                      xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(df2.values, title='df2.all' + Bname,
    #                      xlab='genes', ylab='cells', vmax=max, vmin=min)
    # df_jlst
    scimpute.heatmap_vis(df_valid_jlst.values, title='DF_jlst.valid' + Aname,
                         xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(df_jlst.values, title='DF_jlst.all' + Aname,
                         xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(df2_valid_jlst.values, title='DF2_jlst.valid' + Bname,
                         xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(df2_jlst.values, title='DF2_jlst.all' + Bname,
                         xlab='genes', ylab='cells', vmax=max, vmin=min)
    # H
    scimpute.heatmap_vis(H_valid_df, title='h.valid' + Aname + '.pred',
                         xlab='genes\n' + "MatrixMSE_Valid" + str(round(Matrix_MSE_Valid, 4)),
                         ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(H_df, title='h.all' + Aname + '.pred',
                         xlab='genes\n' + "MatrixMSE_All" + str(round(Matrix_MSE_all, 4)) ,
                         ylab='cells', vmax=max, vmin=min)

visualization_of_dfs()

# corr_heatmap
def corrcoef_matrix_vis (df, title='xxx.imputation.corr_gene_wise'):
    corrcoef_matrix_gene_wise = np.corrcoef(df, rowvar=False)
    scimpute.hist_arr_flat(corrcoef_matrix_gene_wise,
                           title=title+"hist.png")
    scimpute.heatmap_vis(corrcoef_matrix_gene_wise,
                         title=title+".heatmap.png", vmin=-1, vmax=1)

corrcoef_matrix_vis(df_jlst, title="DF.corr_gene_wise")
corrcoef_matrix_vis(df2_jlst, title="DF2.corr_gene_wise")
corrcoef_matrix_vis(H_df, title="step2(focusFnn).imputation.corr_gene_wise")



corrcoef_matrix_vis(df_valid_jlst, title="DF.valid.corr_gene_wise")
corrcoef_matrix_vis(df2_valid_jlst, title="DF2.valid.corr_gene_wise")
corrcoef_matrix_vis(H_valid_df, title="step2(focusFnn).valid.imputation.corr_gene_wise")


# Gene-Gene relationships #
list = [[0, 1],
        [200, 201],
        [400, 401],
        [600, 601],
        [800, 801],
        [200, 800]
        ]  # todo: this list only validated for splatter dataset E/F


# GroundTruth
for i, j in list:
    scimpute.scatterplot2(df2.ix[:, i], df2.ix[:, j],
                          title="Gene" + str(i + 1) + 'vs Gene' + str(j + 1) + 'in ' + Bname,
                          xlabel='Gene' + str(i + 1), ylabel='Gene' + str(j + 1))
# Input
for i, j in list:
    scimpute.scatterplot2(df.ix[:, i], df.ix[:, j], title="Gene" + str(i + 1) + 'vs Gene' + str(j + 1) + 'in ' + Aname,
                          xlabel='Gene' + str(i + 1), ylabel='Gene' + str(j + 1))
# Prediction
for i, j in list:
    scimpute.scatterplot2(H_df.ix[:, i], H_df.ix[:, j], title="Gene" + str(i + 1) + 'vs Gene' + str(j + 1) + 'in ' + Aname + '.pred',
                          xlabel='Gene' + str(i + 1), ylabel='Gene' + str(j + 1))