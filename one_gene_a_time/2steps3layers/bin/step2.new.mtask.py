#!/usr/bin/python

#  todo:
# 1. [x] restore (TL)
# 2. [x] use functions in model
# 3. [x] mtask (one gene a time with one network)
# 4. [-] exclude zeros or not
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


def evaluate_epoch_step2():
    print("> Evaluation:")
    epoch_log.append(epoch)
    # mse2 (h vs M)
    mse2_train = sess.run(mse2, feed_dict={X: df_train, M: df2_train,
                                           pIn_holder: 1, pHidden_holder: 1})
    mse2_valid = sess.run(mse2, feed_dict={X: df_valid, M: df2_valid,
                                           pIn_holder: 1, pHidden_holder: 1})
    mse2_batch_vec.append(mse2_train)  # approximation
    mse2_valid_vec.append(mse2_valid)
    print("mse2_train=", round(mse2_train, 3), "mse2_valid=", round(mse2_valid, 3))

    # mse1 (h vs X)
    mse1_train = sess.run(mse1, feed_dict={X: df_train, M: df2_train,
                                           pHidden_holder: 1.0, pIn_holder: 1.0})
    mse1_batch_vec.append(mse1_train)

    mse1_valid = sess.run(mse1, feed_dict={X: df_valid, M: df2_valid,
                                           pHidden_holder: 1.0, pIn_holder: 1.0})
    mse1_valid_vec.append(mse1_valid)
    print("mse1_train=", round(mse1_train, 3), "mse1_valid=", round(mse1_valid, 3))

    h_train = sess.run(h, feed_dict={X: df_train.values, pIn_holder: 1, pHidden_holder: 1})
    h_valid = sess.run(h, feed_dict={X: df_valid.values, pIn_holder: 1, pHidden_holder: 1})
    corr2_train = scimpute.medium_corr(df2_train.values, h_train)
    corr2_valid = scimpute.medium_corr(df2_valid.values, h_valid)
    medium_cell_corr2_batch_vec.append(corr2_train)
    medium_cell_corr2_valid_vec.append(corr2_valid)
    print("Medium-cell-corr2, train, valid:", corr2_train, corr2_valid)
    # # tb todo
    # merged_summary = tf.summary.merge_all()
    # summary_batch = sess.run(merged_summary, feed_dict={X: df_train, M: df2_train,  # M is not used here, just dummy
    #                                                     pIn_holder: 1.0, pHidden_holder: 1.0})
    # summary_valid = sess.run(merged_summary, feed_dict={X: df_valid.values, M: df2_valid.values,
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
    summary_valid = sess.run(merged_summary, feed_dict={X: df_valid.values, M: df2_valid.values,
                                                        pIn_holder: 1.0, pHidden_holder: 1.0})
    batch_writer.add_summary(summary_batch, epoch)
    valid_writer.add_summary(summary_valid, epoch)
    toc = time.time()
    print('tb_summary time:', round(toc-tic,2))


def learning_curve_step2():
    print('> plotting learning curves')
    scimpute.learning_curve_mse(epoch_log, mse2_batch_vec, mse2_valid_vec,
                                title="Learning Curve MSE2 (Pred vs GroundTruth)",
                                ylabel='MSE2')
    scimpute.learning_curve_mse(epoch_log, mse1_batch_vec, mse1_valid_vec,
                                title="Learning Curve MSE1 (Pred vs Input)",
                                ylabel="MSE1")
    # scimpute.learning_curve_corr(epoch_log, medium_cell_corr2_batch_vec, medium_cell_corr2_valid_vec)


def snapshot():
    print("> Snapshot (save inference, save session, calculate whole dataset cell-pearsonr ): ")
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
    scimpute.save_hd5(df_h_input, log_dir + "/imputation.step2.hd5")
    # save model
    save_path = saver.save(sess, log_dir + "/step2.ckpt")
    print("Model saved in: %s" % save_path)
    return (h_train, h_valid, h_input)


def save_bottle_neck_representation():
    print("> save bottle-neck_representation")
    code_bottle_neck_input = sess.run(e_a1, feed_dict={X: df.values,
                                                       pIn_holder: 1, pHidden_holder: 1})  # change with layer
    np.save('pre_train/code_neck_valid.npy', code_bottle_neck_input)
    clustermap = sns.clustermap(code_bottle_neck_input)
    clustermap.savefig('./plots/bottle_neck.hclust.png')


def groundTruth_vs_prediction():
    print("> Ground truth vs prediction")
    for j in [2, 3, 205, 206, 4058, 7496, 8495, 12871]:  # Cd34, Gypa, Klf1, Sfpi1
            scimpute.scatterplot2(df2_valid.values[:, j], h_valid[:, j], range='same',
                                  title=str('scatterplot1, gene-' + str(j) + ', valid, step1'),
                                  xlabel='Ground Truth ' + Bname,
                                  ylabel='Prediction ' + Aname
                                  )
            scimpute.scatterplot2(df2_valid.values[:, j], h_valid[:, j], range='flexible',
                                      title=str('scatterplot2, gene-' + str(j) + ', valid, step1'),
                                      xlabel='Ground Truth ' + Bname,
                                      ylabel='Prediction ' + Aname
                                      )


def gene_gene_relationship():
    print('> gene-gene relationship before/after inference')
    List = [[2, 3],
            [205, 206],
            [4058, 7496],
            [8495, 12871]
            ]
    # Valid set: Prediction
    for i, j in List:
        scimpute.scatterplot2(h_valid[:, i], h_valid[:, j],
                              title="Gene" + str(i) + 'vs Gene' + str(j) + '.in ' + Aname + '.pred.valid',
                              xlabel='Gene' + str(i) + '.valid', ylabel='Gene' + str(j + 1))
    # Valid set: GroundTruth
    for i, j in List:
        scimpute.scatterplot2(df2_valid.ix[:, i], df2_valid.ix[:, j],
                              title="Gene" + str(i) + 'vs Gene' + str(j) + '.in ' + Bname + '.GroundTruth.valid',
                              xlabel='Gene' + str(i) + '.valid', ylabel='Gene' + str(j))
    # Input set: Prediction
    for i, j in List:
        scimpute.scatterplot2(h_train[:, i], h_train[:, j],
                              title="Gene" + str(i) + 'vs Gene' + str(j) + '.in ' + Aname + '.pred.input',
                              xlabel='Gene' + str(i) + '.input', ylabel='Gene' + str(j + 1))

    # Input set: GroundTruth
    for i, j in List:
        scimpute.scatterplot2(df2.ix[:, i], df2.ix[:, j],
                              title="Gene" + str(i) + 'vs Gene' + str(j) + '.in ' + Bname + '.GroundTruth.input',
                              xlabel='Gene' + str(i) + '.input', ylabel='Gene' + str(j))


def weights_visualization(w_name, b_name):
    print('visualization of weights/biases for each layer')
    w = eval(w_name)
    b = eval(b_name)
    w_arr = sess.run(w)
    b_arr = sess.run(b)
    b_arr = b_arr.reshape(len(b_arr), 1)
    b_arr_T = b_arr.T
    scimpute.visualize_weights_biases(w_arr, b_arr_T, w_name + ',' + b_name)  # todo: update name (low priority)


def visualize_weights():
    # todo: update when model changes depth
    weights_visualization('e_w1', 'e_b1')
    weights_visualization('d_w1', 'd_b1')
    # weights_visualization('e_w2', 'e_b2')
    # weights_visualization('d_w2', 'd_b2')
    # weights_visualization('e_w3', 'e_b3')
    # weights_visualization('d_w3', 'd_b3')
    # weights_visualization('e_w4', 'e_b4')
    # weights_visualization('d_w4', 'd_b4')


def save_weights():
    # todo: update when model changes depth
    print('save weights in csv')
    np.save('pre_train/e_w1', sess.run(e_w1))
    np.save('pre_train/d_w1', sess.run(d_w1))
    # np.save('pre_train/e_w2', sess.run(e_w2))
    # np.save('pre_train/d_w2', sess.run(d_w2))
    # np.save('pre_train/e_w3', sess.run(e_w3))
    # np.save('pre_train/d_w3', sess.run(d_w3))
    # np.save('pre_train/e_w4', sess.run(e_w4))
    # np.save('pre_train/d_w4', sess.run(d_w4))
    # scimpute.save_csv(sess.run(d_w2), 'pre_train/d_w2.csv.gz')


def visualization_of_dfs():
    print('visualization of dfs')
    max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid])
    # max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid, h, df.values])
    scimpute.heatmap_vis(df_valid.values, title='df.valid'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(h_valid, title='h.valid'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(df.values, title='df'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(h, title='h'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)

# print versions / sys.path
print ('python version:', sys.version)
print('tf.__version__', tf.__version__)
print('sys.path', sys.path)

# refresh folder
log_dir = './re_train'
scimpute.refresh_logfolder(log_dir)

# read data
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

# split data and save indexes
df_train, df_valid, df_test = scimpute.split_df(df, a=0.7, b=0.15, c=0.15)
df2_train, df2_valid, df2_test = df2.ix[df_train.index], df2.ix[df_valid.index], df2.ix[df_test.index]
df_train.to_csv('re_train/df_train.step2_index.csv', columns=[], header=False)  # save index for future use
df_valid.to_csv('re_train/df_valid.step2_index.csv', columns=[], header=False)
df_test.to_csv('re_train/df_test.step2_index.csv', columns=[], header=False)


# parameters
n_input = n
n_hidden_1 = 200  # change with layer
n_hidden_2 = 600
n_hidden_3 = 400
n_hidden_4 = 200
pIn = 0.8
pHidden = 0.5
learning_rate = 0.0003  # 0.0003 for 3-7L, 0.00003 for 9L # change with layer
sd = 0.0001  # 3-7L:1e-3, 9L:1e-4
batch_size = 1803
training_epochs = 100000  #3L:100, 5L:1000, 7L:1000, 9L:3000
display_step = 20
snapshot_step = 1000
# print_parameters()
j_lst = [4058, 7496, 8495, 12871]  # Cd34, Gypa, Klf1, Sfpi1

# Start model
tf.reset_default_graph()

# define placeholders and variables
X = tf.placeholder(tf.float32, [None, n_input], name='X_input')  # input
M = tf.placeholder(tf.float32, [None, n_input], name='M_ground_truth')  # benchmark
pIn_holder = tf.placeholder(tf.float32, name='pIn')
pHidden_holder = tf.placeholder(tf.float32, name='pHidden')
J = tf.placeholder(tf.int32, name='j')

# define layers and variables
tf.set_random_seed(3)  # seed
# change with layer
with tf.name_scope('Encoder_L1'):
    e_w1, e_b1 = scimpute.weight_bias_variable('encoder1', n, n_hidden_1, sd)
    e_a1 = scimpute.dense_layer('encoder1', X, e_w1, e_b1, pIn_holder)
# with tf.name_scope('Encoder_L2'):
#     e_w2, e_b2 = scimpute.weight_bias_variable('encoder2', n_hidden_1, n_hidden_2, sd)
#     e_a2 = scimpute.dense_layer('encoder2', e_a1, e_w2, e_b2, pHidden_holder)
# with tf.name_scope('Encoder_L3'):
#     e_w3, e_b3 = scimpute.weight_bias_variable('encoder3', n_hidden_2, n_hidden_3, sd)
#     e_a3 = scimpute.dense_layer('encoder3', e_a2, e_w3, e_b3, pHidden_holder)
# with tf.name_scope('Encoder_L4'):
#     e_w4, e_b4 = scimpute.weight_bias_variable('encoder4', n_hidden_3, n_hidden_4, sd)
#     e_a4 = scimpute.dense_layer('encoder4', e_a3, e_w4, e_b4, pHidden_holder)
# with tf.name_scope('Decoder_L4'):
#     d_w4, d_b4 = scimpute.weight_bias_variable('decoder4', n_hidden_4, n_hidden_3, sd)
#     d_a4 = scimpute.dense_layer('decoder4', e_a4, d_w4, d_b4, pHidden_holder)
# with tf.name_scope('Decoder_L3'):
#     d_w3, d_b3 = scimpute.weight_bias_variable('decoder3', n_hidden_3, n_hidden_2, sd)
#     d_a3 = scimpute.dense_layer('decoder3', d_a4, d_w3, d_b3, pHidden_holder)
# with tf.name_scope('Decoder_L2'):
#     d_w2, d_b2 = scimpute.weight_bias_variable('decoder2', n_hidden_2, n_hidden_1, sd)
#     d_a2 = scimpute.dense_layer('decoder2', d_a3, d_w2, d_b2, pHidden_holder)
with tf.name_scope('Decoder_L1'):
    d_w1, d_b1 = scimpute.weight_bias_variable('decoder1', n_hidden_1, n, sd)
    d_a1 = scimpute.dense_layer('decoder1', e_a1, d_w1, d_b1, pHidden_holder)  # todo: change input activations if model changed

# define input/output
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
optimizer = tf.train.AdamOptimizer(learning_rate)
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
num_batch = int(math.floor(len(df_train) // batch_size))  # floor
epoch_log = []
mse1_batch_vec, mse1_valid_vec = [], []  # mse1 = MSE(X, h)
mse2_batch_vec, mse2_valid_vec = [], []  # mse2 = MSE(M, h)
medium_cell_corr1_batch_vec, medium_cell_corr1_valid_vec = [], []  # medium_cell_corr, for subset of cells
medium_cell_corr2_batch_vec, medium_cell_corr2_valid_vec = [], []  # 1 ~ (X, h); 2 ~ (M, h)

# evaluate epoch0
evaluate_epoch_step2()

# Outer loop (epochs)
for epoch in range(1, training_epochs+1):
    # training on non-zero(nz) cells for gene-j #
    tic_cpu, tic_wall = time.clock(), time.time()
    # rand mini-batch
    random_indices = np.random.choice(len(df_train), batch_size, replace=False)
    # inner loop (mini-batches)
    for i in range(num_batch):
        # x_batch
        indices = np.arange(batch_size * i, batch_size*(i+1))
        x_batch = df_train.ix[indices, :]  # [bs, n]
        # j_batch
        j = np.random.choice(range(n_input), 1)[0]  # todo: change to jlist for test, range(n_input) for real usage
        # solid data
        nz_indices = (x_batch.ix[:, j:j + 1] > 0).values  #todo: get nz first then choose mini-batch
        x_batch_nz = x_batch[nz_indices]

        sess.run(trainer, feed_dict={X: x_batch_nz,
                                     pIn_holder: pIn, pHidden_holder: pHidden,
                                     J: j})
    toc_cpu, toc_wall = time.clock(), time.time()


    # report per epoch #
    if (epoch == 1) or (epoch % display_step == 0):
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

        mse2_valid = sess.run(mse2, feed_dict={X: df_valid, M: df2_valid,
                                               pHidden_holder: 1.0, pIn_holder: 1.0})
        mse2_valid_vec.append(mse2_valid)
        print('mse2_valid:', mse2_valid)

        # log mse1 (h vs X)
        mse1_batch = sess.run(mse1, feed_dict={X: x_batch, M: df2.ix[x_batch.index],
                                               pHidden_holder: 1.0, pIn_holder: 1.0})
        mse1_batch_vec.append(mse1_batch)
        print('mse1_batch:', mse1_batch)

        mse1_valid = sess.run(mse1, feed_dict={X: df_valid, M: df2_valid,
                                               pHidden_holder: 1.0, pIn_holder: 1.0})
        mse1_valid_vec.append(mse1_valid)
        print('mse1_valid:', mse1_valid)

        toc_log = time.time()
        epoch_log.append(epoch)
        print('log time for each epoch:', round(toc_log - tic_log, 1))

    # report and save sess per observation interval
    if (epoch % snapshot_step == 0) or (epoch == training_epochs):
        tic_log2 = time.time()
        learning_curve_step2()
        h_train, h_valid, h_input = snapshot()  # inference, save sess, save imputation.hd5


evaluate_epoch_step2()  # careful, this append new mse to mse_vec


batch_writer.close()
valid_writer.close()
sess.close()
print("Finished!")

