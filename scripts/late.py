#!/usr/bin/python
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
import psutil
import time
import seaborn as sns
import importlib.util
import importlib
from scipy.sparse import csr_matrix
import scimpute
import gc


def evaluate_epoch_step2():
    print("> Evaluation: epoch{}".format(epoch))
    epoch_log.append(epoch)
    mse_train, mse_nz_train = sess.run([mse, mse_nz],
                                       feed_dict={X: sample_train,
                                                  pHidden_holder: 1.0, pIn_holder: 1.0})
    mse_valid, mse_nz_valid = sess.run([mse, mse_nz],
                                       feed_dict={X: sample_valid,
                                                  pHidden_holder: 1.0, pIn_holder: 1.0})
    mse_batch_vec.append(mse_train)
    mse_valid_vec.append(mse_valid)
    mse_nz_batch_vec.append(mse_nz_train)
    mse_nz_valid_vec.append(mse_nz_valid)
    print("mse_nz_train=", round(mse_nz_train, 3), "mse_nz_valid=",
          round(mse_nz_valid, 3))
    print("mse_train=", round(mse_train, 3), "mse_valid=", round(mse_valid, 3))


def tb_summary():
    print('> Tensorboard summaries')
    tic = time.time()
    # run_metadata = tf.RunMetadata()
    # batch_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)
    merged_summary = tf.summary.merge_all()
    summary_batch = sess.run(merged_summary,
                             feed_dict={
                                 X: x_batch,
                                 pIn_holder: 1.0,
                                 pHidden_holder: 1.0})
    summary_valid = sess.run(merged_summary,
                             feed_dict={
                                 X: sample_valid,
                                 pIn_holder: 1.0,
                                 pHidden_holder: 1.0})
    batch_writer.add_summary(summary_batch, epoch)
    valid_writer.add_summary(summary_valid, epoch)
    toc = time.time()
    print('tb_summary time:', round(toc-tic, 2))


def learning_curve_mse(skip=1):
    print('> plotting learning curves')
    scimpute.learning_curve(epoch_log, mse_batch_vec, mse_valid_vec,
                                title="Learning Curve MSE.{}".format(p.stage),
                                ylabel='MSE (X vs Y, nz)',
                                dir=p.stage,
                                skip=skip
                            )
    _ = np.asarray(list(zip(epoch_log, mse_batch_vec, mse_valid_vec)))
    _ = pd.DataFrame(data=_,
                     index=epoch_log,
                     columns=['Epoch', 'MSE_batch', 'MSE_valid']
                     ).set_index('Epoch')
    _.to_csv("./{}/mse.csv".format(p.stage))


def learning_curve_mse_nz(skip=1):
    print('> plotting learning curves')
    scimpute.learning_curve(epoch_log, mse_nz_batch_vec, mse_nz_valid_vec,
                                title="Learning Curve MSE_NZ.{}".format(p.stage),
                                ylabel='MSE_NZ (X vs Y, nz)',
                                dir=p.stage,
                                skip=skip
                            )
    _ = np.asarray(list(zip(epoch_log, mse_nz_batch_vec, mse_nz_valid_vec)))
    _ = pd.DataFrame(data=_,
                     index=epoch_log,
                     columns=['Epoch', 'MSE_NZ_batch', 'MSE_NZ_valid']
                     ).set_index('Epoch')
    _.to_csv("./{}/mse_nz.csv".format(p.stage))


def save_fast_imputation():
    print("> Impute and save.. ")
    # Imputation of samples
    # Y_train_arr = sess.run(h, feed_dict={X: sample_train,
    #                                  pIn_holder: 1, pHidden_holder: 1})
    # Y_valid_arr = sess.run(h, feed_dict={X: sample_valid,
    #                                  pIn_holder: 1, pHidden_holder: 1})
    if m > p.large_size:
        Y_input_arr = sess.run(h, feed_dict={X: sample_input,
                                         pIn_holder: 1, pHidden_holder: 1})
        # save sample imputation
        Y_input_df = pd.DataFrame(data=Y_input_arr,
                                  columns=gene_ids,
                                  index=sample_input_cell_ids)
        print('RAM usage during sample imputation and saving output: ',
              '{} M'.format(usage()))
        scimpute.save_hd5(Y_input_df, "{}/sample_imputation.{}.hd5".format(p.stage,
                                                                        p.stage))
    else:
        Y_input_arr = sess.run(h, feed_dict={X: input_matrix.todense(),
                                         pIn_holder: 1, pHidden_holder: 1})
        # save sample imputation
        Y_input_df = pd.DataFrame(data=Y_input_arr,
                                  columns=gene_ids,
                                  index=cell_ids)
        print('RAM usage during whole data imputation and saving output: ',
              '{} M'.format(usage()))
        scimpute.save_hd5(Y_input_df, "{}/imputation.{}.hd5".format(p.stage,
                                                                        p.stage))


def save_whole_imputation():
    if m > p.large_size:
        # impute and save whole matrix by mini-batch
        n_out_batches = m//p.sample_size
        print('num_out_batches:', n_out_batches)
        handle2 = open('./{}/latent_code.{}.csv'.format(p.stage, p.stage), 'w')
        with open('./{}/imputation.{}.csv'.format(p.stage, p.stage), 'w') as handle:
            for i_ in range(n_out_batches+1):
                start_idx = i_*p.sample_size
                end_idx = min((i_+1)*p.sample_size, m)
                print('saving:', start_idx, end_idx)

                x_out_batch = input_matrix[start_idx:end_idx, :].todense()

                y_out_batch = sess.run(
                    h,
                    feed_dict={
                        X: x_out_batch,
                        pIn_holder: 1, pHidden_holder: 1
                    }
                )
                df_out_batch = pd.DataFrame(
                    data=y_out_batch,
                    columns=gene_ids,
                    index=cell_ids[range(start_idx, end_idx)]
                )

                latent_code = sess.run(
                    a_bottle_neck,
                    feed_dict={
                        X: x_out_batch,
                        pIn_holder: 1, pHidden_holder: 1
                    }
                )
                latent_code_df = pd.DataFrame(
                    data=latent_code,
                    index=cell_ids[range(start_idx, end_idx)]
                )

                if i_ == 0:
                    df_out_batch.to_csv(handle, float_format='%.6f')
                    latent_code_df.to_csv(handle2, float_format='%.6f')
                    print('RAM usage during mini-batch imputation and saving output: ',
                          '{} M'.format(usage()))
                else:
                    df_out_batch.to_csv(handle, header=None)
                    latent_code_df.to_csv(handle2, header=None)
        handle2.close()

    else:
        Y_input_arr = sess.run(h, feed_dict={X: input_matrix.todense(),
                                         pIn_holder: 1, pHidden_holder: 1})
        # save sample imputation
        Y_input_df = pd.DataFrame(data=Y_input_arr,
                                  columns=gene_ids,
                                  index=cell_ids)
        latent_code = sess.run(a_bottle_neck, feed_dict={X: input_matrix.todense(),
                                         pIn_holder: 1, pHidden_holder: 1})
        latent_code_df = pd.DataFrame(data=latent_code,
                                  index=cell_ids)
        print('RAM usage during whole data imputation and saving output: ',
              '{} M'.format(usage()))
        scimpute.save_hd5(Y_input_df, "{}/imputation.{}.hd5".format(p.stage,
                                                                        p.stage))
        scimpute.save_hd5(latent_code_df, "{}/latent_code.{}.hd5".format(p.stage,
                                                                    p.stage))


def save_model():    # save model
    print('> Saving model..')
    save_path = saver.save(sess, log_dir + "/{}.ckpt".format(p.stage))
    print("Model saved in: %s" % save_path)


def save_bottle_neck_representation():
    print("> save bottle-neck_representation")
    code_bottle_neck_input = sess.run(a_bottle_neck,
                                      feed_dict={
                                          X: sample_input,
                                          pIn_holder: 1,
                                          pHidden_holder: 1})
    np.save('{}/code_neck_valid.{}.npy'.format(p.stage, p.stage),
            code_bottle_neck_input)


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


def usage():
    process = psutil.Process(os.getpid())
    ram = process.memory_info()[0] / float(2 ** 20)
    ram = round(ram, 1)
    return ram


# sys.path.append('./bin')
# print('sys.path', sys.path)
print('python version:', sys.version)
print('tf.__version__', tf.__version__)

print("Usage: python -u <translate.py> <params.py>")
if len(sys.argv) == 2:
    param_file = sys.argv[1]
    param_name = param_file.rstrip('.py')

    from importlib.machinery import SourceFileLoader
    cwd = os.getcwd()
    p = SourceFileLoader(param_name,
                           cwd + '/' + param_file).load_module()

else:
    raise Exception('cmd err')

# refresh folder
log_dir = './{}'.format(p.stage)
scimpute.refresh_logfolder(log_dir)
tic_start = time.time()

# READ DATA into cell_row
print('>READING DATA..')
print('RAM usage before reading data: {} M'.format(usage()))
if p.fname_input.endswith('h5'):
    # for 10x genomics large h5 files
    input_obj = scimpute.read_sparse_matrix_from_h5(p.fname_input, p.genome_input,
                                                    p.ori_input)
    # gene_be_matrix.matrix = input_obj.matrix.log1p()
    input_matrix = input_obj.matrix
    gene_ids = input_obj.gene_ids
    cell_ids = input_obj.barcodes
    print('RAM usage after reading sparse matrix: {} M'.format(usage()))
    gc.collect()

    # Data Transformation
    print('> DATA TRANSFORMATION..')
    input_matrix = scimpute.sparse_matrix_transformation(input_matrix,
                                                         p.transformation_input)
    del(input_obj)
    gc.collect()
    print('RAM usage after {} transformation: {} M'.format(p.transformation_input,
                                                           usage()))

    # Test or not
    if p.test_flag:
        print('in test mode')
        input_matrix = input_matrix[:p.m, :p.n]
        gene_ids = gene_ids[:p.n]
        cell_ids = cell_ids[:p.m]
        gc.collect()

else:
    # For smaller files (hd5, csv, csv.gz)
    input_df = scimpute.read_data_into_cell_row(p.fname_input, p.ori_input)
    print('RAM usage after reading input_df: {} M'.format(usage()))

    # Data Transformation
    print('> DATA TRANSFORMATION..')
    input_df = scimpute.df_transformation(
        input_df.transpose(),
        transformation=p.transformation_input
    ).transpose() # [genes, cells] in df_trans()
    print('pandas input_df mem usage: ')
    input_df.info(memory_usage='deep')

    # Test or not
    if p.test_flag:
        print('in test mode')
        input_df = input_df.ix[:p.m, :p.n]
        gc.collect()

    # To sparse
    input_matrix = csr_matrix(input_df)  # todo: directly read into csr, get rid of input_df
    gene_ids = input_df.columns
    cell_ids = input_df.index
    print('RAM usage before deleting input_df: {} M'.format(usage()))
    del(input_df)
    gc.collect()  # working on mac
    print('RAM usage after deleting input_df: {} M'.format(usage()))


# Summary of data
print("name_input:", p.name_input)
_ = pd.DataFrame(data=input_matrix[:20, :4].todense(), index=cell_ids[:20],
                 columns=gene_ids[:4])
print("input_df:\n", _, "\n")
m, n = input_matrix.shape  # m: n_cells; n: n_genes
print('input_matrix: {} cells, {} genes\n'.format(m, n))

# split data and save indexes
input_train, input_valid, input_test, train_idx, valid_idx, test_idx = \
    scimpute.split__csr_matrix(input_matrix, a=p.a, b=p.b, c=p.c)

cell_ids_train = cell_ids[train_idx]
cell_ids_valid = cell_ids[valid_idx]
cell_ids_test = cell_ids[test_idx]

np.savetxt('{}/train.{}_index.txt'.format(p.stage, p.stage), cell_ids_train,
           fmt='%s')
np.savetxt('{}/valid.{}_index.txt'.format(p.stage, p.stage), cell_ids_valid,
           fmt='%s')
np.savetxt('{}/test.{}_index.txt'.format(p.stage, p.stage), cell_ids_test, fmt='%s')

print('RAM usage after splitting input data is: {} M'.format(usage()))

# todo: for backward support for older parameter files only
try:
    p.sample_size
    sample_size = p.sample_size
except:
    sample_size = int(9e4)

if sample_size < m:
    np.random.seed(1)
    rand_idx = np.random.choice(
        range(len(cell_ids_train)), min(sample_size, len(cell_ids_train)))
    sample_train = input_train[rand_idx, :].todense()
    sample_train_cell_ids = cell_ids_train[rand_idx]

    rand_idx = np.random.choice(
        range(len(cell_ids_valid)), min(sample_size, len(cell_ids_valid)))
    sample_valid = input_valid[rand_idx, :].todense()
    sample_valid_cell_ids = cell_ids_valid[rand_idx]

    rand_idx = np.random.choice(range(m), min(sample_size, m))
    sample_input = input_matrix[rand_idx, :].todense()
    sample_input_cell_ids = cell_ids[rand_idx]
    del rand_idx
    gc.collect()
    np.random.seed()
else:
    sample_train = input_train.todense()
    sample_valid = input_valid.todense()
    sample_input = input_matrix.todense()
    sample_train_cell_ids = cell_ids_train
    sample_valid_cell_ids = cell_ids_valid
    sample_input_cell_ids = cell_ids


print('len of sample_train: {}, sample_valid: {}, sample_input {}'.format(
    len(sample_train_cell_ids), len(sample_valid_cell_ids), len(sample_input_cell_ids)
))

# Start model
tf.reset_default_graph()

# define placeholders and variables
X = tf.placeholder(tf.float32, [None, n], name='X_input')  # input
pIn_holder = tf.placeholder(tf.float32, name='p.pIn')
pHidden_holder = tf.placeholder(tf.float32, name='p.pHidden')

# define layers and variables
tf.set_random_seed(3)  # seed
if p.L == 7:
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
    # # with tf.name_scope('Encoder_L4'):
    # #     e_w4, e_b4 = scimpute.weight_bias_variable('encoder4', p.n_hidden_3, p.n_hidden_4, p.sd)
    # #     e_a4 = scimpute.dense_layer('encoder4', e_a3, e_w4, e_b4, pHidden_holder)
    # # with tf.name_scope('Decoder_L4'):
    # #     d_w4, d_b4 = scimpute.weight_bias_variable('decoder4', p.n_hidden_4, p.n_hidden_3, p.sd)
    # #     d_a4 = scimpute.dense_layer('decoder4', e_a4, d_w4, d_b4, pHidden_holder)
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
elif p.L == 5:
    # change with layer
    with tf.name_scope('Encoder_L1'):
        e_w1, e_b1 = scimpute.weight_bias_variable('encoder1', n, p.n_hidden_1, p.sd)
        e_a1 = scimpute.dense_layer('encoder1', X, e_w1, e_b1, pIn_holder)
    with tf.name_scope('Encoder_L2'):
        e_w2, e_b2 = scimpute.weight_bias_variable('encoder2', p.n_hidden_1, p.n_hidden_2, p.sd)
        e_a2 = scimpute.dense_layer('encoder2', e_a1, e_w2, e_b2, pHidden_holder)
    with tf.name_scope('Decoder_L2'):
        d_w2, d_b2 = scimpute.weight_bias_variable('decoder2', p.n_hidden_2, p.n_hidden_1, p.sd)
        d_a2 = scimpute.dense_layer('decoder2', e_a2, d_w2, d_b2, pHidden_holder)
    with tf.name_scope('Decoder_L1'):
        d_w1, d_b1 = scimpute.weight_bias_variable('decoder1', p.n_hidden_1, n, p.sd)
        d_a1 = scimpute.dense_layer('decoder1', d_a2, d_w1, d_b1, pHidden_holder)  # todo: change input activations if model changed
    # define input/output
    a_bottle_neck = e_a2
elif p.L == 3:
    # change with layer
    with tf.name_scope('Encoder_L1'):
        e_w1, e_b1 = scimpute.weight_bias_variable('encoder1', n, p.n_hidden_1, p.sd)
        e_a1 = scimpute.dense_layer('encoder1', X, e_w1, e_b1, pIn_holder)
    with tf.name_scope('Decoder_L1'):
        d_w1, d_b1 = scimpute.weight_bias_variable('decoder1', p.n_hidden_1, n, p.sd)
        d_a1 = scimpute.dense_layer('decoder1', e_a1, d_w1, d_b1,
                                    pHidden_holder)  # todo: change input activations if model changed
    # define input/output
    a_bottle_neck = e_a1
else:
    raise Exception("{} L not defined, only 3, 5, 7 implemented".format(p.L))

h = d_a1

# define loss
with tf.name_scope("Metrics"):
    omega = tf.sign(X)  # 0 if 0, 1 if > 0; not possibly < 0 in our data
    mse_nz = tf.reduce_mean(
                    tf.multiply(
                        tf.pow(X-h, 2),
                        omega
                        )
                )
    mse = tf.reduce_mean(tf.pow(X-h, 2))
    reg_term = tf.reduce_mean(tf.pow(h, 2)) * p.reg_coef
    tf.summary.scalar('mse_nz__Y_vs_X', mse_nz)

    mse = tf.reduce_mean(tf.pow(X - h, 2))  # for report
    tf.summary.scalar('mse__Y_vs_X', mse)

# trainer
optimizer = tf.train.AdamOptimizer(p.learning_rate)
if p.mse_mode in ('mse_omega', 'mse_nz'):
    print('training on mse_nz')
    trainer = optimizer.minimize(mse_nz + reg_term)
elif p.mse_mode == 'mse':
    print('training on mse')
    trainer = optimizer.minimize(mse + reg_term)
else:
    raise Exception('mse_mode spelled wrong')

# start session
sess = tf.Session()

# restore variables
saver = tf.train.Saver()
if p.run_flag == 'load_saved':
    print('*** In TL Mode')
    saver.restore(sess, "./step1/step1.ckpt")
elif p.run_flag == 'rand_init':
    print('*** In Rand Init Mode')
    init = tf.global_variables_initializer()
    sess.run(init)
elif p.run_flag == 'impute':
    print('*** In impute mode   loading "step2.ckpt"..')
    saver.restore(sess, './step2/step2.ckpt')
    p.max_training_epochs = 0
    p.learning_rate = 0.0
    save_whole_imputation()
    print('imputation finished')
    toc_stop = time.time()
    print("reading took {:.1f} seconds".format(toc_stop - tic_start))
    exit()
else:
    raise Exception('run_flag err')

# define tensor_board writer
batch_writer = tf.summary.FileWriter(log_dir + '/batch', sess.graph)
valid_writer = tf.summary.FileWriter(log_dir + '/valid', sess.graph)

# prep mini-batch, and reporter vectors
epoch = 0
num_batch = int(math.floor(len(train_idx) // p.batch_size))  # floor
epoch_log = []
mse_nz_batch_vec, mse_nz_valid_vec, mse_nz_train_vec = [], [], []
mse_batch_vec, mse_valid_vec = [], []  # mse = MSE(X, h)
msej_batch_vec, msej_valid_vec = [], []  # msej = MSE(X, h), for genej, nz_cells

print('RAM usage after building the model is: {} M'.format(usage()))

# evaluate epoch0
evaluate_epoch_step2()

# Outer loop (epochs)
for epoch in range(1, p.max_training_epochs+1):
    tic_cpu, tic_wall = time.clock(), time.time()
    ridx_full = np.random.choice(len(train_idx), len(train_idx), replace=False)
    # inner loop (mini-batches)
    for i in range(num_batch):
        # x_batch
        indices = np.arange(p.batch_size * i, p.batch_size*(i+1))
        ridx_batch = ridx_full[indices]
        # x_batch = df1_train.ix[ridx_batch, :]
        x_batch = input_train[ridx_batch, :].todense()

        sess.run(trainer, feed_dict={X: x_batch,
                                     pIn_holder: p.pIn, pHidden_holder: p.pHidden})
    toc_cpu, toc_wall = time.clock(), time.time()

    # display step
    if (epoch == 1) or (epoch % p.display_step == 0):
        tic_log = time.time()

        print('#Epoch  {}  took:  {}  CPU seconds;  {} Wall seconds'.format(
            epoch, round(toc_cpu - tic_cpu, 2), round(toc_wall - tic_wall, 2)
        ))
        print('num-mini-batch per epoch: {}, till now: {}'.format(i+1, epoch*(i+1)))
        print('RAM usage: {:0.1f} M'.format(usage()))

        # debug
        # print('d_w1', sess.run(d_w1[1, 0:4]))  # verified when GradDescent used

        mse_batch, mse_nz_batch, h_batch = sess.run(
            [mse, mse_nz, h],
            feed_dict={X: x_batch, pHidden_holder: 1.0, pIn_holder: 1.0}
        )
        mse_valid, mse_nz_valid, Y_valid = sess.run(
            [mse, mse_nz, h],
            feed_dict={X: sample_valid, pHidden_holder: 1.0, pIn_holder: 1.0}
        )
        mse_batch_vec.append(mse_batch)
        mse_valid_vec.append(mse_valid)
        mse_nz_batch_vec.append(mse_nz_batch)
        mse_nz_valid_vec.append(mse_nz_valid)
        epoch_log.append(epoch)
        toc_log = time.time()
        print('mse_nz_batch:{};  mse_omage_valid: {}'.
              format(mse_nz_batch, mse_nz_valid))
        print('mse_batch:', mse_batch, '; mse_valid:', mse_valid)
        print('log time for each epoch: {}\n'.format(round(toc_log - tic_log, 1)))

    # snapshot step
    if (epoch % p.snapshot_step == 0) or (epoch == p.max_training_epochs):
        tic_log2 = time.time()
        save_fast_imputation()
        save_model()
        if p.mse_mode in ('mse_nz', 'mse_omega'):
            learning_curve_mse_nz(skip=math.floor(epoch / 5 / p.display_step))
        elif p.mse_mode == 'mse':
            learning_curve_mse(skip=math.floor(epoch / 5 / p.display_step))

        save_bottle_neck_representation()
        save_weights()
        visualize_weights()
        toc_log2 = time.time()
        log2_time = round(toc_log2 - tic_log2, 1)
        min_mse_valid = min(mse_nz_valid_vec)
        # os.system(
        #     '''for file in {0}/*npy
        #     do python -u weight_clustmap.py $file {0}
        #     done'''.format(p.stage)
        # )
        print('min_mse_nz_valid till now: {}'.format(min_mse_valid))
        print('snapshot_step: {}s'.format(log2_time))

batch_writer.close()
valid_writer.close()
sess.close()
toc_stop = time.time()
time_finish = round((toc_stop - tic_start), 2)
print("Gene-info Imputation Finished!")
print("Wall Time Used: {} seconds".format(time_finish))
exit()