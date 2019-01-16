#!/usr/bin/python
import sys
import os
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from importlib.machinery import SourceFileLoader
import math
import psutil
import time
from scipy.sparse import csr_matrix
import gc
import matplotlib
matplotlib.use('Agg')
import scimpute


def learning_curve_mse(epoch_log, mse_batch_vec, mse_valid_vec, stage, skip=1):
    '''Save mse curves to csv files
    
    Parameters:
    -----------
	skip: 
	epoch_log:
	mse_batch_vec:
	mse_valid_vec:
	stage: step1 or step2
	
	'''
    print('> plotting learning curves')
    scimpute.learning_curve(epoch_log, mse_batch_vec, mse_valid_vec,
                                title="Learning Curve MSE.{}".format(stage),
                                ylabel='MSE (X vs Y, nz)',
                                dir=stage,
                                skip=skip
                            )
    _ = np.asarray(list(zip(epoch_log, mse_batch_vec, mse_valid_vec)))
    _ = pd.DataFrame(data=_,
                     index=epoch_log,
                     columns=['Epoch', 'MSE_batch', 'MSE_valid']
                     ).set_index('Epoch')
    _.to_csv("./{}/mse.csv".format(stage))


#def learning_curve_mse_nz(skip=1):
def learning_curve_mse_nz(epoch_log, mse_nz_batch_vec, mse_nz_valid_vec, stage, skip=1):
    '''Save mse curves to csv files	
	
    Parameters:
    -----------
    skip:
    epoch_log:
    mse_nz_batch_vec:
    mse_nz_valid_vec:
    stage:
    
    '''
    print('> plotting learning curves')
    scimpute.learning_curve(epoch_log, mse_nz_batch_vec, mse_nz_valid_vec,
                                title="Learning Curve MSE_NZ.{}".format(stage),
                                ylabel='MSE_NZ (X vs Y, nz)',
                                dir=stage,
                                skip=skip
                            )
    _ = np.asarray(list(zip(epoch_log, mse_nz_batch_vec, mse_nz_valid_vec)))
    _ = pd.DataFrame(data=_,
                     index=epoch_log,
                     columns=['Epoch', 'MSE_NZ_batch', 'MSE_NZ_valid']
                     ).set_index('Epoch')
    _.to_csv("./{}/mse_nz.csv".format(stage))

def fast_imputation(sess, h, X, pIn_holder, pHidden_holder, input_data, gene_ids, cell_ids):
	'''Calculate /and save/ the snapshot results of the current model on the whole dataset
	
	Parameters:
	-----------
	
	'''
	Y_input_arr = sess.run(h, feed_dict={X: input_data, pIn_holder: 1, pHidden_holder: 1})
	# save sample imputation
	Y_input_df = pd.DataFrame(data=Y_input_arr, columns=gene_ids, index=cell_ids)
	
	return Y_input_df  
																		
#def save_whole_imputation:
def save_whole_imputation(sess, X, h, a_bottleneck, pIn_holder,pHidden_holder, input_matrix, gene_ids, cell_ids, p, m):
    ''' calculate and save imputation results for an input matrix at the 'impute' mode. If  the number 
    of cells is larger than a threshold (large_size: 1e5), save results of m//p.sample_size 'folds'.
    
    Parameters
    ----------
    
    '''
    
    if m > p.large_size:
        #impute on small data blocks to avoid high memory cost 
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
                    a_bottleneck,
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

    else: # if m the # of cells is less than large_size (1e5))
        Y_input_arr = sess.run(h, feed_dict={X: input_matrix.todense(),
                                         pIn_holder: 1, pHidden_holder: 1})
        # save sample imputation
        Y_input_df = pd.DataFrame(data=Y_input_arr,
                                  columns=gene_ids,
                                  index=cell_ids)
        latent_code = sess.run(a_bottleneck, feed_dict={X: input_matrix.todense(),
                                         pIn_holder: 1, pHidden_holder: 1})
        latent_code_df = pd.DataFrame(data=latent_code,
                                  index=cell_ids)
        print('RAM usage during whole data imputation and saving output: ',
              '{} M'.format(usage()))
        scimpute.save_hd5(Y_input_df, "{}/imputation.{}.hd5".format(p.stage,
                                                                        p.stage))
        scimpute.save_hd5(latent_code_df, "{}/latent_code.{}.hd5".format(p.stage,
                                                                    p.stage))
																	
def visualize_weight(sess, stage, w_name, b_name):
    w = eval(w_name)
    b = eval(b_name)
    w_arr = sess.run(w)
    b_arr = sess.run(b)
    b_arr = b_arr.reshape(len(b_arr), 1)
    b_arr_T = b_arr.T
    scimpute.visualize_weights_biases(w_arr, b_arr_T,
                                      '{},{}.{}'.format(w_name, b_name, stage),
                                      dir=stage)
									  
def visualize_weights(sess, stage, en_de_layers):
    for l1 in range(1, en_de_layers+1):
        encoder_weight = 'e_w'+str(l1)
        encoder_bias = 'e_b'+str(l1)
        visualize_weight(sess, stage, encoder_weight, encoder_bias)
        
        decoder_bias = 'd_b'+str(l1)
        decoder_weight = 'd_w'+str(l1)
        visualize_weight(sess, stage, decoder_weight, decoder_bias)


def save_weights(sess, stage, en_de_layers):
    print('save weights in npy')
    for l1 in range(1, en_de_layers+1):
        encoder_weight_name = 'e_w'+str(l1)
        encoder_bias_name = 'e_b'+str(l1)
        decoder_bias_name = 'd_b'+str(l1)
        decoder_weight_name = 'd_w'+str(l1)
        np.save('{}/{}.{}'.format(stage, encoder_weight_name, stage),
                sess.run(eval(encoder_weight_name)))
        np.save('{}/{}.{}'.format(stage, decoder_weight_name, stage),
                sess.run(eval(decoder_weight_name)))
        np.save('{}/{}.{}'.format(stage, encoder_bias_name, stage),
                sess.run(eval(encoder_bias_name)))
        np.save('{}/{}.{}'.format(stage, decoder_bias_name, stage),
                sess.run(eval(decoder_bias_name)))


def usage():
    process = psutil.Process(os.getpid())
    ram = process.memory_info()[0] / float(2 ** 20)
    ram = round(ram, 1)
    return ram


# sys.path.append('./bin')
# print('sys.path', sys.path)
#print('python version:', sys.version)
#print('tf.__version__', tf.__version__)

def late_main(p, log_dir, rand_state=3):

	##0. read data and extract gene IDs and cell IDs
	input_matrix, gene_ids, cell_ids = read_data(p)
	
	##1. split data and save indexes
	#input p, input_matrix, cell_ids
	#return cell_ids_train, cell_ids_valid, cell_ids_test
	m, n = input_matrix.shape
	input_train, input_valid, input_test, train_idx, valid_idx, test_idx = \
		scimpute.split__csr_matrix(input_matrix, a=p.a, b=p.b, c=p.c)

	cell_ids_train = cell_ids[train_idx]
	cell_ids_valid = cell_ids[valid_idx]
	cell_ids_test = cell_ids[test_idx]

	np.savetxt('{}/train.{}_index.txt'.format(p.stage, p.stage), cell_ids_train, fmt='%s')
	np.savetxt('{}/valid.{}_index.txt'.format(p.stage, p.stage), cell_ids_valid, fmt='%s')
	np.savetxt('{}/test.{}_index.txt'.format(p.stage, p.stage), cell_ids_test, fmt='%s')
    
	print('RAM usage after splitting input data is: {} M'.format(usage()))

	# todo: for backward support for older parameter files only
    # sample_size is 1000 in default; if sample_size is less than the number of cells (m), 
    # we reconstruct the training and validation sets by randomly sampling.
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
        #?? the following sample_input is a matrix sampled randomly, and should it be a matrix containing
        # sample_training and sample_valid
		rand_idx = np.random.choice(range(m), min(sample_size, m))
		sample_input = input_matrix[rand_idx, :].todense()
		sample_input_cell_ids = cell_ids[rand_idx]
		
		del rand_idx
		gc.collect()
		np.random.seed()
	else:
		sample_input = input_matrix.todense()
		sample_train = input_train.todense()
		sample_valid = input_valid.todense()
		sample_input_cell_ids = cell_ids
		sample_train_cell_ids = cell_ids_train
		sample_valid_cell_ids = cell_ids_valid

	print('len of sample_train: {}, sample_valid: {}, sample_input {}'.format(
		len(sample_train_cell_ids), len(sample_valid_cell_ids), len(sample_input_cell_ids)
	))
	
	##2. model training and validation
	#2.1 init --> keep this in the main
	tf.reset_default_graph()
	# define placeholders and variables
	X = tf.placeholder(tf.float32, [None, n], name='X_input')  # input
	pIn_holder = tf.placeholder(tf.float32, name='p.pIn')  #keep_prob for dropout
	pHidden_holder = tf.placeholder(tf.float32, name='p.pHidden')#keep_prob for dropout

	#2.2 define layers and variables
	# input p, X, pIn_holder, pHidden_holder, n
	# return a_bottleneck, h(d_a1)
	a_bottleneck, h = build_late(X, pHidden_holder, pIn_holder, p, n, rand_state = 3)	

	#2.3 define loss
	# input X, h, p
	# return mse_nz, mse, reg_term	
	mse_nz, mse, reg_term = build_metrics(X, h, p.reg_coef)

	#2.4 costruct the trainer --> keep this section in the main
	optimizer = tf.train.AdamOptimizer(p.learning_rate)
	if p.mse_mode in ('mse_omega', 'mse_nz'):
		print('training on mse_nz')
		trainer = optimizer.minimize(mse_nz + reg_term)
	elif p.mse_mode == 'mse':
		print('training on mse')
		trainer = optimizer.minimize(mse + reg_term)
	else:
		raise Exception('mse_mode spelled wrong')

	#2.5 Init a session accoding to the run_flag
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
		
		## save_whole_imputation
		save_whole_imputation(sess, X, h, a_bottleneck, pIn_holder, 
								pHidden_holder, input_matrix, gene_ids, 
								cell_ids, p, m)
		print('imputation finished')
		#toc_stop = time.time()
		#print("reading took {:.1f} seconds".format(toc_stop - tic_start))
		exit()
	else:
		raise Exception('run_flag err')

	# define tensor_board writer
	batch_writer = tf.summary.FileWriter(log_dir + '/batch', sess.graph)
	valid_writer = tf.summary.FileWriter(log_dir + '/valid', sess.graph)

	# prep mini-batch, and reporter vectors
	num_batch = int(math.floor(len(train_idx) // p.batch_size))  # floor
	epoch_log = []
	mse_nz_batch_vec, mse_nz_valid_vec = [], [] #, mse_nz_train_vec = [], [], []
	mse_batch_vec, mse_valid_vec = [], []  # mse = MSE(X, h)
	#msej_batch_vec, msej_valid_vec = [], []  # msej = MSE(X, h), for genej, nz_cells
	print('RAM usage after building the model is: {} M'.format(usage()))

	epoch = 0    
    #2.6. pre-training epoch (0)
	#save imputation results before training steps
	print("Evaluation: epoch{}".format(epoch))
	epoch_log.append(epoch)
	mse_train, mse_nz_train = sess.run([mse, mse_nz], feed_dict={X: sample_train,pHidden_holder: 1.0, pIn_holder: 1.0})
	mse_valid, mse_nz_valid = sess.run([mse, mse_nz],feed_dict={X: sample_valid,pHidden_holder: 1.0, pIn_holder: 1.0})
	print("mse_nz_train=", round(mse_nz_train, 3), "mse_nz_valid=",round(mse_nz_valid, 3))
	print("mse_train=", round(mse_train, 3),"mse_valid=", round(mse_valid, 3))
	mse_batch_vec.append(mse_train)
	mse_valid_vec.append(mse_valid)
	mse_nz_batch_vec.append(mse_nz_train)
	mse_nz_valid_vec.append(mse_nz_valid)  
	
	#2.7. training epochs (1-)
	for epoch in range(1, p.max_training_epochs+1):
		tic_cpu, tic_wall = time.clock(), time.time()
		
		ridx_full = np.random.choice(len(train_idx), len(train_idx), replace=False)
		
		#2.7.1 training model on mini-batches
		for i in range(num_batch):
			# x_batch
			indices = np.arange(p.batch_size * i, p.batch_size*(i+1))
			ridx_batch = ridx_full[indices]
			# x_batch = df1_train.ix[ridx_batch, :]
			x_batch = input_train[ridx_batch, :].todense()

			sess.run(trainer, feed_dict={X: x_batch,
										 pIn_holder: p.pIn, 
										 pHidden_holder: p.pHidden})
					
		toc_cpu, toc_wall = time.clock(), time.time()

		#2.7.2 save the results of epoch 1 and all display steps (epochs)
		if (epoch == 1) or (epoch % p.display_step == 0):
			tic_log = time.time()
			
			print('#Epoch  {}  took:  {}  CPU seconds;  {} Wall seconds'.format(
				epoch, round(toc_cpu - tic_cpu, 2), round(toc_wall - tic_wall, 2)
			))
			print('num-mini-batch per epoch: {}, till now: {}'.format(i+1, epoch*(i+1)))
			print('RAM usage: {:0.1f} M'.format(usage()))

			# debug
			# print('d_w1', sess.run(d_w1[1, 0:4]))  # verified when GradDescent used
			
			# training mse and mse_nz of the last batch
			mse_batch, mse_nz_batch, h_batch = sess.run(
				[mse, mse_nz, h],
				feed_dict={X: x_batch, pHidden_holder: 1.0, pIn_holder: 1.0}
			)
			# validation mse and mse_nz of the sample validation set (1000)
			mse_valid, mse_nz_valid, Y_valid = sess.run(
				[mse, mse_nz, h],
				feed_dict={X: sample_valid, pHidden_holder: 1.0, pIn_holder: 1.0}
			)
			
			toc_log = time.time()
			
			print('mse_nz_batch:{};  mse_omage_valid: {}'.
				  format(mse_nz_batch, mse_nz_valid))
			print('mse_batch:', mse_batch, '; mse_valid:', mse_valid)
			print('log time for each epoch: {}\n'.format(round(toc_log - tic_log, 1)))
			
			mse_batch_vec.append(mse_batch)
			mse_valid_vec.append(mse_valid)
			mse_nz_batch_vec.append(mse_nz_batch)
			mse_nz_valid_vec.append(mse_nz_valid)
			epoch_log.append(epoch)

		#2.7.3 save snapshot step
		if (epoch % p.snapshot_step == 0) or (epoch == p.max_training_epochs):
			tic_log2 = time.time()
			
			#1.save imputation results
            #if the input matrix is large (m > p.large_size), only save the 
			#imputation results of a small sample set (sample_input)
			print("> Impute and save.. ")
			if m > p.large_size:
				Y_input_df = fast_imputation(sess, h, X, pIn_holder, pHidden_holder, sample_input, gene_ids, sample_input_cell_ids)	  
				scimpute.save_hd5(Y_input_df, "{}/sample_imputation.{}.hd5".format(p.stage,
																				p.stage))
			else:
				Y_input_df = fast_imputation(sess, h, X, pIn_holder, pHidden_holder, input_matrix.todense(), gene_ids, cell_ids)	
				scimpute.save_hd5(Y_input_df, "{}/imputation.{}.hd5".format(p.stage,
																				p.stage))
			#2.save model
			print('> Saving model..')
			save_path = saver.save(sess, log_dir + "/{}.ckpt".format(p.stage))
			print("Model saved in: %s" % save_path)
			
			#3.save the training and test curve
			if p.mse_mode in ('mse_nz', 'mse_omega'):
				#learning_curve_mse_nz(skip=math.floor(epoch / 5 / p.display_step))
				learning_curve_mse_nz(epoch_log, mse_nz_batch_vec, mse_nz_valid_vec, 
                          p.stage, skip=math.floor(epoch / 5 / p.display_step))
			elif p.mse_mode == 'mse':
				#learning_curve_mse(skip=math.floor(epoch / 5 / p.display_step))
				learning_curve_mse(epoch_log, mse_batch_vec, mse_valid_vec, p.stage, 
                       skip=math.floor(epoch / 5 / p.display_step))
									
			#4.save save_bottleneck_representation
			print("> save bottleneck_representation")
			code_bottleneck_input = sess.run(a_bottleneck,
											  feed_dict={
												  X: sample_input,
												  pIn_holder: 1,
												  pHidden_holder: 1})
			np.save('{}/code_neck_valid.{}.npy'.format(p.stage, p.stage),
					code_bottleneck_input)
		
			#save_weights()
			save_weights(sess, p.stage, en_de_layers=p.l)
			
			#visualize_weights()
			visualize_weights(sess, p.stage, en_de_layers=p.l)
			
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

def build_late(X, pHidden_holder, pIn_holder, p, n, rand_state = 3):
	#5.2 define layers and variables
	# input p, X, pIn_holder, pHidden_holder, n
	# return a_bottleneck, h(d_a1)
	tf.set_random_seed(rand_state)  # seed
	global e_w1, e_b1, e_a1, e_w2, e_b2, e_a2, e_w3, e_b3, e_a3
	global d_w1, d_b1, d_a1, d_w2, d_b2, d_a2, d_w3, d_b3, d_a3
	
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
		a_bottleneck = e_a3
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
		a_bottleneck = e_a2
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
		a_bottleneck = e_a1
	else:
		raise Exception("{} L not defined, only 3, 5, 7 implemented".format(p.L))

	h = d_a1
	
	return a_bottleneck, h
	
def build_metrics(X, h, coef):
	with tf.name_scope("Metrics"):
		omega = tf.sign(X)  # 0 if 0, 1 if > 0; not possibly < 0 in our data
		mse_nz = tf.reduce_mean(
						tf.multiply(
							tf.pow(X-h, 2),
							omega
							)
					)
		mse = tf.reduce_mean(tf.pow(X-h, 2))
		reg_term = tf.reduce_mean(tf.pow(h, 2)) * coef
		tf.summary.scalar('mse_nz__Y_vs_X', mse_nz)

		mse = tf.reduce_mean(tf.pow(X - h, 2))  # for report
		tf.summary.scalar('mse__Y_vs_X', mse)
		
	return 	mse_nz, mse, reg_term

def load_params(mode, infile):
	'''load the 'global_params.py' file '''
	cwd = os.getcwd()
	param_file = 'global_params.py'
	param_name = param_file.rstrip('.py')
	p = SourceFileLoader(param_name,
						   cwd + '/' + param_file).load_module()
	p.fname_input = infile
	p.mode = mode
	if mode == 'pre-training':
	    # step1/rand_init for pre-training on reference
	    p.stage = 'step1'
	    p.run_flag = 'rand_init'
	    p.learning_rate = 3e-4  # step1: 3e-4 for 3-7L, 3e-5 for 9L
	elif mode == 'translate':
	    # step2/load_saved from step1, for transfer learning
	    p.stage = 'step2'  # step1/step2 (not others)
	    p.run_flag = 'load_saved'  # rand_init/load_saved
	    p.learning_rate = 3e-5  # step2: 3e-5 for 3-7L, 3e-6 for 9L
	elif mode == 'late':
	    # step2/rand_init for one step training
	    p.stage = 'step2'
	    p.run_flag = 'rand_init'
	    p.learning_rate = 3e-4  # step1: 3e-4 for 3-7L, 3e-5 for 9L
	elif mode == 'impute':
	    # step2/load_saved/learning_rate=0, just impute and output
	    p.stage = 'impute'
	    p.run_flag = 'impute'
	    p.learning_rate = 0.0
	elif mode == 'analysis':
		p.tag = 'Eval'
		p.stage = 'Eval'
	else:
	    print('The mode you entered cannot be recognized.')
	    print('Valid mode options: pre-training | late | translate | impute | analysis')
	    p.mode = 'invalid'
	    return p
		
	if p.test_flag:
	    p.max_training_epochs = 10 # 3L:100, 5L:1000, 7L:1000, 9L:3000
	    p.display_step = 1  # interval on learning curve
	    p.snapshot_step = 5  # interval of saving session, imputation
	    p.m = 1000
	    p.n = 300
	    p.sample_size = int(240)
	    print('in test mode\n',
	          'num-genes set to {}, num-cells set to {}\n'.format(p.n, p.m),
	          'sample size set to {}'.format(p.sample_size))	
	return p

# to do: modify to display based on mode
#
def display_params(p):
	# PRINT PARAMETERS
	print('\nmode:', p.mode)
	print('\nData:')
	print('fname_input:', p.fname_input)
	print('name_input:', p.name_input)
	print('ori_input:', p.ori_input)
	print('transformation_input:', p.transformation_input)

	if (p.mode == 'pre-training') or (p.mode == 'late') or (p.mode == 'translate'):
		print('data split: [{}/{}/{}]'.format(p.a, p.b, p.c))
		
		print('\nParameters:')
		print('mse_mode:', p.mse_mode)
		print('stage:', p.stage)
		print('init:', p.run_flag)
		print('test_mode:', p.test_flag)
		print('total number of layers: {}'.format(p.L))
		for l_tmp in range(1, p.l+1):
		  print("n_hidden{}: {}".format(l_tmp, eval('p.n_hidden_'+str(l_tmp))))
		
		print('learning_rate:', p.learning_rate)
		print('reg_coef:', p.reg_coef)
		print('batch_size:', p.batch_size)
		print('sample_zie: ', p.sample_size)
		print('pIn:', p.pIn)
		print('pHidden:', p.pHidden)
		print('max_training_epochs:', p.max_training_epochs)
		print('display_step', p.display_step)
		print('snapshot_step', p.snapshot_step)
	elif p.mode == 'analysis':
		print('fname_imputation:', p.fname_imputation)
		print('transformation_imputation', p.transformation_imputation)
		print('fname_ground_truth: ', p.fname_ground_truth)
		print('transformation_ground_truth', p.transformation_ground_truth)
		print('gene_pair_list: ', p.gene_pair_list)
		
	print('\n')

def read_data(p):
	'''READ DATA
	Parameters
	------------
	p: 
		
	Return
	-----------
	
	'''

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

		# Test or not: m*n subset (1000 * 300). Delete later
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
	
	return input_matrix, gene_ids, cell_ids


def load_results(p):
	'''READ DATA
	Parameters
	------------
		p: parameters from global_params.py and example.py
		
	Return
	-----------
                X: input data matrix; genes in columns (same below)
                Y: imputed data matrix
                G: ground truth	
	'''

#	print('>READING DATA..')
#	X = scimpute.read_data_into_cell_row(p.fname_input, p.ori_input)
	X, gene_ids, cell_ids = read_data(p)
	X = pd.DataFrame(data=X.todense(), index=cell_ids,
					 columns=gene_ids)
	Y = scimpute.read_data_into_cell_row(p.fname_imputation, p.ori_imputation)
	if p.fname_input == p.fname_ground_truth:
		G = X
	else:
		G = scimpute.read_data_into_cell_row(p.fname_ground_truth, p.ori_ground_truth)
	
#	print('> DATA TRANSFORMATION..')
	Y = scimpute.df_transformation(Y.transpose(), transformation=p.transformation_imputation).transpose()
#	X = scimpute.df_transformation(X.transpose(), transformation=p.transformation_input).transpose()
	if p.fname_input == p.fname_ground_truth:
		G = X
	else:
		G = scimpute.df_transformation(G.transpose(), transformation=p.transformation_ground_truth).transpose()

	# subset/sort X, G to match Y
	# todo: support sparse matrix
	X = X.loc[Y.index, Y.columns]
	G = G.loc[Y.index, Y.columns]

	# TEST MODE OR NOT
	if p.test_flag:
		print('in test mode')
		Y = Y.ix[0:p.m, 0:p.n]
		G = G.ix[0:p.m, 0:p.n]
		X = X.ix[0:p.m, 0:p.n]

	# INPUT SUMMARY
	print('\nIn this code, matrices should have already been transformed into cell_row')
	print('Y (imputation):', p.fname_imputation, p.ori_imputation, p.transformation_imputation,'\n', Y.ix[0:20, 0:3])
	print('X (input):', p.fname_input, p.ori_input, p.transformation_input,'\n', X.ix[0:20, 0:3])
	print('G (ground truth):', p.fname_ground_truth, p.ori_ground_truth, p.transformation_ground_truth,'\n', G.ix[0:20, 0:3])
	print('Y.shape', Y.shape)
	print('X.shape', X.shape)
	print('G.shape', G.shape)

	return X, Y, G


def calculate_MSEs(X, Y, G):
	'''calculate MSEs
	MSE between imputation and input
	MSE between imputation and ground truth

	Parameters
	------------
		X: input data matrix; genes in columns (same below)
		Y: imputed data matrix
		G: ground truth
		
	Return
	-----------
		4 MSEs
	'''

	print('\n> MSE Calculation')
	max_y, min_y = scimpute.max_min_element_in_arrs([Y.values])
	print('Max in Y is {}, Min in Y is{}'.format(max_y, min_y))
	max_g, min_g = scimpute.max_min_element_in_arrs([G.values])
	print('Max in G is {}, Min in G is{}'.format(max_g, min_g))

	mse1_nz = scimpute.mse_omega(Y, X)
	mse1_nz = round(mse1_nz, 7)
	print('MSE1_NZ between Imputation and Input: ', mse1_nz)

	mse1 = scimpute.mse(Y, X)
	mse1 = round(mse1, 7)
	print('MSE1 between Imputation and Input: ', mse1)

	mse2_nz = scimpute.mse_omega(Y, G)
	mse2_nz = round(mse2_nz, 7)
	print('MSE2_NZ between Imputation and Ground_truth: ', mse2_nz)

	mse2 = scimpute.mse(Y, G)
	mse2 = round(mse2, 7)
	print('MSE2 between Imputation and Ground_truth: ', mse2)

	return mse1_nz, mse1, mse2_nz, mse2


def analyze_variation_in_genes(X, Y, G, p):
	'''calculate and visualize standard deviation in each gene
	write SDs to files
	plot histograms of SDs

	Parameters
	------------
		X: input data matrix; genes in columns (same below)
		Y: imputed data matrix
		G: ground truth
	p: parameters
                
	Return
	-----------
		None
	'''

	print('\n calculating standard deviation in each gene for input and imputed matrix')

	x_std_df, y_std_df = scimpute.nz_std(X, Y)
	x_std_df, g_std_df = scimpute.nz_std(X, G)  # purpose: compare G with Y


	#std_ratio_yx_df = pd.DataFrame(data= y_std_df.values / x_std_df.values, index=X.columns, columns=['sd_ratio'])
	#std_ratio_yg_df = pd.DataFrame(data= y_std_df.values / g_std_df.values, index=X.columns, columns=['sd_ratio'])
	std_ratio_yx_data = [(y/x if x!=0 else None) for y, x in zip(y_std_df.values, x_std_df.values)]
	std_ratio_yx_df =pd.DataFrame(data = std_ratio_yx_data, index=X.columns, columns=['sd_ratio'])	
	std_ratio_yg_data = [(y/x if x!=0 else None) for y, x in zip(y_std_df.values, g_std_df.values)]
	std_ratio_yg_df = pd.DataFrame(data= std_ratio_yg_data, index=X.columns, columns=['sd_ratio'])
	
	std_min = min(y_std_df.min(), x_std_df.min(), g_std_df.min())
	std_max = max(y_std_df.max(), x_std_df.max(), g_std_df.max())

	print('generating histograms of standard deviations')
	scimpute.hist_df(
		y_std_df,
		xlab='Standard Deviation', title='Imputation({})'.format(p.name_imputation),
		range=(std_min, std_max),
		dir=p.tag)
	scimpute.hist_df(
		x_std_df,
		xlab='Standard Deviation', title='Input({})'.format(p.name_input),
		range=(std_min, std_max),
		dir=p.tag)
	scimpute.hist_df(
		g_std_df,
		xlab='Standard Deviation', title='Ground Truth({})'.format(p.name_input),
		range=(std_min, std_max),
		dir=p.tag)
	scimpute.hist_df(
		std_ratio_yx_df,
		xlab='Ratio of Imputation SD vs Input SD',
		title='',
		range=(std_min, std_max),  
		dir=p.tag)
	scimpute.hist_df(
		std_ratio_yg_df,
		xlab='Ratio of Imputation SD vs Ground Truth SD',
		title='',
		range=(std_min, std_max),
		dir=p.tag)

	std_ratio_yx_df.to_csv('sd_ratio_imputed_vs_input.csv')
	std_ratio_yg_df.to_csv('sd_ratio_imputed_vs_groundtruth.csv')


def visualize_all_genes(X, Y, G, p):
	''' generate plots using all genes
        
	Parameters
        ------------
                X: input data matrix; genes in columns (same below)
                Y: imputed data matrix
                G: ground truth
                p: parameters
                
        Return
        -----------
                None
        '''

	# histograms of gene expression
	max_expression = max(G.values.max(), X.values.max(), Y.values.max())
	min_expression = min(G.values.min(), X.values.min(), Y.values.min())
	print('\n max expression:', max_expression)
	print('\n min expression:', min_expression)

	scimpute.hist_df(
		Y, xlab='Expression', title='Imputation({})'.format(p.name_imputation),
		dir=p.tag, range=[min_expression, max_expression])
	scimpute.hist_df(
		X,  xlab='Expression', title='Input({})'.format(p.name_input),
		dir=p.tag, range=[min_expression, max_expression])
	scimpute.hist_df(
		G,  xlab='Expression', title='Ground Truth({})'.format(p.name_ground_truth),
		dir=p.tag, range=[min_expression, max_expression])

	# histograms of correlations between genes in imputation and ground truth
	# and of correlations between cells in imputation and ground truth
	# when ground truth is not provide, 
	# input is used as ground truth
	print('\n> Correlations between ground truth and imputation')
	print('ground truth dimension: ', G.shape, 'imputation dimension: ', Y.shape)
	print('generating histogram for correlations of genes between ground truth and imputation')
	scimpute.hist_2matrix_corr(
		G.values, Y.values,
		title="Correlation for each gene\n(Ground_truth vs Imputation)\n{}\n{}".
			format(p.name_ground_truth, p.name_imputation),
		dir=p.tag, mode='column-wise', nz_mode='first'  # or ignore
	)

	print('generating histogram for correlations of cells between ground truth and imputation')
	scimpute.hist_2matrix_corr(
		G.values, Y.values,
		title="Correlation for each cell\n(Ground_truth vs Imputation)\n{}\n{}".
			format(p.name_ground_truth, p.name_imputation),
		dir=p.tag, mode='row-wise', nz_mode='first'
	)

	#  heatmaps of data matrices
	print('\n> Generating heatmaps of data matrices')
	range_max, range_min = scimpute.max_min_element_in_arrs([Y.values, G.values, X.values])
	print('\nrange:', range_max, ' ', range_min)

	scimpute.heatmap_vis(Y.values,
		title='Imputation ({})'.format(p.name_imputation),
		xlab='Genes', ylab='Cells', vmax=range_max, vmin=range_min, dir=p.tag)

	scimpute.heatmap_vis(X.values,
		title='Input ({})'.format(p.name_input),
		xlab='Genes', ylab='Cells', vmax=range_max, vmin=range_min, dir=p.tag)

	scimpute.heatmap_vis(G.values,
		title='Ground_truth ({})'.format(p.name_ground_truth),
		xlab='Genes', ylab='Cells', vmax=range_max, vmin=range_min, dir=p.tag)

	# PCA and tSNE plots
	print('\n> Generating PCA and tSNE plots')
	if p.cluster_file is not None:
		cluster_info = scimpute.read_data_into_cell_row(p.cluster_file)
		# cluster_info = cluster_info.astype('str')
	else:
		cluster_info = None

	scimpute.pca_tsne(df_cell_row=Y, cluster_info=cluster_info,
                            title=p.name_imputation, dir=p.tag)
	scimpute.pca_tsne(df_cell_row=X, cluster_info=cluster_info,
                            title=p.name_input, dir=p.tag)
	scimpute.pca_tsne(df_cell_row=G, cluster_info=cluster_info,
                            title=p.name_ground_truth, dir=p.tag)


def visualize_selected_genes(X, Y, G, p):
	''' generate plots for genes specified by the user
        
	Parameters
	------------
		X: input data matrix; genes in columns (same below)
		Y: imputed data matrix
		G: ground truth
		p: parameters
                
	Return
	-----------
		None
	'''

	gene_pair_dir = p.tag+'/pairs'

	List = p.gene_pair_list
	print(">n> Scatterplots of selected gene pairs")
	scimpute.gene_pair_plot(Y, list=List, tag='(Imputation)', dir=gene_pair_dir)
	scimpute.gene_pair_plot(X, list=List, tag='(Input)', dir=gene_pair_dir)
	scimpute.gene_pair_plot(G, list=List, tag='(Ground_truth)', dir=gene_pair_dir)

	print("\n> Scatterplots for selected genes")
	print("ground truth vs imputation, ground truth vs input")
	gene_dir = p.tag+'/genes'
	# genetate a list of genes using the gene_pair_list
	gene_list = [gene for pair in List for gene in pair]
	for j in gene_list:
		try:
			print('for ', j)
			Y_j = Y.ix[:, j]
			G_j = G.ix[:, j]
			X_j = X.ix[:, j]
		except KeyError:
			print('KeyError: gene ID does not exist')
			continue

	scimpute.scatterplot2(G_j, Y_j, range='same',
                          title=str(str(j) + '\n(Ground Truth vs Imputation) '),
                          xlabel='Ground Truth',
                          ylabel='Imputation',
                          dir=gene_dir
                          )
	scimpute.scatterplot2(G_j, X_j, range='same',
                          title=str(str(j) + '\n(Ground Truth vs Input) '),
                          xlabel='Ground Truth',
                          ylabel='Input',
                          dir=gene_dir
                          )


	# Discretize gene expression values
	# and re-generate pairwise plots 
	Y = scimpute.df_exp_discretize_log10(Y)

	print('\n> Discrete gene pair relationship in imputation')
	gene_pair_dir = p.tag+'/pairs_discrete'
#	List = p.gene_pair_list
	scimpute.gene_pair_plot(Y, list=List, tag='(Imputation Discrete) ',
                        dir=gene_pair_dir)

	print("\n> Discrete imputation vs ground truth")
	gene_dir = p.tag+'/genes_discrete'
	for j in gene_list:
		try:
			print('for ', j)
			Y_j = Y.ix[:, j]
			G_j = G.ix[:, j]
			X_j = X.ix[:, j]
		except KeyError:
			print('KeyError: gene ID does not exist')
			continue

	scimpute.scatterplot2(G_j, Y_j, range='same',
                          title=str(str(j) + '\n(Ground_truth vs Imputation) '),
                          xlabel='Ground Truth',
                          ylabel='Imputation',
                          dir=gene_dir
                          )
	scimpute.scatterplot2(G_j, X_j, range='same',
                          title=str(str(j) + '\n(Ground_truth vs Input) '),
                          xlabel='Ground Truth',
                          ylabel='Input',
                          dir=gene_dir
                          )


def result_analysis_main(p):
	'''analyzing imputation output

	Parameters
        ------------
                p: parameters from global_params.py and example.py
                
        Return
        -----------
		None
	'''

	# load imputation results and input data
	X, Y, G = load_results(p)

	# calculate MSEs
	mse1_nz, mse1, mse2_nz, mse2 = calculate_MSEs(X, Y, G)

	# calculate and visualize variation in genes
	analyze_variation_in_genes(X, Y, G, p)

	# visualize results using all genes
	visualize_all_genes(X, Y, G, p)

	# visualize selected genes
	visualize_selected_genes(X, Y, G, p)


def parse_args(argv):
	parser = argparse.ArgumentParser(description = 'Help information')
	parser.add_argument('-mode', help='mode options: pre-training | late | translate | impute | analysis')
	parser.add_argument('-infile', help='file path of input data') 
	
	return parser.parse_args(argv)

if __name__ == '__main__':
	
	##1. load parameter module and use name 'p'
	#print("Usage: python late.py -mode <late> -infile <xx.hd5>")
	argms = parse_args(sys.argv[1:])
	p = load_params(argms.mode, argms.infile)	
	if p.mode =='invalid':
		exit(0)
	##2. refresh folder
	log_dir = './{}'.format(p.stage)
	scimpute.refresh_logfolder(log_dir)

	tic_start = time.time()
	#3. load data
	input_matrix, gene_ids, cell_ids = read_data(p)
	
	#4. call late
	late_main(input_matrix, gene_ids, cell_ids, p, log_dir, rand_state = 3)
	toc_stop = time.time()
	time_finish = round((toc_stop - tic_start), 2)
	
	print("Imputation Finished!")
	print("Wall Time Used: {} seconds".format(time_finish))