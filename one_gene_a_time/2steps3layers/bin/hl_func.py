# high level functions

def print_parameters():
    print(os.getcwd(), "\n",
          "\n# Parameters: 9L",
          "\nn_features: ", n,
          "\nn_hidden1: ", n_hidden_1,  # todo: adjust based on model
          "\nn_hidden2: ", n_hidden_2,
          "\nn_hidden3: ", n_hidden_3,
          "\nn_hidden4: ", n_hidden_4,
          "\nlearning_rate :", learning_rate,
          "\nbatch_size: ", batch_size,
          "\nepoches: ", training_epochs, "\n",
          "\npIn_holder: ", pIn,
          "\npHidden_holder: ", pHidden, "\n",
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
    corr_train = scimpute.medium_corr(df_train.values, h_train)
    corr_valid = scimpute.medium_corr(df_valid.values, h_valid)
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
    print("medium cell-pearsonr(all train): ",
          scimpute.medium_corr(df2_train.values, h_train, num=len(df_train)))
    print("medium cell-pearsonr(all valid): ",
          scimpute.medium_corr(df2_valid.values, h_valid, num=len(df_valid)))
    print("medium cell-pearsonr in all imputation cells: ",
          scimpute.medium_corr(df2.values, h_input, num=m))
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
    code_bottle_neck_input = sess.run(e_a4, feed_dict={X: df.values, pIn_holder: 1, pHidden_holder: 1})
    np.save('pre_train/code_neck_valid.npy', code_bottle_neck_input)
    # # todo: hclust, but seaborn not on server yet
    # clustermap = sns.clustermap(code_bottle_neck_input)
    # clustermap.savefig('./plots/bottle_neck.hclust.png')


def groundTruth_vs_prediction():
    print("> Ground truth vs prediction")
    for j in [4058, 7496, 8495, 12871]:  # Cd34, Gypa, Klf1, Sfpi1
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
    List = [[4058, 7496],
            [8495, 12871],
            [2, 3],
            [205, 206]
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
    weights_visualization('e_w2', 'e_b2')
    weights_visualization('d_w2', 'd_b2')
    weights_visualization('e_w3', 'e_b3')
    weights_visualization('d_w3', 'd_b3')
    weights_visualization('e_w4', 'e_b4')
    weights_visualization('d_w4', 'd_b4')


def save_weights():
    # todo: update when model changes depth
    print('save weights in csv')
    np.save('pre_train/e_w1', sess.run(e_w1))
    np.save('pre_train/d_w1', sess.run(d_w1))
    np.save('pre_train/e_w2', sess.run(e_w2))
    np.save('pre_train/d_w2', sess.run(d_w2))
    np.save('pre_train/e_w3', sess.run(e_w3))
    np.save('pre_train/d_w3', sess.run(d_w3))
    np.save('pre_train/e_w4', sess.run(e_w4))
    np.save('pre_train/d_w4', sess.run(d_w4))
    # scimpute.save_csv(sess.run(d_w2), 'pre_train/d_w2.csv.gz')


def visualization_of_dfs():
    print('visualization of dfs')
    max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid])
    # max, min = scimpute.max_min_element_in_arrs([df_valid.values, h_valid, h, df.values])
    scimpute.heatmap_vis(df_valid.values, title='df.valid'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    scimpute.heatmap_vis(h_valid, title='h.valid'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(df.values, title='df'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)
    # scimpute.heatmap_vis(h, title='h'+Aname, xlab='genes', ylab='cells', vmax=max, vmin=min)