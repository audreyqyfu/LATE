import matplotlib
matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt
import time
import os
import math
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import tensorflow as tf
import collections
import scipy.sparse as sp_sparse
import tables
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE  # single core
from MulticoreTSNE import MulticoreTSNE as TSNE  # MCORE

# Sys
def usage():
    process = psutil.Process(os.getpid())
    ram = process.memory_info()[0] / float(2 ** 20)
    ram = round(ram, 1)
    return ram


# DATA I/O # todo: check gene_id barcode uniqueness
def read_csv(fname):
    '''read_csv into pd.df, assuming index_col=0, and header=True'''
    print('reading ', fname)
    tic = time.time()
    df = pd.read_csv(fname, index_col=0)
    # print("read matrix: [genes, cells]")
    print('shape:', df.shape)
    # print(df.axes)
    if df.shape[0] > 2 and df.shape[1] > 2:
        print(df.ix[0:3, 0:2])
    toc = time.time()
    print("reading took {:.1f} seconds".format(toc - tic))
    return df

def read_tsv(fname):
    '''read_csv into pd.df, assuming index_col=0, and header=True'''
    print('reading ', fname)
    tic = time.time()
    df = pd.read_csv(fname, index_col=0, delimiter='\t')
    # print("read matrix: [genes, cells]")
    print('shape:', df.shape)
    # print(df.axes)
    if df.shape[0] > 2 and df.shape[1] > 2:
        print(df.ix[0:3, 0:2])
    toc = time.time()
    print("reading took {:.1f} seconds".format(toc - tic))
    return df

def save_csv(arr, fname):
    '''if fname=x.csv.gz, will be compressed
    if fname=x.csv, will not be compressed'''
    tic = time.time()
    print('saving: ', arr.shape)
    np.savetxt(fname, arr, delimiter=',', newline='\n')
    toc = time.time()
    print("saving" + fname + " took {:.1f} seconds".format(toc - tic))


def save_hd5(df, out_name):
    tic = time.time()
    print('saving: ', df.shape)
    df.to_hdf(out_name, key='null', mode='w', complevel=9, complib='blosc')
    toc = time.time()
    print("saving" + out_name + " took {:.1f} seconds".format(toc - tic))


def read_hd5(in_name):
    '''
    :param in_name: 
    :return df: 
    '''
    print('reading: ', in_name)
    df = pd.read_hdf(in_name)
    print('read', df.shape)
    # print(df.axes)
    if df.shape[0] > 2 and df.shape[1] > 2:
        print(df.ix[0:3, 0:2])
    return df


GeneBCMatrix = collections.namedtuple(
    'GeneBCMatrix',
    ['gene_ids', 'gene_names', 'barcodes', 'matrix'])


def read_sparse_matrix_from_h5(fname, genome, file_ori):
    '''
    for 10x_genomics h5 file:
    always transpose into cell_row if gene_row is the input
    https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices
    
    :return: cell_row sparse matrix
    :param fname: 
    :param genome: 
    :return: 
    '''
    tic = time.time()
    print('reading {} {}'.format(fname, genome))
    with tables.open_file(fname, 'r') as f:
        try:
            dsets = {}
            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()
            matrix = sp_sparse.csc_matrix(
                (dsets['data'], dsets['indices'], dsets['indptr']),
                shape=dsets['shape'])
            print('shape is {}'.format(matrix.shape))

            if file_ori == 'cell_row':
                pass
            elif file_ori == 'gene_row':
                matrix = matrix.transpose()
            else:
                raise Exception('file orientation {} not recognized'.format(file_ori))
            obj = GeneBCMatrix(dsets['genes'], dsets['gene_names'],
                                dsets['barcodes'], matrix)
            nz_count = len(obj.matrix.nonzero()[0])
            nz_rate = nz_count / (obj.matrix.shape[0] * obj.matrix.shape[1])
            nz_rate = round(nz_rate, 3)
            print('nz_rate is {}'.format(nz_rate))
            print('nz_count is {}\n'.format(nz_count))
            toc = time.time()
            print("reading took {:.1f} seconds".format(toc - tic))
            return obj
        except tables.NoSuchNodeError:
            raise Exception("Genome %s does not exist in this file." % genome)
        except KeyError:
            raise Exception("File is missing one or more required datasets.")


def save_sparse_matrix_to_h5(gbm, filename, genome):
    '''
    for 10x_genomics h5 file:
    https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices

    :return: 
    :param filename: 
    :param genome: 
    :return: 
    '''
    flt = tables.Filters(complevel=1)
    print('saving: ', gbm.matrix.shape)
    with tables.open_file(filename, 'w', filters=flt) as f:
        try:
            group = f.create_group(f.root, genome)
            f.create_carray(group, 'genes', obj=gbm.gene_ids)
            f.create_carray(group, 'gene_names', obj=gbm.gene_names)
            f.create_carray(group, 'barcodes', obj=gbm.barcodes)
            f.create_carray(group, 'data', obj=gbm.matrix.data)
            f.create_carray(group, 'indices', obj=gbm.matrix.indices)
            f.create_carray(group, 'indptr', obj=gbm.matrix.indptr)
            f.create_carray(group, 'shape', obj=gbm.matrix.shape)
        except:
            raise Exception("Failed to write H5 file.")


def read_data_into_cell_row(fname, orientation='cell_row', genome='mm10'):
    '''
    read hd5 or csv, into cell_row format
    :param fname: 
    :param orientation: of file
    :return: cell_row df
    '''
    tic = time.time()
    print('reading {} into cell_row data frame'.format(fname))
    if fname.endswith('hd5'):
        df_tmp = read_hd5(fname)
    elif fname.endswith('csv'):
        df_tmp = read_csv(fname)
    elif fname.endswith('tsv'):
        df_tmp = read_tsv(fname)
    elif fname.endswith('csv.gz'):
        df_tmp = read_csv(fname)
    elif fname.endswith('h5'):  # not hd5
        df_tmp = read_sparse_matrix_from_h5(fname, genome=genome, file_ori=orientation)
        print('sparse_matrix have been read')
    else:
        raise Exception('file name not ending in hd5 nor csv, not recognized')

    if orientation == 'gene_row':
        df_tmp = df_tmp.transpose()
    elif orientation == 'cell_row':
        pass
    else:
        raise Exception('parameter err: for {}, orientation {} not correctly spelled'.format(fname, orientation))

    #print("after transpose into cell row (if correct file_orientation provided)")
    if fname.endswith('h5'):
        print("shape is {}".format(df_tmp.matrix.shape))
    else:
        print("shape is {}".format(df_tmp.shape))
        print('nz_rate is {}'.format(nnzero_rate_df(df_tmp)))
        print('nz_count is {}\n'.format(nnzero_count_df(df_tmp)))
    toc = time.time()
    print("reading took {:.1f} seconds".format(toc - tic))
    return df_tmp


# PRE-PROCESSING OF DATA FRAMES #
def df_filter(df):
    df_filtered = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]
    print("filtered out any rows and columns with sum of zero")
    return df_filtered


def df_normalization(df, scale=1e6):
    '''
    RPM when default
    :param df: [gene, cell]
    :param scale: 
    :return: 
    '''
    read_counts = df.sum(axis=0)  # colsum
    # df_normalized = df.div(read_counts, axis=1).mul(np.median(read_counts)).mul(1)
    df_normalized = df.div(read_counts, axis=1).mul(scale)
    return df_normalized


def df_log10_transformation(df, pseudocount=1):
    '''
    log10
    :param df: 
    :param pseudocount: 
    :return: 
    '''
    df_log10 = np.log10(np.add(df, pseudocount))
    return df_log10


def df_rpm_log10(df, pseudocount=1):
    '''
    log10
    :param df: [gene, cell]
    :return: 
    '''
    df_tmp = df.copy()
    df_tmp = df_normalization(df_tmp)
    df_tmp = df_log10_transformation(df_tmp, pseudocount=pseudocount)
    return df_tmp


def df_exp_rpm_log10(df, pseudocount=1):
    '''
    log10
    :param df: [gene, cell]
    :pseudocount: for exp transformation and log10 transformation
    :return: 
    '''
    df_tmp = df.copy()
    df_tmp = np.power(10, df_tmp) - pseudocount
    df_tmp = df_normalization(df_tmp)
    df_tmp = df_log10_transformation(df_tmp, pseudocount=pseudocount)
    return df_tmp


def df_exp_discretize_log10(df, pseudocount=1):
    '''
    For better comparison with ground-truth in gene-scatterplot visualization
    Input should be the output of df_log10_transformation (log10(x+1))
    If so, all values ≥ 0
    1. 10^x-1
    2. arount
    3. log10(x+1)
    :param df: 
    :param pseudocount: 
    :return: 
    '''
    df_tmp = df.copy()
    df_tmp = np.around(np.power(10, df_tmp) - pseudocount)
    df_tmp = np.log10(df_tmp + pseudocount)
    return df_tmp


def df_transformation(df, transformation='as_is'):
    '''
    data_transformation
    df not copied
    :param df: [genes, cells]
    :param format: as_is, log10, rpm_log10, exp_rpm_log10
    :return: df_formatted
    '''
    if transformation == 'as_is':
        pass  # do nothing
    elif transformation == 'log10':
        df = df_log10_transformation(df)
    elif transformation == 'rpm_log10':
        df = df_rpm_log10(df)
    elif transformation == 'exp_rpm_log10':
        df == df_exp_rpm_log10(df)
    else:
        raise Exception('format {} not recognized'.format(transformation))

    print('data formatting: ', transformation)
    return df


def mask_df(df, nz_goal):
    '''
    
    :param df: any direction
    :param nz_goal: 
    :return: 
    '''
    df_msked = df.copy()
    nz_now = nnzero_rate_df(df)
    nz_goal = nz_goal/nz_now
    zero_goal = 1-nz_goal
    df_msked = df_msked.where(np.random.uniform(size=df.shape) > zero_goal, 0)
    return df_msked


def multinormial_downsampling(in_df, libsize_out):
    out_df = in_df.copy()
    for i in range(len(in_df)):
        slice_arr = in_df.values[i, :]
        libsize = slice_arr.sum()
        p_lst = slice_arr / libsize
        slice_resample = np.random.multinomial(libsize_out, p_lst)
        out_df.ix[i, :] = slice_resample
    return out_df


def split_arr(arr, a=0.8, b=0.1, c=0.1, seed_var=1):
    """input array, output rand split arrays
    a: train, b: valid, c: test
    e.g.: [arr_train, arr_valid, arr_test] = split(df.values)"""
    print(">splitting data")
    np.random.seed(seed_var)  # for splitting consistency
    train_indices = np.random.choice(arr.shape[0], int(round(arr.shape[0] * a // (a + b + c))), replace=False)
    remain_indices = np.array(list(set(range(arr.shape[0])) - set(train_indices)))
    valid_indices = np.random.choice(remain_indices, int(round(len(remain_indices) * b // (b + c))), replace=False)
    test_indices = np.array(list(set(remain_indices) - set(valid_indices)))
    np.random.seed()  # cancel seed effect
    print("total samples being split: ", len(train_indices) + len(valid_indices) + len(test_indices))
    print('train:', len(train_indices), ' valid:', len(valid_indices), 'test:', len(test_indices))

    arr_train = arr[train_indices]
    arr_valid = arr[valid_indices]
    arr_test = arr[test_indices]

    return (arr_train, arr_valid, arr_test)


def split_df(df, a=0.8, b=0.1, c=0.1, seed_var=1):
    """input df, output rand split dfs
    a: train, b: valid, c: test
    e.g.: [df_train, df2, df_test] = split(df, a=0.7, b=0.15, c=0.15)"""
    np.random.seed(seed_var)  # for splitting consistency
    train_indices = np.random.choice(df.shape[0], int(df.shape[0] * a // (a + b + c)), replace=False)
    remain_indices = np.array(list(set(range(df.shape[0])) - set(train_indices)))
    valid_indices = np.random.choice(remain_indices, int(len(remain_indices) * b // (b + c)), replace=False)
    test_indices = np.array(list(set(remain_indices) - set(valid_indices)))
    np.random.seed()  # cancel seed effect
    print("total samples being split: ", len(train_indices) + len(valid_indices) + len(test_indices))
    print('train:', len(train_indices), ' valid:', len(valid_indices), 'test:', len(test_indices))

    df_train = df.ix[train_indices, :]
    df_valid = df.ix[valid_indices, :]
    df_test = df.ix[test_indices, :]

    return df_train, df_valid, df_test


def random_subset_arr(arr, m_max, n_max):
    [m, n] = arr.shape
    m_reduce = min(m, m_max)
    n_reduce = min(n, n_max)
    np.random.seed(1201)
    row_rand_idx = np.random.choice(m, m_reduce, replace=False)
    col_rand_idx = np.random.choice(n, n_reduce, replace=False)
    np.random.seed()
    arr_sub = arr[row_rand_idx][:, col_rand_idx]
    print('matrix from [{},{}] to a random subset of [{},{}]'.
          format(m, n, arr_sub.shape[0], arr_sub.shape[1]))
    return arr_sub


def subset_df(df_big, df_subset):
    return (df_big.ix[df_subset.index, df_subset.columns])


def sparse_matrix_transformation(csr_matrix, transformation='log10'):
    '''
    data_transformation
    df not copied
    :param csr_matrix: 
    :param transformation: as_is, log10
    :return: 
    '''
    if transformation == 'as_is':
        pass  # do nothing
    elif transformation == 'log10':
        csr_matrix = csr_matrix.log1p()
    elif transformation == 'rpm_log10':
        raise Exception('rpm_log10 not implemented yet')
    elif transformation == 'exp_rpm_log10':
        raise Exception('exp_rpm_log10 not implemented yet')
    else:
        raise Exception('format {} not recognized'.format(transformation))

    print('data tranformation: ', transformation)
    return csr_matrix


def subsample_matrix(gbm, barcode_indices):
    return GeneBCMatrix(gbm.gene_ids, gbm.gene_names,
                        gbm.barcodes[barcode_indices],
                        gbm.matrix[:, barcode_indices])


def subgene_matrix(gbm, gene_indices):
    return GeneBCMatrix(gbm.gene_ids[gene_indices], gbm.gene_names[gene_indices],
                        gbm.barcodes,
                        gbm.matrix[gene_indices, :])


def get_expression(gbm, gene_name):
    gene_indices = np.where(gbm.gene_names == gene_name)[0]
    if len(gene_indices) == 0:
        raise Exception("%s was not found in list of gene names." % gene_name)
    return gbm.matrix[gene_indices[0], :].toarray().squeeze()


def split__csr_matrix(csr_matrix, a=0.8, b=0.1, c=0.1, seed_var=1):
    """
    input: csr_matrix(cell_row), 
    output: rand split datasets (train/valid/test)
    a: train
    b: valid
    c: test
    e.g. [csr_train, csr_valid, csr_test] = split(df.values)"""
    print(">splitting data..")
    np.random.seed(seed_var)  # for splitting consistency
    [m, n] = csr_matrix.shape
    train_indices = np.random.choice(m, int(round(m*a//(a+b+c))), replace=False)
    remain_indices = np.array(list(set(range(m)) - set(train_indices)))
    valid_indices = np.random.choice(remain_indices, int(round(len(remain_indices)*b//(b + c))), replace=False)
    test_indices = np.array(list(set(remain_indices) - set(valid_indices)))
    np.random.seed()  # cancel seed effect
    print("total samples being split: ", len(train_indices) + len(valid_indices) + len(test_indices))
    print('train:', len(train_indices), ' valid:', len(valid_indices), 'test:', len(test_indices))

    csr_train = csr_matrix[train_indices, :]
    csr_valid = csr_matrix[valid_indices, :]
    csr_test = csr_matrix[test_indices, :]

    return (csr_train, csr_valid, csr_test, train_indices, valid_indices, test_indices)



# STAT CALCULATION #
def nnzero_rate_df(df):
    idx = df != 0
    nnzero_rate = round(sum(sum(idx.values)) / df.size, 3)
    return nnzero_rate


def nnzero_count_df(df):
    idx = df != 0
    nnzero_count = sum(sum(idx.values))
    return nnzero_count


def mean_df(df):
    Sum = sum(sum(df.values))
    Mean = Sum / df.size
    return (Mean)


def square_err(arr1, arr2):
    '''
    arr1 and arr2 of same shape, return squared err between them
    arr and df both works
    '''
    diff = np.subtract(arr1, arr2)
    square_err_ = np.sum(np.power(diff, 2))
    count = int(arr1.shape[0] * arr1.shape[1])
    return square_err_, count


def square_err_omega(arr, arr_ground_truth):
    '''
    input: arr and arr_ground_truth of same shape
    return: squared err omega (excluding zeros in ground truth)
    arr and df both works
    only zeros are ignored, negatives should not show up
    '''
    omega = np.sign(arr_ground_truth)
    diff = np.subtract(arr, arr_ground_truth)
    square_err_ = np.power(diff, 2)
    square_err_nz = np.sum(np.multiply(square_err_, omega))
    count = int(arr.shape[0] * arr.shape[1])
    return square_err_nz, count


def mse_omega(arr_h, arr_m):
    '''arr and df both works'''
    omega = np.sign(arr_m)  # if x>0, 1; elif x == 0, 0;
    diff = np.subtract(arr_h, arr_m)
    squared = np.power(diff, 2)
    non_zero_squared = np.multiply(squared, omega)
    mse_omega = np.mean(np.mean(non_zero_squared))
    return mse_omega


def mse(arr_h, arr_m):
    '''MSE between H and M'''
    diff = np.subtract(arr_h, arr_m)
    squared = np.power(diff, 2)
    mse = np.mean(np.mean(squared))
    return mse


def nz_std(X, Y):
    '''
    Goal: Evaluate gene-level imputation with STD of non-zero values of that gene
    Takes two cell_row DFs, X and Y, with same shape
    Calculate STD for each column(gene)
    Treating zeros in X as Nones, And corresponding values in Y as Nones, too
    :param X: Input cell_row matrix
    :param Y: Imputation cell_row matrix
    :return: two list of NZ_STDs, used for evaluation of imputation
    '''
    idx_zeros = (X == 0)
    X_ = X.copy()
    Y_ = Y.copy()
    X_[idx_zeros] = None
    Y_[idx_zeros] = None
    return (X_.std(), Y_.std())


def nz2_corr(x, y):
    '''
    the nz2_corr between two vectors, excluding any element with zero in either vectors
    :param x: vector1
    :param y: vector2
    :return: 
    '''
    nas = np.logical_or(x == 0, y == 0)
    result = pearson_cor(x[~nas], y[~nas])
    if not math.isnan(result):
        result = round(result, 4)
    return result


def gene_mse_nz_from_df(Y, X):
    '''
    get gene_mse from gene_expression_df (cell_row, with cell_id as index)
    X: input/ground-truth
    Y: imputation
    return a [gene, 1] pd.series with index of gene_ids 
    '''
    mse_df = pd.DataFrame(columns=['gene_name'])
    for i in range(X.shape[1]):
        mse_ = scimpute.mse_omega(Y.iloc[:, i], X.iloc[:, i])
        gene_name = X.columns[i]
        mse_df.loc[X.columns[i], 'gene_name']= mse_
    mse_df = mse_df.iloc[:, 0]
    print(mse_df.head(), '\n', mse_df.shape)
    return mse_df


def combine_gene_imputation_of_two_df(Y1, Y2, metric1, metric2, mode='smaller'):
    '''
    Y1, Y2: two imputation results (cell_row, df)
    Metric1, Metric2: [num-gene, 1], df, same metircs for Y1 and Y2, e.g. MSE, SD
    select rows of Y1, Y2 into Y_combined
    mode: smaller/larger (being selected), e.g. smaller MSE, larger SD
    Output in index/column order of Y1
    '''
    if mode == 'smaller':
        idx_better = metric1 < metric2
    elif mode == 'larger':
        idx_better = metric1 > metric2
    else:
        raise Exception('mode err')
    # try:
    #         idx_better = idx_better.iloc[:, 0]  # df to series, important
    #     except 'IndexingError':
    #         pass
    print('yg_better boolean series:\n', idx_better.head())

    Y_better_lst = [Y1.transpose()[idx_better],
                    Y2.transpose()[~idx_better]]  # list of frames
    Y_better = pd.concat(Y_better_lst)
    Y_better = Y_better.transpose()  # tr back
    Y_better = Y_better.loc[
        Y1.index, Y1.columns]  # get Y1 original order, just in case

    print('Y1:\n', Y1.iloc[:5, :3])
    print('Y2:\n', Y2.iloc[:5, :3])
    print("metrics1:\n", metric1.iloc[:5])
    print("metrics2:\n", metric2.iloc[:5])
    print('Y_combined:\n', Y_better.iloc[:5, :3])

    return Y_better


# PLOTS #
def refresh_logfolder(log_dir):
    '''delete and recreate log_dir'''
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
        print(log_dir, "deleted")
    tf.gfile.MakeDirs(log_dir)
    print(log_dir, 'created\n')


def max_min_element_in_arrs(arr_list):
    '''input a list of np.arrays
    e.g: max_element_in_arrs([df_valid.values, h_valid])'''
    max_list = []
    for x in arr_list:
        max_tmp = np.nanmax(x)
        max_list.append(max_tmp)
    max_all = np.nanmax(max_list)

    min_list = []
    for x in arr_list:
        min_tmp = np.nanmin(x)
        min_list.append(min_tmp)
    min_all = np.nanmin(min_list)

    return max_all, min_all


def scatterplot(x, y,
                title='scatterplot', dir='plots', xlab='xlab', ylab='ylab',
                alpha=1):
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = "./{}/{}".format(dir, title)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(x, y, 'o', alpha=alpha)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('heatmap vis ', title, ' done')


def scatterplot2(x, y, title='title', xlabel='x', ylabel='y', range='same', dir='plots'):
    '''
    x is slice, y is a slice
    have to be slice to help pearsonr(x,y)[0] work
    range= same/flexible

    :param x: 
    :param y: 
    :param title: 
    :param xlabel: 
    :param ylabel: 
    :param range: 
    :param dir: 
    :param corr: 
    :return: 
    '''
    # create plots directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    fprefix = "./{}/{}".format(dir, title)
    # corr
    corr = pearson_cor(x, y)
    if not math.isnan(corr):
        corr = str(round(corr, 4))
    # nz2_corr
    nz_corr = nz2_corr(x, y)
    
    print('corr: {}; nz_corr: {}'.format(corr, nz_corr))
    
    # density plot
    from scipy.stats import gaussian_kde
    # Calculate the point density
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
        # sort: dense on top (plotted last)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        # plt
        fig = plt.figure(figsize=(5, 5))
        fig, ax = plt.subplots()
        cax = ax.scatter(x, y, c=z, s=50, edgecolor='')
        plt.colorbar(cax)
    except np.linalg.linalg.LinAlgError:
        plt.plot(x, y, 'b.', alpha=0.3)

    plt.title('{}\ncorr: {}; corr-nz: {}'.format(title, corr, nz_corr))  # nz2
    plt.xlabel(xlabel + "\nmean: " + str(round(np.mean(x), 2)))
    plt.ylabel(ylabel + "\nmean: " + str(round(np.mean(y), 2)))

    if range is 'same':
        max, min = max_min_element_in_arrs([x, y])
        plt.xlim(min, max)
        plt.ylim(min, max)
    elif range is 'flexible':
        next
    else:
        plt.xlim(range[0], range[1])
        plt.ylim(range[0], range[1])

    plt.savefig(fprefix + '.png', bbox_inches='tight')
    plt.close('all')


def density_plot(x, y,
                 title='density plot', dir='plots', xlab='x', ylab='y'):
    '''x and y must be arr [m, 1]'''
    from scipy.stats import gaussian_kde
    # create plots directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = "./{}/{}".format(dir, title)
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # sort: dense on top (plotted last)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    # plt
    fig = plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots()
    cax = ax.scatter(x, y, c=z, s=50, edgecolor='')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar(cax)
    plt.savefig(fname + ".png", bbox_inches='tight')
    plt.close(fig)


def gene_pair_plot(df, list, tag, dir='./plots'):
    '''
    scatterplot2 of two genes in a df
    :param df: [cells, genes]
    :param list: [2, 3] OR [id_i, id_j]
    :param tag: output_tag e.g. 'PBMC'
    :param dir: output_dir
    :return: 
    '''
    for i, j in list:
        print('gene_pair: ', i, type(i), j, type(j))
        try:
            x = df.ix[:, i]
            y = df.ix[:, j]
        except KeyError:
            print('KeyError: the gene index does not exist')
            continue

        scatterplot2(x, y,
                     title='Gene' + str(i) + ' vs Gene' + str(j) + '\n' + tag,
                     xlabel='Gene' + str(i), ylabel='Gene' + str(j),
                     dir=dir)


def cluster_scatterplot(df2d, labels, title):
    '''
    PCA or t-SNE 2D visualization
    
    `cluster_scatterplot(tsne_projection, cluster_info.Cluster.values.astype(int),
                    title='projection.csv t-SNE')`
                    
    :param df2d: PCA or t-SNE projection df, cell as row, feature as columns
    :param labels: 
    :param title: 
    :return: 
    '''
    legends = np.unique(labels)
    print('all labels:', legends)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    for i in legends:
        _ = df2d.iloc[labels == i]
        num_cells = str(len(_))
        percent_cells = str(round(int(num_cells) / len(df2d) * 100, 1)) + '%'
        ax.scatter(_.iloc[:, 0], _.iloc[:, 1],
                   alpha=0.5, marker='.',
                   label='c' + str(i) + ':' + num_cells + ', ' + percent_cells
                   )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel('legend format:  cluster_id:num-cells')

    plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()
    plt.close('all')


def pca_tsne(df_cell_row, cluster_info=None, title='data', dir='plots',
             num_pc=50, num_tsne=2, ncores=8):
    '''
    PCA and tSNE plots for DF_cell_row, save projections.csv
    :param df_cell_row: data matrix, features as columns, e.g. [cell, gene] 
    :param cluster_info: cluster_id for each cell_id
    :param title: figure title, e.g. Late
    :param num_pc: 50
    :param num_tsne: 2
    :return: tsne_df, plots saved, pc_projection.csv, tsne_projection.csv saved
    '''

    if not os.path.exists(dir):
        os.makedirs(dir)

    title = './'+dir+'/'+title

    df = df_cell_row
    if cluster_info is None:
        cluster_info = pd.DataFrame(0, index=df.index, columns=['cluster_id'])

    tic = time.time()
    # PCA
    pca = PCA(n_components=num_pc)
    pc_x = pca.fit_transform(df)
    df_pc_df = pd.DataFrame(data=pc_x, index=df.index, columns=range(num_pc))
    df_pc_df.index.name = 'cell_id'
    df_pc_df.columns.name = 'PC'
    df_pc_df.to_csv(title+'.pca.csv')
    print('dim before PCA', df.shape)
    print('dim after PCA', df_pc_df.shape)
    print('explained variance ratio: {}'.format(
        sum(pca.explained_variance_ratio_)))

    colors = cluster_info.reindex(df_pc_df.index)
    colors = colors.dropna().iloc[:, 0]
    print('matched cluster_info:', colors.shape)
    print('unmatched data will be excluded from the plot')  # todo: include unmatched

    df_pc_ = df_pc_df.reindex(colors.index)  # only plot labeled data?
    cluster_scatterplot(df_pc_, colors.values.astype(str), title=title+' (PCA)')

    # tSNE
    print('MCORE-TSNE, with ', ncores, ' cores')
    df_tsne = TSNE(n_components=num_tsne, n_jobs=ncores).fit_transform(df_pc_)
    print('tsne done')
    df_tsne_df = pd.DataFrame(data=df_tsne, index=df_pc_.index)
    print('wait to output tsne')
    df_tsne_df.to_csv(title+'.tsne.csv')
    print('wrote tsne to output')
    cluster_scatterplot(df_tsne_df, colors.values.astype(str), title=title+' ('
                                                                           't-SNE)')
    toc = time.time()
    print('PCA and tSNE took {:.1f} seconds\n'.format(toc-tic))

    return df_tsne_df


def heatmap_vis(arr, title='visualization of matrix in a square manner', cmap="rainbow",
                vmin=None, vmax=None, xlab='', ylab='', dir='plots'):
    '''heatmap visualization of 2D matrix, with plt.imshow(), in a square manner
    cmap options PiYG for [neg, 0, posi]
    Greys Reds for [0, max]
    rainbow for [0,middle,max]'''
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = './' + dir + '/' + title + '.vis.png'

    if (vmin is None):
        vmin = np.min(arr)
    if (vmax is None):
        vmax = np.max(arr)

    fig = plt.figure(figsize=(9, 9))
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar()
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('heatmap vis ', title, ' done')


def heatmap_vis2(arr, title='visualization of matrix', cmap="rainbow",
                 vmin=None, vmax=None, xlab='', ylab='', dir='plots'):
    '''heatmap visualization of 2D matrix, with plt.pcolor()
    cmap options PiYG for [neg, 0, posi]
    Greys Reds for [0, max]
    rainbow for [0,middle,max]'''
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = './' + dir + '/' + title + '.vis.png'

    if (vmin is None):
        vmin = np.min(arr)
    if (vmax is None):
        vmax = np.max(arr)

    fig = plt.figure(figsize=(9, 9))
    plt.pcolor(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar()
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('heatmap vis ', title, ' done')


def curveplot(x, y, title, xlabel, ylabel, dir='plots'):
    # scimpute.curveplot(epoch_log, corr_log_valid,
    #                      title='learning_curve_pearsonr.step2.gene'+str(j)+", valid",
    #                      xlabel='epoch',
    #                      ylabel='Pearson corr (predction vs ground truth, valid, including cells with zero gene-j)')
    # create plots directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    fprefix = "./{}/{}".format(dir, title)
    # plot
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fprefix + '.png', bbox_inches='tight')
    plt.close()


def curveplot2(x, y, z, title, xlabel, ylabel, dir='plots'):
    '''curveplot2(epoch_log, train_log, valid_log, title="t", xlabel="x", ylabel="y")'''
    # scimpute.curveplot2(epoch_log, corr_log_train, corr_log_valid,
    #                      title='learning_curve_pearsonr.step2.gene'+str(j)+", train_valid",
    #                      xlabel='epoch',
    #                      ylabel='Pearson corr (predction vs ground truth, valid, including cells with zero gene-j)')
    # create plots directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    fprefix = "./{}/{}".format(dir, title)
    # plot
    plt.plot(x, y, label='train')
    plt.plot(x, z, label='valid')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fprefix + '.png', bbox_inches='tight')
    plt.close()


def hist_list(list, xlab='xlab', title='histogram', bins=100, dir='plots'):
    '''output histogram of a list into png'''
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = str(title) + '.png'
    fname = "./{}/{}".format(dir, fname)
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel('Density')
    hist = plt.hist(list, bins=bins, density=True)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('hist of {} is done'.format(title))
    return hist


def hist_arr_flat(arr, title='hist', xlab='x', ylab='Frequency', bins=100, dir='plots'):
    '''create histogram for flattened arr'''
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = "./{}/{}".format(dir, title) + '.png'

    fig = plt.figure(figsize=(9, 9))
    n, bins, patches = plt.hist(arr.flatten(), bins, normed=1, facecolor='green', alpha=0.75)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print("histogram ", title, ' done')


def hist_df(df, title="hist of df", xlab='xlab', bins=100, dir='plots', range=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    df_flat = df.values.reshape(df.size, 1)
    # fig = plt.figure(figsize=(9, 9))
    hist = plt.hist(df_flat, bins=bins, density=True, range=range)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel('Density')
    plt.savefig('./{}/{}.png'.format(dir, title), bbox_inches='tight')
    plt.close()
    print('hist of ', title, 'is done')
    return hist


def pearson_cor (x, y):
    '''This function calculates Pearson correlation between vector x and y.
    It returns nan if x or y has 2 data points or less, or does not vary
            
	Parameters
	------------
		x: numpy array
		y: numpy array
                
	Return
	-----------
		Pearson correlation or nan
	'''
    if (len(x) > 2) and (x.std() > 0) and (y.std() > 0):
        corr = pearsonr(x, y)[0]
    else:
        corr = np.nan

    return corr


def hist_2matrix_corr(arr1, arr2, mode='column-wise', nz_mode='ignore',
                   title='hist_corr', dir='plots'):
    '''Calculate correlation between two matrices column-wise or row-wise
    default: arr[cells, genes], gene-wise corr (column-wise)
    assume: arr1 from benchmark matrix (e.g. input), arr2 from imputation
    if corr = NaN, it will be excluded from result 
    
    mode: column-wise, row-wise
    nz_mode: 
        ignore (all values in vectors included)
        strict (zero values excluded from both vector x,y)
        first (zero values excluded from x in arr1 only, 
    title: 'hist_corr' or custom
    dir: 'plots' or custom
    '''
    # create plots directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    fprefix = "./{}/{}".format(dir, title)

    # if arr1.shape is arr2.shape:
    if mode == 'column-wise':
        range_size = arr2.shape[1]
    elif mode == 'row-wise':
        range_size = arr2.shape[0]
    else:
        raise Exception('mode not recognized')

    hist = []
    for i in range(range_size):
        if mode == 'column-wise':
            x = arr1[:, i]
            y = arr2[:, i]
        elif mode == 'row-wise':
            x = arr1[i, :]
            y = arr2[i, :]
        else:
            raise Exception('mode not recognized')

    # Pearson correlation can be calculated
    # only when there are more than 2 nonzero
    # values, and when the standard deviation
    # is positive for both x and y
        if nz_mode == 'strict':
            nas = np.logical_or(x==0, y==0)
            corr = pearson_cor (x[~nas], y[~nas])
        elif nz_mode == 'first':
            nas = (x==0)
            corr = pearson_cor (x[~nas], y[~nas])
        elif nz_mode == 'ignore':
            corr = pearson_cor(x, y)
        else:
            raise Exception('nz_mode not recognized')

        if not math.isnan(corr):
            hist.append(corr)
 
    print('correlation calculation completed')
    
    hist.sort()
    median_corr = round(np.median(hist), 3)
    mean_corr = round(np.mean(hist), 3)
    print(title)
    print('median corr: {}    mean corr: {}'.format(median_corr, mean_corr))

    # histogram of correlation
    fig = plt.figure(figsize=(5, 5))
    plt.hist(hist, bins=100, density=True)
    plt.xlabel('median=' + str(median_corr) + ', mean=' + str(mean_corr))
    plt.ylabel('Density') #todo freq to density
    plt.xlim(-1, 1)
    plt.title(title)
    plt.savefig(fprefix + ".png", bbox_inches='tight') #todo remove \n from out-name
    plt.close(fig)
    return hist


# TF #
def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        # mean = tf.reduce_mean(var)
        # tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(name_scope, dim_in, dim_out, sd):
    """
    define weights
    
    :param name_scope: 
    :param dim_in: 
    :param dim_out: 
    :param sd: 
    :return: 
    """
    with tf.name_scope(name_scope):
        W = tf.Variable(tf.random_normal([dim_in, dim_out], stddev=sd),
                        name=name_scope + '_W')

    variable_summaries(name_scope + '_W', W)

    return W


def bias_variable(name_scope, dim_out, sd):
    """
    define biases

    :param name_scope: 
    :param dim_out: 
    :param sd: 
    :return: 
    """
    with tf.name_scope(name_scope):
        b = tf.Variable(tf.random_normal([dim_out], mean=100 * sd, stddev=sd),
                        name=name_scope + '_b')

    variable_summaries(name_scope + '_b', b)

    return b


def weight_bias_variable(name_scope, dim_in, dim_out, sd):
    """
    define weights and biases

    :param name_scope: 
    :param dim_in: 
    :param dim_out: 
    :param sd: 
    :return: 
    """
    with tf.name_scope(name_scope):
        W = tf.Variable(tf.random_normal([dim_in, dim_out], stddev=sd, dtype=tf.float32),
                        name=name_scope + '_W')
        b = tf.Variable(tf.random_normal([dim_out], mean=100 * sd, stddev=sd, dtype=tf.float32),
                        name=name_scope + '_b')

    variable_summaries(name_scope + '_W', W)
    variable_summaries(name_scope + '_b', b)

    return W, b


def dense_layer(name, input, W, b, pRetain):
    """
    define a layer and return output
    
    :param name: 
    :param input: X_placeholder or a(l-1)
    :param W: weights
    :param b: biases 
    :param pRetain: 
    :return: 
    """
    x_drop = tf.nn.dropout(input, pRetain)
    z = tf.add(tf.matmul(x_drop, W), b)
    a = tf.nn.relu(z)

    variable_summaries(name + '_a', a)

    return a


def dense_layer_BN(name, input, W, b, pRetain, epsilon=1e-3):
    """
    define a layer and return output

    :param name: 
    :param input: X_placeholder or a(l-1)
    :param W: weights
    :param b: biases 
    :param pRetain: 
    :return: 
    """
    x_drop = tf.nn.dropout(input, pRetain)
    z = tf.add(tf.matmul(x_drop, W), b)
    # BN
    batch_mean, batch_var = tf.nn.moments(z, [0])
    z_bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, epsilon)
    # NL
    a = tf.nn.relu(z_bn)

    variable_summaries(name + '_a', a)

    return a


def learning_curve_mse(epoch, mse_batch, mse_valid,
                       title='learning curve (MSE)', xlabel='epochs', ylabel='MSE',
                       range=None,
                       dir='plots'):
    """
    depreciated
    """

    # create plots directory
    if not os.path.exists(dir):
        os.makedirs(dir)

    # list to np.array, to use index
    epoch = np.array(epoch)
    mse_batch = np.array(mse_batch)
    # mse_train = np.array(mse_train)
    mse_valid = np.array(mse_valid)

    # plot (full range)
    fprefix = "./{}/{}".format(dir, title)
    plt.plot(epoch, mse_batch, 'b--', label='mse_batch')
    # plt.plot(epoch, mse_train, 'g--', label='mse_train')
    plt.plot(epoch, mse_valid, 'r-', label='mse_valid')
    plt.title(title)
    plt.xlabel(xlabel + '\nfinal valid mse:' + str(mse_valid[-1]))
    plt.ylabel(ylabel)
    plt.legend()
    if range is None:
        max, min = max_min_element_in_arrs([mse_batch, mse_valid])
        # max, min = max_min_element_in_arrs([mse_batch, mse_train, mse_valid])
        plt.ylim(min, max)
    else:
        plt.ylim(range[0], range[1])

    plt.savefig(fprefix + '.png', bbox_inches='tight')
    plt.close()

    # plot (no epoch0)
    fprefix = "./{}/{}".format(dir, title) + '.cropped'
    zoom = np.arange(1, len(mse_batch))
    plt.plot(epoch[zoom], mse_batch[zoom], 'b--', label='mse_batch')
    # plt.plot(epoch[zoom], mse_train[zoom], 'g--', label='mse_train')
    plt.plot(epoch[zoom], mse_valid[zoom], 'r-', label='mse_valid')
    plt.title(title)
    plt.xlabel(xlabel + '\nfinal valid mse:' + str(mse_valid[-1]))
    plt.ylabel(ylabel)
    plt.legend()
    if range is None:
        max, min = max_min_element_in_arrs([mse_batch[zoom], mse_valid[zoom]])
        # max, min = max_min_element_in_arrs([mse_batch, mse_train, mse_valid])
        plt.ylim(min, max)
    else:
        plt.ylim(range[0], range[1])

    plt.savefig(fprefix + '.png', bbox_inches='tight')
    plt.close()


def learning_curve_corr(epoch, corr_batch, corr_valid,
                        title='learning curve (corr)',
                        xlabel='epochs',
                        ylabel='median cell-corr (100 cells)',
                        range=None,
                        dir='plots'):
    """
    depreciated
    """

    # create plots directory
    if not os.path.exists(dir):
        os.makedirs(dir)

    # list to np.array, to use index
    epoch = np.array(epoch)
    corr_batch = np.array(corr_batch)
    # corr_train = np.array(corr_train)
    corr_valid = np.array(corr_valid)

    # plot (full range)
    fprefix = "./{}/{}".format(dir, title)
    plt.plot(epoch, corr_batch, 'b--', label='corr_batch')
    # plt.plot(epoch, corr_train, 'g--', label='corr_train')
    plt.plot(epoch, corr_valid, 'r-', label='corr_valid')
    plt.title(title)
    plt.xlabel(xlabel + '\nfinal valid corr:' + str(corr_valid[-1]))
    plt.ylabel(ylabel)
    plt.legend()
    if range is None:
        max, min = max_min_element_in_arrs([corr_batch, corr_valid])
        # max, min = max_min_element_in_arrs([corr_batch, corr_train, corr_valid])
        plt.ylim(min, max)
    else:
        plt.ylim(range[0], range[1])

    plt.savefig(fprefix + '.png', bbox_inches='tight')
    plt.close()

    # plot (no epoch0)
    fprefix = "./{}/{}".format(dir, title) + '.cropped'
    zoom = np.arange(1, len(corr_batch))
    plt.plot(epoch[zoom], corr_batch[zoom], 'b--', label='corr_batch')
    # plt.plot(epoch[zoom], corr_train[zoom], 'g--', label='corr_train')
    plt.plot(epoch[zoom], corr_valid[zoom], 'r-', label='corr_valid')
    plt.title(title)
    plt.xlabel(xlabel + '\nfinal valid corr:' + str(corr_valid[-1]))
    plt.ylabel(ylabel)
    plt.legend()
    if range is None:
        max, min = max_min_element_in_arrs([corr_batch[zoom], corr_valid[zoom]])
        # max, min = max_min_element_in_arrs([corr_batch, corr_train, corr_valid])
        plt.ylim(min, max)
    else:
        plt.ylim(range[0], range[1])

    plt.savefig(fprefix + '.png', bbox_inches='tight')
    plt.close()

def learning_curve(epoch, metrics_batch, metrics_valid,
                   title='Learning curve (Metrics)',
                   xlabel='epochs',
                   ylabel='Metrics',
                   range=None,
                   skip=1,
                   dir='plots'):
    '''plot learning curve
	
    :param epoch: vector
    :param metrics_batch: vector
    :param metrics_valid: vector
    :param title: 
    :param xlabel: 
    :param ylabel: 
    :param range: 
    :param dir:
    :return: 
    '''

    # create plots directory
    if not os.path.exists(dir):
        os.makedirs(dir)

    # list to np.array, to use index
    epoch = np.array(epoch)
    metrics_batch = np.array(metrics_batch)
    metrics_valid = np.array(metrics_valid)

    # plot (full range)
    fprefix = "./{}/{}".format(dir, title)
    plt.plot(epoch, metrics_batch, 'b--', label='batch')
    plt.plot(epoch, metrics_valid, 'r-', label='valid')
    plt.title(title)
    plt.xlabel(xlabel + '\nfinal valid:' + str(metrics_valid[-1]))
    plt.ylabel(ylabel)
    plt.legend()
    if range == None:
        max, min = max_min_element_in_arrs([metrics_batch, metrics_valid])
        plt.ylim(min, max)
    else:
        plt.ylim(range[0], range[1])

    plt.savefig(fprefix + '.png', bbox_inches='tight')
    plt.close()

    # plot (zoom)
    fprefix = "./{}/{}".format(dir, title) + '.cropped'
    zoom = np.arange(skip, len(metrics_batch))
    plt.plot(epoch[zoom], metrics_batch[zoom], 'b--', label='batch')
    plt.plot(epoch[zoom], metrics_valid[zoom], 'r-', label='valid')
    plt.title(title)
    plt.xlabel(xlabel + '\nfinal valid:' + str(metrics_valid[-1]))
    plt.ylabel(ylabel)
    plt.legend()
    if range == None:
        max, min = max_min_element_in_arrs([metrics_batch[zoom], metrics_valid[zoom]])
        plt.ylim(min, max)
    else:
        plt.ylim(range[0], range[1])

    plt.savefig(fprefix + '.png', bbox_inches='tight')
    plt.close()

def visualize_weights_biases(weight, bias, title, cmap='rainbow', dir='plots'):
    '''heatmap visualization of weight and bias
    weights: [1000, 500]
    bias: [1, 500]
    '''
    # https://stackoverflow.com/questions/43076488/single-row-or-column-heat-map-in-python
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = "./{}/{}".format(dir, title) + '.vis.png'

    vmax_w, vmin_w = max_min_element_in_arrs([weight])
    vmax_b, vmin_b = max_min_element_in_arrs([bias])

    norm_w = matplotlib.colors.Normalize(vmin=vmin_w, vmax=vmax_w)
    norm_b = matplotlib.colors.Normalize(vmin=vmin_b, vmax=vmax_b)

    grid = dict(height_ratios=[weight.shape[0], weight.shape[0] / 40, weight.shape[0] / 40],
                width_ratios=[weight.shape[1], weight.shape[1] / 40])
    fig, axes = plt.subplots(ncols=2, nrows=3, gridspec_kw=grid)

    axes[0, 0].imshow(weight, aspect="auto", cmap=cmap, norm=norm_w)
    axes[1, 0].imshow(bias, aspect="auto", cmap=cmap, norm=norm_b)

    for ax in [axes[1, 0]]:
        ax.set_xticks([])

    for ax in [axes[1, 0]]:
        ax.set_yticks([])

    for ax in [axes[1, 1], axes[2, 1]]:
        ax.axis("off")

    # axes[1, 0].set_xlabel('node out')
    # axes[1, 0].set_ylabel('node in')

    sm_w = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm_w)
    sm_w.set_array([])
    sm_b = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm_b)
    sm_b.set_array([])

    fig.colorbar(sm_w, cax=axes[0, 1])
    fig.colorbar(sm_b, cax=axes[2, 0], orientation="horizontal")
    # todo: add title in plot
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)


# TF: Factors Affecting Gene Prediction
def gene_mse_list(arr1, arr2):
    '''mse for each gene(column)
    arr [cells, genes]
    arr1: X
    arr2: H'''
    n = arr2.shape[1]
    list = []
    for j in range(n):
        mse = ((arr1[:, j] - arr2[:, j]) ** 2).mean()
        list.append(mse)
    list = np.array(list)
    return list


def gene_nz_rate_list(arr1):
    '''nz_rate for each gene(column)
    arr [cells, genes]
    arr1: X'''
    n = arr1.shape[1]
    list = []
    for j in range(n):
        nz_rate = np.count_nonzero(arr1[:, j]) / n
        list.append(nz_rate)
    list = np.array(list)
    return list


def gene_var_list(arr1):
    '''variation for each gene(column)
    arr [cells, genes]
    arr: X'''
    n = arr1.shape[1]
    list = []
    for j in range(n):
        var = np.var(arr1[:, j])
        list.append(var)
    list = np.array(list)
    return list


def gene_nzvar_list(arr1):
    '''variation for non-zero values in each gene(column)
    arr [cells, genes]
    arr: X'''
    n = arr1.shape[1]
    list = []
    for j in range(n):
        data = arr1[:, j]
        nz_data = data[data.nonzero()]
        var = np.var(nz_data)
        list.append(var)
    list = np.array(list)
    return list


# DEPRECIATED #
def genescatterplot(gene1, gene2, scdata):
    gene1 = str(gene1);
    gene2 = str(gene2)
    fig, ax = scdata.scatter_gene_expression([gene1, gene2])
    fig.savefig(gene1 + "_" + gene2 + '.biaxial.png')
    # after magic
    fig, ax = scdata.magic.scatter_gene_expression([gene1, gene2])
    fig.savefig(gene1 + "_" + gene2 + '.magic.biaxial.png')
    plt.close(fig)


def genescatterplot3d(gene1, gene2, gene3, scdata):
    gene1 = str(gene1);
    gene2 = str(gene2);
    gene3 = str(gene3);
    fig, ax = scdata.scatter_gene_expression([gene1, gene2, gene3])
    fig.savefig(gene1 + "_" + gene2 + "_" + gene3 + '.biaxial.png')
    # after magic
    fig, ax = scdata.magic.scatter_gene_expression([gene1, gene2, gene3])
    fig.savefig(gene1 + "_" + gene2 + "_" + gene3 + '.magic.biaxial.png')
    plt.close(fig)


def bone_marrow_biaxial_plots(scdata):
    # Gene-Gene scatter plot (before & after magic)
    # Fig3
    print("gene-gene plot for bone marrow dataset")
    genescatterplot('Cd34', 'Gypa', scdata)  # CD325a
    genescatterplot('Cd14', 'Itgam', scdata)  # cd11b
    genescatterplot('Cd34', 'Fcgr2b', scdata)  # cd32, similar plot
    genescatterplot3d('Cd34', 'Gata1', 'Gata2', scdata)
    genescatterplot3d('Cd44', 'Gypa', 'Cpox', scdata)
    genescatterplot3d('Cd34', 'Itgam', 'Cd14', scdata)
    # Fig12
    genescatterplot('Cd34', 'Itgam', scdata)
    genescatterplot('Cd34', 'Apoe', scdata)
    genescatterplot('Cd34', 'Gata1', scdata)
    genescatterplot('Cd34', 'Gata2', scdata)
    genescatterplot('Cd34', 'Ephb6', scdata)
    genescatterplot('Cd34', 'Lepre1', scdata)
    genescatterplot('Cd34', 'Mrpl44', scdata)
    genescatterplot('Cd34', 'Cnbp', scdata)
    # Fig14
    genescatterplot('Gata1', 'Gata2', scdata)
    genescatterplot('Klf1', 'Sfpi1', scdata)
    genescatterplot('Meis1', 'Cebpa', scdata)
    genescatterplot('Elane', 'Cebpe', scdata)


def read_data(data_name):
    if data_name == 'splatter':  # only this mode creates gene-gene plot
        file = "../data/v1-1-5-3/v1-1-5-3.E3.hd5"  # data need imputation
        file_benchmark = "../data/v1-1-5-3/v1-1-5-3.E3.hd5"
        name1 = '(E3)'
        name2 = '(E3)'  # careful
        df = pd.read_hdf(file).transpose()  # [cells,genes]
        df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    elif data_name == 'EMT2730':  # 2.7k cells used in magic paper
        file = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5"  # data need imputation
        file_benchmark = "../../../../data/mouse_bone_marrow/python_2730/bone_marrow_2730.norm.log.hd5"
        name1 = '(EMT2730)'
        name2 = '(EMT2730)'
        df = pd.read_hdf(file).transpose()  # [cells,genes]
        df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    elif data_name == 'EMT9k':  # magic imputation using 8.7k cells > 300 reads/cell
        file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.hd5"  # data need imputation
        file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.hd5"
        name1 = '(EMT9k)'
        name2 = '(EMT9k)'
        df = pd.read_hdf(file).transpose()  # [cells,genes]
        df2 = pd.read_hdf(file_benchmark).transpose()  # [cells,genes]
    elif data_name == 'EMT9k_log':  # magic imputation using 8.7k cells > 300 reads/cell
        file = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # data need imputation
        file_benchmark = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"
        name1 = '(EMT9kLog)'
        name2 = '(EMT9kLog)'
        df = pd.read_hdf(file).transpose()  # .ix[:, 1:1000]  # [cells,genes]
        df2 = pd.read_hdf(file_benchmark).transpose()  # .ix[:, 1:1000]  # [cells,genes]
    else:
        raise Warning("data name not recognized!")

    # df = df.ix[1:1000] # todo: for development
    # df2 = df.ix[1:1000]

    m, n = df.shape  # m: n_cells; n: n_genes
    print("\ninput df: ", name1, " ", file, "\n", df.values[0:4, 0:4], "\n")
    print("ground-truth df: ", name2, " ", file_benchmark, "\n", df2.values[0:4, 0:4], "\n")

    return (df, df2, name1, name2, m, n)
