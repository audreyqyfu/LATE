#!/usr/bin/python
# load weights from npy
# matplotlib.pyplot.vis()
# seaborn clustering and plotting


import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt
import seaborn as sns



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


# read cmd
if len(sys.argv) != 3:
    print('usage: <weights_visualization.py> <w_name.npy> <out_tag>')
    print(sys.argv)
    raise Exception('cmd error')
in_name = sys.argv[1]
tag = sys.argv[2]
print('usage:', sys.argv)


# read data
arr = np.load(in_name)

[m, n] = arr.shape

# exclude saved bias files
if (m == 1 or n == 1):
    raise Exception('Not matrix, but vector, so skipped')

print('matrix sample', arr[0:2, 0:2])
print('matrix shape:', arr.shape)

# exclude large matrix
m_max = 1000
n_max = 1000
if (m > m_max or n > n_max):
    print('matrix too large, down-sample to 1000 max each dim')
    arr = random_subset_arr(arr, m_max, n_max)

# seaborn clustering (the rows are rows, columns are columns in clustmap)
heatmap = sns.clustermap(arr, method='average', cmap="summer", robust=True)
heatmap.savefig(in_name+'.'+tag+'.png', bbox_inches='tight')
print('\n\n')