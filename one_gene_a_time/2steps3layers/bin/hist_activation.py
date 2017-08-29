#!/usr/bin/python
print('''plot the hidden nodes: 
    the overall variance distribution of hidden nodes
    the histogram of the activations of specific hidden nodes
     ''')

import numpy as np
import pandas
import scimpute

# input data
arr = np.load('pre_train/code_neck_valid.npy')
print(arr.shape)

# calculate variance for each hidden node
var = np.var(arr, 0)

# summary
scimpute.hist_arr_flat(var, title='variance of activations', xlab='variance of each node', ylab='frequency')

# index, from small to large
index_increasing = var.argsort()
index_decreasing = index_increasing[::-1]

max_indice = index_increasing[-1]
# max_indice = np.where(var == var.max())

i = 0
for j in index_decreasing[0:5]:
    i += 1
    print(i, j)
    activations = arr[:, j]
    scimpute.hist_arr_flat(activations, title='max-variance num'+ str(i) +', node' + str(j), xlab='activation value', ylab='freq')

# a random node
i = 100
scimpute.hist_arr_flat(arr[:, i], title='activation ' + str(i), xlab='activation value', ylab='frequency')