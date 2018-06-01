#!/usr/bin/python
print('''plot the hidden nodes: 
    the overall variance distribution of hidden nodes
    the histogram of the activations of specific hidden nodes
     ''')

import numpy as np
import pandas
import scimpute

# input data
arr = np.load('step2/code_neck_valid.step2.npy')
print('code.shape:', arr.shape)

# Variation of each node
var = np.var(arr, 0)
# Hist
scimpute.hist_arr_flat(var, title='Hist var(code)', xlab='variance of each node',
                       ylab='frequency')

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

a_top1 = arr[:, index_decreasing[0]]
a_top2 = arr[:, index_decreasing[1]]
scimpute.scatterplot2(a_top1, a_top2, title="top1 vs top2", xlab='node_var_top1', ylab='node_var_top2')
# a random node
i = 100
scimpute.hist_arr_flat(arr[:, i], title='activation ' + str(i), xlab='activation value', ylab='frequency')