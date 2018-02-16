#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import importlib
import scimpute

# info
cwd = os.getcwd()
print('cwd: ', cwd)

# READ CMD
print('reads matrix.hd5, then analysis the result')
print('usage: python -u result_analysis.py params.py')

if len(sys.argv) == 2:
    param_file = sys.argv[1]
    param_file = param_file.rstrip('.py')
    p = importlib.import_module(param_file)
else:
    raise Exception('cmd err')


# READ DATA
print("> READ DATA..")
H = scimpute.read_data_into_cell_row(p.file_h, p.file_h_ori)

# Data Transformation for H
print('> DATA TRANSFORMATION..')
H = scimpute.df_transformation(H.transpose(), transformation=p.data_transformation).transpose()

# TEST MODE OR NOT
test_flag = 0
m = 100
n = 200
if test_flag > 0:
    print('in test mode')
    H = H.ix[0:m, 0:n]

# INPUT SUMMARY
print('\ninside this code, matrices are supposed to be transformed into cell_row')
print('H:', p.file_h, p.file_h_ori, p.data_transformation, '\n', H.ix[0:3, 0:2])
print('H.shape', H.shape)


# HIST OF H
scimpute.hist_df(H, title='H({})'.format(p.file_h), dir=p.tag)

#  VISUALIZATION OF DFS, todo clustering based on H
print('\n> Visualization of dfs')
max, min = scimpute.max_min_element_in_arrs([H.values])
scimpute.heatmap_vis(H.values,
                     title='H ({})'.format(p.file_h),
                     xlab='genes',
                     ylab='cells', vmax=max, vmin=min,
                     dir=p.tag)


# Gene-Gene in M, X, H
print('\n> Gene-gene relationship (H, X, M), before/after inference')
gene_pair_dir = p.tag+'/pairs'
List = p.pair_list
scimpute.gene_pair_plot(H, list=List, tag=p.tag, dir=gene_pair_dir)
