#!/usr/bin/python
# load weights from npy
# matplotlib.pyplot.vis()
# seaborn clustering and plotting


import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt


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
print('matrix sample', arr[0:3, 0:3])
print('matrix shape:', arr.shape)

# seaborn clustering
heatmap = sns.clustermap(arr)
heatmap.savefig(in_name+'.'+tag+'.png', bbox_inches='tight')

