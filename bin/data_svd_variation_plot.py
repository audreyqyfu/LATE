#!/usr/bin/python
# read gene expression matrix into df [genes, cells]
# perform randomized SVD to know the internal dimension complexity of datasets
# 'usage: <data_svd_variation_plot.py> <input.hd5> <out_prefix> <n_rank>'

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd #https://stackoverflow.com/questions/31523575/get-u-sigma-v-matrix-from-truncated-svd-in-scikit-learn
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use('Agg')  # for plotting without GUI
import matplotlib.pyplot as plt
import time
import os
import sys

# Usage
print('usage: <data_svd_variation_plot.py> <input.hd5> <out_prefix> <n_rank>')
print('default input is a gene expression matrix [cell, gene]')
print('so that df in program is [gene, cell]')
print('n_rank is the num of dims you want to check, if not specified, will be n_gene')
print(sys.argv)

if len(sys.argv) < 3 or len(sys.argv) > 4:
    raise Exception('cmd err')

file = sys.argv[1]
out_prefix = sys.argv[2]

# Read hd5 data
df = pd.read_hdf(file)  # df: [gene, cell]
print("reading complete")
print("original data matrix shape: ", df.shape, "\n")
print('sample: ', df.ix[0:3, 0:3])
print('this inner df should be [gene, cell]')

if len(sys.argv) == 4:
    n_rank = int(sys.argv[3])
elif len(sys.argv) == 3:
    n_rank = min(df.shape[0], df.shape[1]) - 1

# Folder for SVD plots
if not os.path.exists("svd_plots"):
    os.makedirs("svd_plots")

# Explained Variance ratio
print("Calculating Variance ratio with TruncatedSVD")
tic = time.clock()
svd = TruncatedSVD(n_components=n_rank)
svd.fit(df)
toc = time.clock()
print("TruncatedSVD took {:.1f} seconds\n".format(toc - tic))

print("Plotting explained_variance_ratio_: ")
# print(svd.explained_variance_ratio_[])
fig = plt.figure(figsize=(9,9))
plt.plot(svd.explained_variance_ratio_, 'b-')
plt.xlabel('rank')
plt.ylabel('explained_variance_ratio')
plt.title("explained_variance_ratio")
plt.savefig("./svd_plots/" + out_prefix + ".explained_variance_ratio.png", bbox_inches='tight')
plt.close(fig)

print("Plotting explained_variance_ratio_cumsum: ")
explained_variance_ratio_cumsum = np.cumsum(svd.explained_variance_ratio_)
x_lst = np.arange(1, n_rank+1)
print(explained_variance_ratio_cumsum)
fig = plt.figure(figsize=(9, 9))
plt.ylim((0, 1))
plt.plot(x_lst, explained_variance_ratio_cumsum, 'r-')
plt.xlabel('rank')
plt.ylabel('explained_variance_ratio_cumsum')
plt.title("explained_variance_ratio_cumsum")
plt.savefig("./svd_plots/"+ out_prefix + ".explained_variance_ratio_cumsum.png", bbox_inches='tight')
plt.close(fig)
