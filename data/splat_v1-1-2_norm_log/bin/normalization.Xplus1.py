# 0630 v1: added timing, 
# 0630: from csv to pkl, save accuracy float64 (digit18), but didn't make a difference compared with digit6 
# 0701: output as compressed hd5, blosc

print("normalization.py: read csv [row=genes, col=cells], normalize similar to magic")

import pandas as pd
import numpy as np
import time
import os

def df_filter (df):
	df_filtered = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]
	print("filtered out any rows and columns with sum of zero")
	return(df_filtered)

def df_normalization (df):
	read_counts = df.sum(axis=0) #colsum
	df_normalized = df.div(read_counts, axis = 1).mul(np.median(read_counts)).mul(1)
	return(df_normalized)

def df_log_transformation (df, pseudocount=1):
	df_log = np.log10(np.add(df, pseudocount))
	return(df_log)

# input
in_name = "splat.OneGroup.csv"
out_prefix = "splat.OneGroup"

print("started reading file: " + in_name)
tic = time.time()
data = pd.io.parsers.read_csv(in_name, index_col=0) #with rownames and header (default)
print("input matrix:"); print(data.axes);
toc = time.time()
print("reading input took {:.1f} seconds".format(toc-tic))

# filter out genes and cells with no reads
tic = time.time()
data_filtered = df_filter (data)
print("filtered matrix:")
print(data_filtered.axes)
toc = time.time()
print("filtering took {:.1f} seconds".format(toc-tic))

# normalization
tic = time.time()
data_normalized = df_normalization(data_filtered)
print("normalized matrix:")
print(data_normalized.axes)
toc = time.time()
print("normalization took {:.1f} seconds".format(toc-tic))

# log transformation
tic = time.time()
data_normalized_log = df_log_transformation (data_normalized)
print("log transformed matrix:")
print(data_normalized_log.axes)
toc = time.time()
print("log-tansformation took {:.1f} seconds".format(toc-tic))

# output
print("saving output norm.hd5: " + os.getcwd())
tic = time.time()
#data_normalized.to_pickle(out_prefix+".norm.pkl")
data_normalized.to_hdf(out_prefix+'.norm.hd5', key='null', mode='w', complevel=9,complib='blosc')
toc = time.time()
print("saving " + out_prefix + ".norm.hd5 took {:.1f} seconds".format(toc-tic))

print("saving output norm.log.hd5: " + os.getcwd())
tic = time.time()
#data_normalized_log.to_pickle(out_prefix+".norm.log.pkl")
data_normalized_log.to_hdf(out_prefix+'.norm.log.hd5', key='null', mode='w', complevel=9,complib='blosc')
toc = time.time()
print("saving " + out_prefix + ".norm.log.hd5 took {:.1f} seconds".format(toc-tic))

print("saving output norm.csv.gz")
tic = time.time()
data_normalized.to_csv(out_prefix + ".norm.csv.gz", compression='gzip')
toc = time.time()
print("saving " + out_prefix + ".norm.csv.gz took {:.1f} seconds".format(toc-tic))

print("saving output norm.log.csv.gz")
tic = time.time()
data_normalized_log.to_csv(out_prefix + ".norm.log.csv.gz", compression='gzip')
toc = time.time()
print("saving " + out_prefix + ".norm.log.csv.gz took {:.1f} seconds".format(toc-tic))

