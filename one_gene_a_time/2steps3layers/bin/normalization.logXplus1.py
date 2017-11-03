#!/usr/bin/python
# read count.txt/csv; do MAGIC style normalization (scale back to median);
# do log (x+1) transformation
# output in blosc-compressed hd5 and csv.gz format

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
in_name = "GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct.gz"
# in_name = "test.gct.gz"
out_prefix = "gtex_v7"

print("started reading file: ", in_name)
tic = time.time()
data = pd.io.parsers.read_csv(in_name,
							  index_col=1, # col=1 as rownames (gene names)
							  sep="\t",
							  skiprows=2) #col=0 as rownames, first row as colnames(default)print("input matrix:"); print(data.ix[0:2, 0:2]);
del data['Name']  # delete gene_id
toc = time.time()
print("reading input took {:.1f} seconds".format(toc-tic))
print(data.ix[0:2, 0:2])
print(data.shape)

# filter out genes and cells with no reads
tic = time.time()
data_filtered = df_filter(data)
print("filtered matrix:")
print(data_filtered.ix[0:2, 0:2])
toc = time.time()
print("filtering took {:.1f} seconds".format(toc-tic))

# normalization
tic = time.time()
data_normalized = df_normalization(data_filtered)
print("normalized matrix:")
print(data_normalized.ix[0:2, 0:2])
toc = time.time()
print("normalization took {:.1f} seconds".format(toc-tic))

# log transformation
tic = time.time()
data_normalized_log = df_log_transformation (data_normalized)
print("log transformed matrix:")
print(data_normalized_log.ix[0:2, 0:2])
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

