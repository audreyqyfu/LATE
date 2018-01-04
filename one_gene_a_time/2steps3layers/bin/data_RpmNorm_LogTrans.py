#!/usr/bin/python
# read count.txt/csv; do MAGIC style normalization (scale back to median);
# do log (x+1) transformation
# output in blosc-compressed hd5 and csv.gz format

print("data_RpmNorm_LogTrans.py: read csv [row=genes, col=cells], normalize similar to magic")

import pandas as pd
import numpy as np
import time
import os


def df_filter(df):
    df_filtered = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]
    print("filtered out any rows and columns with sum of zero")
    return (df_filtered)


def df_normalization(df):
    read_counts = df.sum(axis=0)  # colsum
    df_normalized = df.div(read_counts, axis=1).mul(np.median(read_counts)).mul(1)
    return (df_normalized)


def df_log_transformation(df, pseudocount=1):
    df_log = np.log10(np.add(df, pseudocount))
    return (df_log)


# input
in_name = 'All_Tissue_Site_Details.combined.reads.gct'
out_prefix = "gtex_v7"

# reading data
print("started reading file: ", in_name)
tic = time.time()
data = pd.io.parsers.read_csv(in_name,
                              index_col=0,  # col=0 as rownames (gene ID)
                              sep="\t",
                              skiprows=2)  # skip first two commentary rows
del data['Description']  # delete column gene_description
toc = time.time()
print("reading input took {:.1f} seconds".format(toc - tic))
print("input matrix:")
print(data.ix[0:2, 0:2])
print(data.shape)

# filter out genes and cells with no reads
tic = time.time()
data_filtered = df_filter(data)
toc = time.time()
print("filtered matrix:")
print(data_filtered.ix[0:2, 0:2])
print(data.shape)
print("filtering took {:.1f} seconds".format(toc - tic))

# normalization
tic = time.time()
data_normalized = df_normalization(data_filtered)
toc = time.time()
print("normalized matrix:")
print(data_normalized.ix[0:2, 0:2])
print(data_normalized.shape)
print("normalization took {:.1f} seconds".format(toc - tic))

# log transformation
tic = time.time()
data_normalized_log = df_log_transformation(data_normalized)
toc = time.time()
print("log transformed matrix:")
print(data_normalized_log.ix[0:2, 0:2])
print(data_normalized_log.shape)
print("log-tansformation took {:.1f} seconds".format(toc - tic))

# output
print("saving output norm.hd5: " + os.getcwd())
tic = time.time()
data_normalized.to_hdf(out_prefix + '.norm.hd5', key='null', mode='w', complevel=9, complib='blosc')
toc = time.time()
print("saving " + out_prefix + ".norm.hd5 took {:.1f} seconds".format(toc - tic))

print("saving output norm.log.hd5: " + os.getcwd())
tic = time.time()
data_normalized_log.to_hdf(out_prefix + '.norm.log.hd5', key='null', mode='w', complevel=9, complib='blosc')
toc = time.time()
print("saving " + out_prefix + ".norm.log.hd5 took {:.1f} seconds".format(toc - tic))

print("saving output norm.csv.gz")
tic = time.time()
data_normalized.to_csv(out_prefix + ".norm.csv.gz", compression='gzip')
toc = time.time()
print("saving " + out_prefix + ".norm.csv.gz took {:.1f} seconds".format(toc - tic))

print("saving output norm.log.csv.gz")
tic = time.time()
data_normalized_log.to_csv(out_prefix + ".norm.log.csv.gz", compression='gzip')
toc = time.time()
print("saving " + out_prefix + ".norm.log.csv.gz took {:.1f} seconds".format(toc - tic))
