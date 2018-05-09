#!/usr/bin/python
# read hd5 and tranform into csv format
import pandas as pd
import sys
import time
import scimpute


print("usage: python data_h5_transpose.py genome in_name.h5  out_name.h5")
print('genome example: mm10')

if len(sys.argv) == 4:
    genome = sys.argv[1]
    in_name = sys.argv[2]
    out_name = sys.argv[3]
    print('usage: {}'.format(sys.argv))
else:
    raise Exception("cmd error")


df = scimpute.read_sparse_matrix_from_h5(in_name,
                                         genome=genome,
                                         file_ori='gene_row' ) # ensures transpose

scimpute.save_sparse_matrix_to_h5(df,
                                  filename=out_name,
                                  genome=genome)
print('finished')