# h5 of 10x_genomics to csv (1.3M cell mouse brain dataset use 70-171GB RAM)
# output is cell_row
# todo: b'ENSMUSG00000092341' issue (`b` added for index and column_id), problem?



import scimpute
import pandas as pd
import os
import psutil

def usage():
    process = psutil.Process(os.getpid())
    ram = process.memory_info()[0] / float(2 ** 20)
    ram = round(ram, 1)
    return ram

fname='../mouse_brain.10kg.h5'
genome='mm10'
file_ori='gene_row'
outname='mouse_brain.10kg.csv'

data = scimpute.read_sparse_matrix_from_h5(fname=fname,
                                          genome=genome,
                                          file_ori=file_ori)
print('RAM usage after reading sparse matrix: ', '{} M'.format(usage()))

print('converting to dense data frame')
df = pd.DataFrame(data=data.matrix.todense(),
                  index=data.barcodes,
                  columns=data.gene_ids)
print('RAM usage after conversion: ', '{} M'.format(usage()))


print('writing..')
df.to_csv(outname)
print(outname, 'saved')

# fix mouse brain data b'id' issue
#  cat mouse_brain.1kg.csv |sed s/^b\'//g|sed s/\'//g|sed s/,b/,/g > temp