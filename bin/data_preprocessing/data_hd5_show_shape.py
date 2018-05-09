#!/usr/bin/python
# read hd5 and tranform into csv format
import pandas as pd
import sys
import time


def read_hd5(in_name):
    '''
    :param in_name: 
    :return df: 
    '''
    print('reading: ', in_name)
    df = pd.read_hdf(in_name)
    print('shape:', df.shape)
    # print(df.axes)
    if df.shape[0] > 2 and df.shape[1] > 2:
        print(df.ix[0:3, 0:2])
    return df


print("usage: python data_hd5_show_shape.py in_name.hd5")

if len(sys.argv) == 2:
    in_name = sys.argv[1]
    print('usage: {}'.format(sys.argv))
else:
    raise Exception("cmd error")


df = read_hd5(in_name)

print('finished')
