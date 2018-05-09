#!/usr/bin/python
# read hd5 and tranform into csv format
import pandas as pd
import sys
import time
import scimpute

print("usage: python data_hd5_show_shape.py in_file")

if len(sys.argv) == 2:
    in_name = sys.argv[1]
    print('usage: {}'.format(sys.argv))
else:
    raise Exception("cmd error")


df = scimpute.read_data_into_cell_row(in_name)

print('finished')
