import pandas as pd
import numpy as np
import time


def read_hd5(in_name):
    '''read in_name into df'''
    print('reading: ', in_name)
    df = pd.read_hdf(in_name)
    print(df.ix[0:3, 0:3])
    print(df.shape)
    return (df)


def split_df(df, a=0.7, b=0.3, c=0.0):
    """input df, output rand split dfs
    a: train, b: valid, c: test
    e.g.: [df_train, df2, df_test] = split(df, a=0.7, b=0.15, c=0.15)"""
    np.random.seed(1)  # for splitting consistency
    train_indices = np.random.choice(df.shape[0], int(df.shape[0] * a // (a + b + c)), replace=False)
    remain_indices = np.array(list(set(range(df.shape[0])) - set(train_indices)))
    valid_indices = np.random.choice(remain_indices, int(len(remain_indices) * b // (b + c)), replace=False)
    test_indices = np.array(list(set(remain_indices) - set(valid_indices)))
    np.random.seed()  # cancel seed effect
    print("total samples being split: ", len(train_indices) + len(valid_indices) + len(test_indices))
    print('train:', len(train_indices), ' valid:', len(valid_indices), 'test:', len(test_indices))

    df_train = df.ix[train_indices, :]
    df_valid = df.ix[valid_indices, :]
    df_test = df.ix[test_indices, :]

    return df_train, df_valid, df_test


def save_hd5(df, out_name):
    tic = time.time()
    df.to_hdf(out_name, key='null', mode='w', complevel=9, complib='blosc')
    toc = time.time()
    print("saving" + out_name + " took {:.1f} seconds".format(toc - tic))

file = 'scdata.magic.data.t.hd5'
df = read_hd5(file)  # genes, cells
dt = df.transpose()
dta, dtb, dtc = split_df(dt, a=0.7, b=0.3, c=0.0)
dfa = dta.transpose()
dfb = dtb.transpose()

save_hd5(dfa, 'EMT.MAGIC.A.hd5')  # genes, cells
save_hd5(dfb, 'EMT.MAGIC.B.hd5')  # genes, cells

