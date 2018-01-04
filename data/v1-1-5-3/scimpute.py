import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt


def read_csv(fname):
    '''read_csv into pd.df, assuming index_col=0, and header=True'''
    print('reading ', fname)
    tic = time.time()
    df = pd.read_csv(fname, index_col=0)
    # print("read matrix: [genes, cells]")
    print('data shape: ', df.shape)
    print(df.ix[0:3, 0:3])
    print(df.ix[-3:, -3:])
    # print(df.axes)
    toc = time.time()
    print("reading took {:.1f} seconds".format(toc - tic))
    return (df)


def save_hd5(df, out_name):
    tic = time.time()
    df.to_hdf(out_name, key='null', mode='w', complevel=9, complib='blosc')
    toc = time.time()
    print("saving" + out_name + " took {:.1f} seconds".format(toc - tic))


def read_hd5(in_name):
    '''read in_name into df'''
    print('reading: ', in_name)
    df = pd.read_hdf(in_name)
    print(df.ix[0:3, 0:3])
    print(df.shape)
    return (df)


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


def density_plot(x, y, title, fname):
    # create plots directory
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fname = "./plots/" + fname
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # sort: dense on top (plotted last)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    # plt
    # fig = plt.figure(figsize=(9,9))
    fig, ax = plt.subplots()
    cax = ax.scatter(x, y, c=z, s=50, edgecolor='')
    plt.title(title)
    plt.colorbar(cax)
    plt.savefig(fname + ".png", bbox_inches='tight')
    plt.close(fig)


def genescatterplot(gene1, gene2, scdata):
    gene1 = str(gene1);
    gene2 = str(gene2)
    fig, ax = scdata.scatter_gene_expression([gene1, gene2])
    fig.savefig(gene1 + "_" + gene2 + '.biaxial.png')
    # after magic
    fig, ax = scdata.magic.scatter_gene_expression([gene1, gene2])
    fig.savefig(gene1 + "_" + gene2 + '.magic.biaxial.png')
    plt.close(fig)


def genescatterplot3d(gene1, gene2, gene3, scdata):
    gene1 = str(gene1);
    gene2 = str(gene2);
    gene3 = str(gene3);
    fig, ax = scdata.scatter_gene_expression([gene1, gene2, gene3])
    fig.savefig(gene1 + "_" + gene2 + "_" + gene3 + '.biaxial.png')
    # after magic
    fig, ax = scdata.magic.scatter_gene_expression([gene1, gene2, gene3])
    fig.savefig(gene1 + "_" + gene2 + "_" + gene3 + '.magic.biaxial.png')
    plt.close(fig)


def hist_list(list, xlab='xlab', title='histogram'):
    '''output histogram of a list into png'''
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fname = str(title) + '.hist.png'
    fname = "./plots/" + fname
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel('Frequency')
    plt.hist(list)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def bone_marrow_biaxial_plots(scdata):
    # Gene-Gene scatter plot (before & after magic)
    # Fig3
    print("gene-gene plot for bone marrow dataset")
    genescatterplot('Cd34', 'Gypa', scdata)  # CD325a
    genescatterplot('Cd14', 'Itgam', scdata)  # cd11b
    genescatterplot('Cd34', 'Fcgr2b', scdata)  # cd32, similar plot
    genescatterplot3d('Cd34', 'Gata1', 'Gata2', scdata)
    genescatterplot3d('Cd44', 'Gypa', 'Cpox', scdata)
    genescatterplot3d('Cd34', 'Itgam', 'Cd14', scdata)
    # Fig12
    genescatterplot('Cd34', 'Itgam', scdata)
    genescatterplot('Cd34', 'Apoe', scdata)
    genescatterplot('Cd34', 'Gata1', scdata)
    genescatterplot('Cd34', 'Gata2', scdata)
    genescatterplot('Cd34', 'Ephb6', scdata)
    genescatterplot('Cd34', 'Lepre1', scdata)
    genescatterplot('Cd34', 'Mrpl44', scdata)
    genescatterplot('Cd34', 'Cnbp', scdata)
    # Fig14
    genescatterplot('Gata1', 'Gata2', scdata)
    genescatterplot('Klf1', 'Sfpi1', scdata)
    genescatterplot('Meis1', 'Cebpa', scdata)
    genescatterplot('Elane', 'Cebpe', scdata)


def heatmap_vis(arr, title='visualization of matrix', cmap="rainbow", 
    vmin = None, vmax = None, xlab = '', ylab = ''):
    '''heatmap visualization of 2D matrix
    cmap options PiYG for [neg, 0, posi]
    Greys Reds for [0, max]
    rainbow for [0,middle,max]'''
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fname = "./plots/" + title + '.vis.png'

    if (vmin is None):
        vmin = np.min(arr)
    if (vmax is None):
        vmax = np.max(arr)

    fig = plt.figure(figsize=(9, 9))
    plt.imshow(arr, cmap=cmap, vmin = vmin, vmax = vmax)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar()
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('heatmap vis ', title, ' done')


def hist_arr_flat (arr, title='', xlab='', ylab=''):
    '''create histogram for flattened arr'''
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fname = "./plots/" + title + '.hist.png'

    fig = plt.figure(figsize=(9,9))
    n, bins, patches = plt.hist(arr.flatten(), 100, normed=1, facecolor='green', alpha=0.75)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print("histogram ", title, ' done')
