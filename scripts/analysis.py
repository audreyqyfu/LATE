#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import importlib
import scimpute


# READ CMD
#print('''
#usage: python -u result_analysis.py params.py
#
#reads Imputation(Y), Input(X) and Ground-truth(G), 
#compare them and analysis the result
#When no G is available, set X as G in params, so that the code can run
#''')

def load_results(p):
	'''READ DATA
	Parameters
	------------
		p: parameters from global_params.py and example.py
		
	Return
	-----------
                X: input data matrix; genes in columns (same below)
                Y: imputed data matrix
                G: ground truth	
	'''

	print('>READING DATA..')
	X = scimpute.read_data_into_cell_row(p.fname_input, p.ori_input)
	Y = scimpute.read_data_into_cell_row(p.fname_imputation, p.ori_imputation)
	if p.fname_input == p.fname_ground_truth:
		G = X
	else:
		G = scimpute.read_data_into_cell_row(p.fname_ground_truth, p.ori_ground_truth)
	
	print('> DATA TRANSFORMATION..')
	Y = scimpute.df_transformation(Y.transpose(), transformation=p.transformation_imputation).transpose()
	X = scimpute.df_transformation(X.transpose(), transformation=p.transformation_input).transpose()
	if p.fname_input == p.fname_ground_truth:
		G = X
	else:
		G = scimpute.df_transformation(G.transpose(), transformation=p.transformation_ground_truth).transpose()

	# subset/sort X, G to match Y
	# todo: support sparse matrix
	X = X.loc[Y.index, Y.columns]
	G = G.loc[Y.index, Y.columns]

	# TEST MODE OR NOT
	if p.test_flag:
		print('in test mode')
		Y = Y.ix[0:p.m, 0:p.n]
		G = G.ix[0:p.m, 0:p.n]
		X = X.ix[0:p.m, 0:p.n]

	# INPUT SUMMARY
	print('\ninside this code, matrices are supposed to be transformed into cell_row')
	print('Y:', p.fname_imputation, p.ori_imputation, p.transformation_imputation,'\n', Y.ix[0:20, 0:3])
	print('X:', p.fname_input, p.ori_input, p.transformation_input,'\n', X.ix[0:20, 0:3])
	print('G:', p.fname_ground_truth, p.ori_ground_truth, p.transformation_ground_truth,'\n', G.ix[0:20, 0:3])
	print('Y.shape', Y.shape)
	print('X.shape', X.shape)
	print('G.shape', G.shape)

	return X, Y, G


def calculate_MSEs(X, Y, G):
	'''calculate MSEs
	MSE between imputation and input
	MSE between imputation and ground truth

	Parameters
	------------
		X: input data matrix; genes in columns (same below)
		Y: imputed data matrix
		G: ground truth
		
	Return
	-----------
		4 MSEs
	'''

	print('\n> MSE Calculation')
	max_y, min_y = scimpute.max_min_element_in_arrs([Y.values])
	print('Max in Y is {}, Min in Y is{}'.format(max_y, min_y))
	max_g, min_g = scimpute.max_min_element_in_arrs([G.values])
	print('Max in G is {}, Min in G is{}'.format(max_g, min_g))

	mse1_nz = scimpute.mse_omega(Y, X)
	mse1_nz = round(mse1_nz, 7)
	print('MSE1_NZ between Imputation and Input: ', mse1_nz)

	mse1 = scimpute.mse(Y, X)
	mse1 = round(mse1, 7)
	print('MSE1 between Imputation and Input: ', mse1)

	mse2_nz = scimpute.mse_omega(Y, G)
	mse2_nz = round(mse2_nz, 7)
	print('MSE2_NZ between Imputation and Ground_truth: ', mse2_nz)

	mse2 = scimpute.mse(Y, G)
	mse2 = round(mse2, 7)
	print('MSE2 between Imputation and Ground_truth: ', mse2)

	return mse1_nz, mse1, mse2_nz, mse2


def analyze_variation_in_genes(X, Y, G, p):
	'''calculate and visualize standard deviation in each gene
	write SDs to files
	plot histograms of SDs

	Parameters
	------------
		X: input data matrix; genes in columns (same below)
		Y: imputed data matrix
		G: ground truth
	p: parameters
                
	Return
	-----------
		None
	'''

	print('\n calculating standard deviation in each gene for input and imputed matrix')

	x_std_df, y_std_df = scimpute.nz_std(X, Y)
	x_std_df, g_std_df = scimpute.nz_std(X, G)  # purpose: compare G with Y


	#std_ratio_yx_df = pd.DataFrame(data= y_std_df.values / x_std_df.values, index=X.columns, columns=['sd_ratio'])
	#std_ratio_yg_df = pd.DataFrame(data= y_std_df.values / g_std_df.values, index=X.columns, columns=['sd_ratio'])
	std_ratio_yx_data = [(y/x if x!=0 else None) for y, x in zip(y_std_df.values, x_std_df.values)]
	std_ratio_yx_df =pd.DataFrame(data = std_ratio_yx_data, index=X.columns, columns=['sd_ratio'])	
	std_ratio_yg_data = [(y/x if x!=0 else None) for y, x in zip(y_std_df.values, g_std_df.values)]
	std_ratio_yg_df = pd.DataFrame(data= std_ratio_yg_data, index=X.columns, columns=['sd_ratio'])
	
	std_min = min(y_std_df.min(), x_std_df.min(), g_std_df.min())
	std_max = max(y_std_df.max(), x_std_df.max(), g_std_df.max())

	scimpute.hist_df(
		y_std_df,
		xlab='Standard Deviation', title='Imputation({})'.format(p.name_imputation),
		range=(std_min, std_max),
		dir=p.tag)
	scimpute.hist_df(
		x_std_df,
		xlab='Standard Deviation', title='Input({})'.format(p.name_input),
		range=(std_min, std_max),
		dir=p.tag)
	scimpute.hist_df(
		g_std_df,
		xlab='Standard Deviation', title='Ground Truth({})'.format(p.name_input),
		range=(std_min, std_max),
		dir=p.tag)
	scimpute.hist_df(
		std_ratio_yx_df,
		xlab='Ratio of Imputation SD vs Input SD',
		title='',
		range=(std_min, std_max),  
		dir=p.tag)
	scimpute.hist_df(
		std_ratio_yg_df,
		xlab='Ratio of Imputation SD vs Ground Truth SD',
		title='',
		range=(std_min, std_max),
		dir=p.tag)

	std_ratio_yx_df.to_csv('sd_ratio_imputed_vs_input.csv')
	std_ratio_yg_df.to_csv('sd_ratio_imputed_vs_groundtruth.csv')


def visualize_all_genes(X, Y, G, p):
	''' generate plots using all genes
        
	Parameters
        ------------
                X: input data matrix; genes in columns (same below)
                Y: imputed data matrix
                G: ground truth
                p: parameters
                
        Return
        -----------
                None
        '''

	# histograms of gene expression
	max_expression = max(G.values.max(), X.values.max(), Y.values.max())
	min_expression = min(G.values.min(), X.values.min(), Y.values.min())
	print('\n max expression:', max_expression)
	print('\n min expression:', min_expression)

	scimpute.hist_df(
		Y, xlab='Expression', title='Imputation({})'.format(p.name_imputation),
		dir=p.tag, range=[min_expression, max_expression])
	scimpute.hist_df(
		X,  xlab='Expression', title='Input({})'.format(p.name_input),
		dir=p.tag, range=[min_expression, max_expression])
	scimpute.hist_df(
		G,  xlab='Expression', title='Ground Truth({})'.format(p.name_ground_truth),
		dir=p.tag, range=[min_expression, max_expression])

	# histograms of correlations between genes in imputation and ground truth
	# and of correlations between cells in imputation and ground truth
	# when ground truth is not provide, 
	# input is used as ground truth
	print('\n> Correlations between ground truth and imputation')
	print('ground truth dimension: ', G.shape, 'imputation dimension: ', Y.shape)
	print('generating histogram for correlations of genes between ground truth and imputation')
	scimpute.hist_2matrix_corr(
		G.values, Y.values,
		title="Correlation for each gene\n(Ground_truth vs Imputation)\n{}\n{}".
			format(p.name_ground_truth, p.name_imputation),
		dir=p.tag, mode='column-wise', nz_mode='first'  # or ignore
	)

	print('generating histogram for correlations of cells between ground truth and imputation')
	scimpute.hist_2matrix_corr(
		G.values, Y.values,
		title="Correlation for each cell\n(Ground_truth vs Imputation)\n{}\n{}".
			format(p.name_ground_truth, p.name_imputation),
		dir=p.tag, mode='row-wise', nz_mode='first'
	)

	#  heatmaps of data matrices
	print('\n> Generating heatmaps of data matrices')
	range_max, range_min = scimpute.max_min_element_in_arrs([Y.values, G.values, X.values])
	print('\nrange:', range_max, ' ', range_min)

	scimpute.heatmap_vis(Y.values,
		title='Imputation ({})'.format(p.name_imputation),
		xlab='Genes', ylab='Cells', vmax=range_max, vmin=range_min, dir=p.tag)

	scimpute.heatmap_vis(X.values,
		title='Input ({})'.format(p.name_input),
		xlab='Genes', ylab='Cells', vmax=range_max, vmin=range_min, dir=p.tag)

	scimpute.heatmap_vis(G.values,
		title='Ground_truth ({})'.format(p.name_ground_truth),
		xlab='Genes', ylab='Cells', vmax=range_max, vmin=range_min, dir=p.tag)

	# PCA and tSNE plots
	print('\n> Generating PCA and tSNE plots')
	if p.cluster_file is not None:
		cluster_info = scimpute.read_data_into_cell_row(p.cluster_file)
		# cluster_info = cluster_info.astype('str')
	else:
		cluster_info = None

	scimpute.pca_tsne(df_cell_row=Y, cluster_info=cluster_info,
                            title=p.name_imputation, dir=p.tag)
	scimpute.pca_tsne(df_cell_row=X, cluster_info=cluster_info,
                            title=p.name_input, dir=p.tag)
	scimpute.pca_tsne(df_cell_row=G, cluster_info=cluster_info,
                            title=p.name_ground_truth, dir=p.tag)


def visualize_selected_genes(X, Y, G, p):
	''' generate plots for genes specified by the user
        
	Parameters
	------------
		X: input data matrix; genes in columns (same below)
		Y: imputed data matrix
		G: ground truth
		p: parameters
                
	Return
	-----------
		None
	'''

	gene_pair_dir = p.tag+'/pairs'
	List = p.gene_pair_list
	scimpute.gene_pair_plot(Y, list=List, tag='(Imputation)', dir=gene_pair_dir)
	scimpute.gene_pair_plot(X, list=List, tag='(Input)', dir=gene_pair_dir)
	scimpute.gene_pair_plot(G, list=List, tag='(Ground_truth)', dir=gene_pair_dir)

	print("\n> ground truth vs imputation, ground truth vs input")
	gene_dir = p.tag+'/genes'
	for j in p.gene_list:
		try:
			print('for ', j)
			Y_j = Y.ix[:, j]
			G_j = G.ix[:, j]
			X_j = X.ix[:, j]
		except KeyError:
			print('KeyError: gene ID does not exist')
			continue

	scimpute.scatterplot2(G_j, Y_j, range='same',
                          title=str(str(j) + '\n(Ground Truth vs Imputation) '),
                          xlabel='Ground Truth',
                          ylabel='Imputation',
                          dir=gene_dir
                          )
	scimpute.scatterplot2(G_j, X_j, range='same',
                          title=str(str(j) + '\n(Ground Truth vs Input) '),
                          xlabel='Ground Truth',
                          ylabel='Input',
                          dir=gene_dir
                          )


	# Discretize gene expression values
	# and re-generate pairwise plots 
	Y = scimpute.df_exp_discretize_log10(Y)

	print('\n> Discrete gene pair relationship in imputation')
	gene_pair_dir = p.tag+'/pairs_discrete'
	List = p.gene_pair_list
	scimpute.gene_pair_plot(Y, list=List, tag='(Imputation Discrete) ',
                        dir=gene_pair_dir)

	print("\n> Discrete imputation vs ground truth")
	gene_dir = p.tag+'/genes_discrete'
	for j in p.gene_list:
		try:
			print('for ', j)
			Y_j = Y.ix[:, j]
			G_j = G.ix[:, j]
			X_j = X.ix[:, j]
		except KeyError:
			print('KeyError: gene ID does not exist')
			continue

	scimpute.scatterplot2(G_j, Y_j, range='same',
                          title=str(str(j) + '\n(Ground_truth vs Imputation) '),
                          xlabel='Ground Truth',
                          ylabel='Imputation',
                          dir=gene_dir
                          )
	scimpute.scatterplot2(G_j, X_j, range='same',
                          title=str(str(j) + '\n(Ground_truth vs Input) '),
                          xlabel='Ground Truth',
                          ylabel='Input',
                          dir=gene_dir
                          )


def result_analysis(p):
	'''analyzing imputation output

	Parameters
        ------------
                p: parameters from global_params.py and example.py
                
        Return
        -----------
		None
	'''
	# show analysis parameters
	print('''
	Result analysis mode:
	fname_imputation: {}
	name_imputation: {}
	ori_imputation: {}
	trannsformation_imputation: {}
	pair_list: {}
	'''.format(p.fname_imputation, 
				p.name_imputation, 
				p.ori_imputation, 
				p.transformation_imputation,
				p.gene_pair_list))

	# load imputation results and input data
	X, Y, G = load_results(p)

	# calculate MSEs
	#mse1_nz, mse1, mse2_nz, mse2 = calculate_MSEs(X, Y, G)

	# calculate and visualize variation in genes
	#analyze_variation_in_genes(X, Y, G, p)

	# visualize results using all genes
	#visualize_all_genes(X, Y, G, p)

	# visualize selected genes
	visualize_selected_genes(X, Y, G, p)







