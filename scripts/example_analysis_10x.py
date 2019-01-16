# -*- coding: utf-8 -*-
import sys
import late
import time

if __name__ == '__main__':
	
	# load parameter module and use name 'p'
	argms = late.parse_args(sys.argv[1:])
	p = late.load_params(argms.mode, argms.infile)	

	# 10x Genomics h5 data has genes in rows and cells in columns
	p.ori_input = 'gene_row'

	# no ground truth data
	# set ground truth the same as the input	   
	p.fname_ground_truth = p.fname_input
	
	# specify gene pairs
	p.gene_pair_list = [
		# Index
		[4, 5], [10,11],
		# Gene names
#		['ENSG00000173372', 'ENSG00000087086'],
	]

	# output parameters for result analysis
	late.display_params(p)
	
	tic_start = time.time()
	# analyze imputation results
	late.result_analysis_main(p)

	toc_stop = time.time()
	time_finish = round((toc_stop - tic_start), 2)

	print("Analysis Finished!")
	print("Wall Time Used: {} seconds".format(time_finish))
