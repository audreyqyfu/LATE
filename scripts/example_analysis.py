# -*- coding: utf-8 -*-
import sys
import late
import time

if __name__ == '__main__':
	
	##1. load parameter module and use name 'p'
#	print("Usage: python example.py -mode <late> -infile <xx.hd5>")
	argms = late.parse_args(sys.argv[1:])
	p = late.load_params(argms.mode, argms.infile)	

	# need to add parameters to global_params
	# need to add ground truth file name here
	
	# specify ground truth data	   
	p.fname_ground_truth = '/Users/audreyq.fu/Documents/SingleCell/Code/gtex_v7.4TISSUES.count.cell_row.hd5'
	
	# specify gene pairs
	p.gene_pair_list = [
		# Index
		[4, 5], [10,11],
		# Gene names
		['ENSG00000173372', 'ENSG00000087086'],
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
