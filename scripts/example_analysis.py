# -*- coding: utf-8 -*-
import sys
import Late1
import time
import analysis

if __name__ == '__main__':
	
	##1. load parameter module and use name 'p'
	print("Usage: python example.py -mode <late> -infile <xx.hd5>")
	argms = Late1.parse_args(sys.argv[1:])
	p = Late1.load_params(argms.mode, argms.infile)	

	# need to add parameters to global_params
	# need to add ground truth file name here
	
	p.fname_ground_truth = '../gtex_v7.4TISSUES.count.cell_row.hd5'
	
	#p.L = 7
	#p.max_training_epochs = int(20)
	#p.display_step = int(5)
	#p.snapshot_step = int(50)
	#p.patience = int(3)
	
	##2. refresh folder
	#log_dir = './{}'.format(p.stage)
	#scimpute.refresh_logfolder(log_dir)
	Late1.display_params(p)
	
	tic_start = time.time()
	#3. load data
	#input_matrix, gene_ids, cell_ids = Late1.read_data(p)
	
	#4. call late
	#Late1.late_main(input_matrix, gene_ids, cell_ids, p, log_dir, rand_state = 3)
	
	#5. analyze imputation results
	analysis.result_analysis(p)

	toc_stop = time.time()
	time_finish = round((toc_stop - tic_start), 2)

	print("Analysis Finished!")
	print("Wall Time Used: {} seconds".format(time_finish))
