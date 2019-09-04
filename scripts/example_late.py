# -*- coding: utf-8 -*-
'''
Usage: running LATE or TRANSLATE.  The general syntax is:

	$ python example_script.py -mode <pre-training | late | translate | impute> -infile <xx.csv/tsv/h5>

'''

import sys
import late
import scimpute
import time

if __name__ == '__main__':
	##1. load parameter module and use name 'p'
	argms = late.parse_args(sys.argv[1:])
	p = late.load_params(argms.mode, argms.infile)	
	
	# a short run with 20 epochs
	# for the input data, which contains 949 genes and 10k cells
	# about 210 seconds on a macbook pro with CPU
	p.max_training_epochs = int(20)
	p.display_step = int(10)
	p.snapshot_step = int(10)
	p.patience = int(3)
	
	late.display_params(p)

	##2. refresh folder
	log_dir = './{}'.format(p.stage)
	scimpute.refresh_logfolder(log_dir)

	tic_start = time.time()
	
	##3. call late
	late.late_main(p, log_dir, rand_state = 3)
	toc_stop = time.time()
	time_finish = round((toc_stop - tic_start), 2)
	
	print("Imputation Finished!")
	print("Wall Time Used: {} seconds".format(time_finish))
