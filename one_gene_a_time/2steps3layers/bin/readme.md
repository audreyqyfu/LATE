# workflow
- 'sh step1.sh'
- 'sh step2.sh'
- use 'sh weight_clustmap.step1/2.sh', if you want weight_clustmap before training finishes (after snapshot has been taken)

# key scripts
- step1.n_to_n.new.py
- step2.new.mtask.py
- result_analysis.py
- weight_clustmap.py

# Pre-processing
- filter_data.py  # select samples/cells
- down_sampling.py  # simulated down-sampling
- log_transformation.py

# misc
- sacct.sh  # monitor slurm usage
- reset.sh  # delete pre_train/ re_train/ plots/ slurm*





# Run note:
