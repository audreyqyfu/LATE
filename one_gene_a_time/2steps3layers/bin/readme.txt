# workflow
sh step1.sh
sh result_analysis.step1.sh
for file in pre_train/*npy; do python weight_visualization.py $file step1; done  # heatmap for weights and bottleneck

sh step2.sh
sh result_analysis.step2.sh
for file in re_train/*npy; do python weight_visualization.py $file step2; done  # heatmap for weights and bottleneck

# key scripts
- step1.n_to_n.new.py
- step2.new.mtask.py
- result_analysis.py
- weight_visualization.py

# Pre-processing
- filter_data.py  # select samples/cells
- down_sampling.py  # simulated down-sampling
- log_transformation.py
-

# misc
- sacct.sh  # monitor slurm usage
- reset.sh  # delete pre_train/ re_train/ plots/ slurm*
-




# Run note: