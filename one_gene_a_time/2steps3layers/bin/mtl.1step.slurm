#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END

echo $(hostname)
nvidia-smi -L
source /usr/modules/init/bash
module load python/3.5.2
python -u ./mtl.1step.py
python -u ./result_analysis.py step2
for file in re_train/*.npy; do python weight_visualization.py $file mtl_1step_Bmsk90; done
echo "*--done--*"