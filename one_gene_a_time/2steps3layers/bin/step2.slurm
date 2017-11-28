#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END

echo $(hostname)
nvidia-smi -L
source /usr/modules/init/bash
module load python/3.5.2


echo "*--training NN--*"
python -u ./step2.new.mtask.py

echo "*--result analysis--*"
python -u ./result_analysis.py step2
python -u weight_visualization.step2.sh


echo "*--done--*"
