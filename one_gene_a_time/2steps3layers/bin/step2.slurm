#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END

echo $(hostname)
nvidia-smi -L
source /usr/modules/init/bash
module load python/3.5.2
python -u ./step2.new.mtask.py
python -u ./result_analysis.step2.py
echo "*--done--*"
