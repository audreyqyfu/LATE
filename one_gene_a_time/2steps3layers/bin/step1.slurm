#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END

echo $(hostname)
nvidia-smi -L
source /usr/modules/init/bash
module load python/3.5.2
python -u ./step1.n_to_n.new.5L.py
python -u hist_activation.py
python -u corr_heatmap.predictions.py
echo "*--done--*"
