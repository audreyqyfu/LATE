#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END

echo $(hostname)
nvidia-smi -L
source /usr/modules/init/bash
module load python/3.5.2
python -u ./result_analysis.py step1
for file in step1/*npy
do python -u weight_clustmap.py $file step1 &
done
echo "*--done--*"

