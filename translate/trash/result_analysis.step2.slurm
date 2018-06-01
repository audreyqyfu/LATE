#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END

echo $(hostname)
nvidia-smi -L
source /usr/modules/init/bash
module load python/3.5.2
python -u ./result_analysis.py step2
for file in step2/*npy
do python -u weight_clustmap.py $file step2 &
done
echo "*--done--*"

