#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH -o step1.out
#SBATCH -e step1.err

echo $(hostname)
nvidia-smi -L
source /usr/modules/init/bash
module load python/3.5.2


echo "*--training NN--*"
python -u ./step1.n_to_n.new.py

echo "*--result analysis--*"
python -u ./result_analysis.py step1

echo "*--cluster map--*"
for file in pre_train/*w*npy; do python weight_visualization.py $file sns_tag; done

echo "*--done--*"
