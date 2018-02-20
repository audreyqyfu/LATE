#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH -p gpu-long --gres=gpu:1
#SBATCH --nodelist=n110

echo $(hostname)
echo $(date)
nvidia-smi -L

source /usr/modules/init/bash
module load python/3.5.2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "*--training NN--*"
python -u ./translate.py step1_params.py
python -u ./translate.py step2_params.py

echo "*--result analysis--*"
python -u ./result_analysis_matrix_based.py result_analysis_matrix_based_params.py

# echo "*--weight_clustermap--*"
# for file in step2/*_w*npy;do python -u weight_clustmap.py $file step2;done
# for file in step2/code*npy;do python -u weight_clustmap.py $file step2;done

echo "*--done--*\n\n"
echo $(date)
