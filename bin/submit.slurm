#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH -p gpu-long --gres=gpu:1 --mem=102G
# #SBATCH --nodelist=n105


# PREP GENERAL
date
hostname
lscpu

# PREP TENSORFLOW PY35 SPECIFIC
nvidia-smi -L
source /usr/modules/init/bash
module load python/3.5.2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# RUN
echo "*--STARTED--*"
everything_start=$SECONDS

echo "*--IMPUTATION STARTED--*"
imputation_start=$SECONDS
python3 -u ./translate.py params.py
echo "*--IMPUTATION FINISHED--*"
imputation_end=$SECONDS
imputation_duration=$((imputation_end - imputation_start))
echo "imputation duration: $(($imputation_duration / 60))min $(($imputation_duration
 % 60))s"


echo "*--WEIGHT CLUSTERMAP-*"
for file in step2/*_w*npy;do python -u weight_clustmap.py $file step2;done
for file in step2/code*npy;do python -u weight_clustmap.py $file step2;done

echo "*--RESULT ANALYSIS--*"
python -u ./result_analysis.py params.py

echo "*--FINISHED--*\n\n"
date
everything_end=$SECONDS
duration=$((everything_end - everything_start))
echo "everything duration: $(($duration / 60))min $(($duration % 60))s"


# THIS IS A TEMPLATE: sbatch; log timing, hostname, cpu, gpu