#!/bin/bash
#SBATCH --job-name='scVI325kC'
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH -p gpu-long --gres=gpu:1 --mem=100G

## SBATCH -p reg --nodes=1 --ntasks-per-node=8 --mem=50GB
## SBATCH -p reg --nodes=1 --ntasks-per-node=16 --mem=150GB # for large dataset


# PREP GENERAL
date
hostname
lscpu
totalm=$(free -m | awk '/^Mem:/{print $2}') ; echo 'RAM' $totalm 'MB'


# PREP SPECIFIC
# source activate py27  # with this use CPU, without this use GPU
cp ~/bin/scVI/* .  #modules


# RUN
echo "*--STARTED--*"
everything_start=$SECONDS
echo "*--IMPUTATION STARTED--*"
imputation_start=$SECONDS

# IMPUTATION
runipy -o scVI_cmd.ipynb

echo "*--IMPUTATION FINISHED--*"
imputation_end=$SECONDS
imputation_duration=$((imputation_end - imputation_start))
echo "imputation duration: $(($imputation_duration / 60))min $(($imputation_duration
 % 60))s"


# ADDITIONAL WORK
gzip *csv
echo "*--FINISHED--*\n\n"
date
everything_end=$SECONDS
duration=$((everything_end - everything_start))
echo "everything duration: $(($duration / 60))min $(($duration % 60))s"