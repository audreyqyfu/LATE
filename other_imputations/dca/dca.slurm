#!/bin/bash
#SBATCH --job-name='DCA_650kC'
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END
##SBATCH -p gpu-long --gres=gpu:1 --mem=100G
#SBATCH -p short --nodes=1 --ntasks-per-node=8 --mem=190GB

# PREP GENERAL
date
hostname
lscpu
totalm=$(free -m | awk '/^Mem:/{print $2}') ; echo 'RAM' $totalm 'MB'


# PREP SPECIFIC
module load python/3.5.2


# RUN
echo "*--STARTED--*"
everything_start=$SECONDS
echo "*--IMPUTATION STARTED--*"
imputation_start=$SECONDS

# IMPUTATION
# dca -t ~/data/cell_row/pbmc.g949_c10k.msk90.csv default_dca  # cell_row
dca ~/imputation/data/10x_mouse_brain_1.3M/1.3M/10kG_subsets/10kG_650kC.h5.csv \
default_dca # gene_row

echo "*--IMPUTATION FINISHED--*"
imputation_end=$SECONDS
imputation_duration=$((imputation_end - imputation_start))
echo "imputation duration: $(($imputation_duration / 60))min $(($imputation_duration
 % 60))s"
echo "*--FINISHED--*\n\n"
date
everything_end=$SECONDS
duration=$((everything_end - everything_start))
echo "everything duration: $(($duration / 60))min $(($duration % 60))s"


# LOG RAM USAGE, AND MORE
sacct --format="JobID, JobName, Partition, AllocCPUS, ReqMem, AveVmSize,
MaxVmSize, State, ExitCode" > sacct.txt
sacct --format="ALL" >> sacct.txt
