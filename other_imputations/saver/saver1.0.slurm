#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH -p long --mem=128G

# PREP GENERAL
date
hostname
lscpu
totalm=$(free -m | awk '/^Mem:/{print $2}') ; echo 'RAM' $totalm 'MB'
module load R/3.3.2

# RUN
echo "*--STARTED--*"
everything_start=$SECONDS

echo "*--IMPUTATION STARTED--*"
imputation_start=$SECONDS
R CMD BATCH saver1.0.R
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


# THIS IS A TEMPLATE: sbatch; log timing, hostname, cpu, gpu