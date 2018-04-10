#!/bin/bash
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH -p reg --mem=100G

echo $(hostname)
echo $(date)
module load R

echo "*--start--*"
R CMD BATCH vivian_scimpute.R

echo "*--done--*\n\n"
echo $(date)