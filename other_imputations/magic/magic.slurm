#!/bin/bash
#SBATCH --job-name='M_MB28kG'
#SBATCH --mail-user=rui@uidaho.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH -p reg --nodes=1 --ntasks-per-node=8 --mem=150GB


# PREP
source activate py35

date
hostname
lscpu
totalm=$(free -m | awk '/^Mem:/{print $2}') ; echo 'RAM' $totalm 'MB'

echo "*--STARTED--*"
everything_start=$SECONDS
echo "*--IMPUTATION STARTED--*"
imputation_start=$SECONDS

# PARAMS
## GTEx
#file='/mnt/lfs2/rui/data/cell_row/gtex_v7.4tissues.msk90.csv.gz'
#outname='gtex_v7.count.4tissues.msk90.magic.csv'
## MBM_MAGIC
#file='/mnt/lfs2/rui/data/cell_row/MBM.MAGIC.9k.B.msk90.csv.gz'
#outname='mbm_magic.msk.magic.csv'
## PBMC.G949_C21K.MSK
#file='/mnt/lfs2/rui/data/cell_row/pbmc.g949_c21k.msk90.csv.gz'
#outname='pbmc.g949_c21k.msk.magic.csv'
## PBMC.G5561
#file='/mnt/lfs2/rui/data/cell_row/pbmc.g5561_c54k.csv.gz'
#outname='pbmc.g5561_c54k.magic.csv'
## PBMC.G9987
#file='/mnt/lfs2/rui/data/cell_row/pbmc.g9987_c54k.csv.gz'
#outname='pbmc.g9987_c54k.magic.csv'
## PBMC.G949_C10K.MSK
#file='/mnt/lfs2/rui/data/cell_row/pbmc.g949_c10k.msk90.csv.gz'
#outname='pbmc.g949_c10k.msk.magic.csv'
## MOUSE BRAIN 10kG
#file='/mnt/lfs2/rui/imputation/data/10x_mouse_brain_1.3M/1.3M/10kG_subsets/10kG_650kC.h5.csv'
#outname='mouse_brain.10kg_650kc.magic.csv'
## MOUSE BRAIN 28kG
file='/mnt/lfs2/rui/imputation/data/10x_mouse_brain_1.3M/1.3M/28kG_subsets/csv/28kG_650kC.csv'
outname='mouse_brain.28kg_650kc.magic.csv'

# RUN
python -u ~/bin/magic/python/magic/MAGIC.py csv -d $file -o $outname \
--cell-axis=rows -p 100 -t=9 -k=12 -ka=4 \
1>./$outname.log 2>&1

echo "*--IMPUTATION FINISHED--*"
imputation_end=$SECONDS
imputation_duration=$((imputation_end - imputation_start))
echo "imputation duration: $(($imputation_duration / 60))min $(($imputation_duration
 % 60))s"


# ADDITIONAL WORK
gzip $outname
echo "*--ZIP FINISHED--*\n\n"
date
everything_end=$SECONDS
duration=$((everything_end - everything_start))
echo "everything duration: $(($duration / 60))min $(($duration % 60))s"