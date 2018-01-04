#!/bin/bash
echo $(hostname)
module load python/3.5.2

echo 'needs to change output dir name from pre_train to step1'
sleep 10s

echo "*--training NN--*"
python -u ./step1.mse.py

echo "*--result analysis--*"
python -u ./result_analysis.py step1

echo "*--weight_clustermap--*"
# needs large RAM
for file in step1/*npy
do python -u weight_clustmap.py $file step1 &
done

echo "*--done--*\n\n"