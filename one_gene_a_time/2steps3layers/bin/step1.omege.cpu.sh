#!/bin/bash
echo $(hostname)
module load python/3.5.2

echo "*--training NN--*"
python -u ./step1.omega.py

echo "*--result analysis--*"
python -u ./result_analysis.py step1

echo "*--weight_clustermap--*"
# needs large RAM
for file in step1/*npy
do python -u weight_clustmap.py $file step1 &
done

echo "*--done--*\n\n"