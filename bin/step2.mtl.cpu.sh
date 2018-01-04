#!/bin/bash
echo $(hostname)
module load python/3.5.2

# sleep 4h

echo "*--training NN--*"
python -u ./step2.new.mtask.py

echo "*--result analysis--*"
python -u ./result_analysis.py step2

echo "*--weight_clustermap--*"
# needs large RAM
for file in step2/*npy
do python -u weight_clustmap.py $file step2 &
done

echo "*--done--*\n\n"