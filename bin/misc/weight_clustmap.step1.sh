echo 'doing clustering for all npy files, needs large RAM'
module load python/3.5.2
for file in step1/*npy
do python -u weight_clustmap.py $file step1 &
done