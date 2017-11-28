module load python/3.5.2
for file in step2/*npy; do python -u weight_visualization.py $file step2; done