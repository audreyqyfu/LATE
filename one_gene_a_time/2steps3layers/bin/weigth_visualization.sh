module load python/3.5.2
for file in audrey_weights_biases/*w*npy; do python weight_visualization.py $file audrey_3000epoch; done