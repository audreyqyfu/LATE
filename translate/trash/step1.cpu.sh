module load python/3.5.2
echo "*--training NN--*"
python -u ./step1.n_to_n.new.py

echo "*--result analysis--*"
python -u ./result_analysis.py step1

echo "*--cluster map--*"
for file in pre_train/*npy; do python weight_visualization.py $file clust; done

echo "*--done--*"
