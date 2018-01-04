echo 'needs to change output dir name from pre_train to step1'
sleep 10s
sbatch -p gpu-long --gres=gpu:1 --nodelist=n105 step1.mse.slurm

# --mem=100G

