sbatch -p gpu-long --gres=gpu:1 --nodelist=n105 translate_params.slurm
# sbatch --mem=100G -p gpu-long --nodelist=n105 --gres=gpu:1 step2.slurm
