sbatch -p gpu-short --gres=gpu:1 step2.omega.slurm
# sbatch --mem=100G -p gpu-long --nodelist=n105 --gres=gpu:1 step2.slurm
