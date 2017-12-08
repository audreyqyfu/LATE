sbatch --mem=50G -p gpu-short --gres=gpu:1 step1.omega.slurm

# sbatch --mem=100G -p gpu-long --gres=gpu:1 --nodelist=n105 step1.slurm

