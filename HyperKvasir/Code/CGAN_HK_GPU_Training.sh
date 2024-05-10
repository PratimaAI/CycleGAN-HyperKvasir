#!/bin/sh
#SBATCH -p dgx2q
#SBATCH --gres=gpu:1
#SBATCH -N 1   # nodes
#SBATCH -n 1   # MPI ranks
#SBATCH -c 6    # OpenMP

ulimit -s 10240
module purge
module load slurm/21.08.8
  
# Activate conda env
conda activate myenv

# Nerf2nerf training script
CUDA_LAUNCH_BLOCKING=1 srun python CGAN_HK_GPU_Training.py


