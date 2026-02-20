#!/bin/bash
#SBATCH --job-name=cld
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem 150G
#SBATCH -p gpu
#SBATCH -o /mnt/home/nzilberstein/slurm_logs/%j.out
#SBATCH -e /mnt/home/nzilberstein/slurm_logs/%j.err

module --force purge
module load modules/2.2-20230808
module load texlive  # Needed for matplotlib in notebooks
module load python/3.11.2
# module load cuda cudnn nccl  # 07/01/24: need to comment this line with newer modules? In fact not needed at all
source /mnt/home/nzilberstein/venvs/diffusion/bin/activate

echo "Starting"
python main.py "$@"
