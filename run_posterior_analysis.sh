srun --pty --gres=gpu:1 -C "a100-80gb|h100" -t 7-00:00:00 --cpus-per-task=10 --mem 150G -p gpu \
bash -c "python sampling_single_sample.py --dataset Celeba; scancel \$SLURM_JOB_ID"