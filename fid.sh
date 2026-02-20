dir_path=samples/Celeba_45_0.06/clean/
dest_folder=samples/Celeba_45_0.06/generated/

srun --pty --gres=gpu:1 -C "a100-80gb|h100" -t 7-00:00:00 --cpus-per-task=10 --mem 150G -p gpu \
bash -c "python -m pytorch_fid $dir_path $dest_folder; scancel \$SLURM_JOB_ID"
