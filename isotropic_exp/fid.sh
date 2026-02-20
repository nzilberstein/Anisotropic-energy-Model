source_folder=ImageNet64_21_daps_freq

dir_path=samples/$source_folder/clean/
dest_folder=samples/$source_folder/generated_daps/

srun --pty --gres=gpu:1 -C "a100-80gb|h100" -t 7-00:00:00 --cpus-per-task=10 --mem 150G -p gpu \
bash -c "python -m pytorch_fid $dir_path $dest_folder; scancel \$SLURM_JOB_ID"