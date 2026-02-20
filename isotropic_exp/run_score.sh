sbatch --gres=gpu:4 -C "a100-80gb|h100" -t 7-00:00:00 job_fg.sbatch \
    --name multigpu/inpainting_afhq/denoiser_song_score_anisoEmb_groupNorm_mult_lr2e-4_lrdecay50000_1000warmup_128bs\
    --dataset AFHQ \
    --no-grayscale \
    --train-batch-size 64 \
    --test-batch-size 4 \
    --network UNet \
    --network-kwargs "{'num_scales':3,'group_size':1,'num_layers_encoder_block':3,'num_layers_mid_block':3,'num_layers_decoder_block':3}" \
    --model DenoiserModel \
    --reparam-kwargs "{'residual':False}" \
    --min-noise-level psnr=90 \
    --max-noise-level psnr=-30 \
    --noise-level-sampler UniformLog \
    --mse-var-exponent -1 \
    --train-noise-score 0 \
    --noise-score-var-exponent 0 \
    --lr 0.0002 \
    --num-training-steps 300000 \
    --lr-decay-every 100000 \
    --warmup-steps 1000 \
    --size-network large \
    --no-frequency-component \
    --spatial-component \
    --noise-covariance 1 \

# sbatch --gres=gpu:4 -C "a100-80gb|h100" -t 7-00:00:00 job_fg.sbatch \
#     --name multigpu/inpainting/denoiser_song_score_anisoEmb_groupNorm_mult_lr2e-4_lrdecay50000_1000warmup\
#     --dataset CIFAR10 \
#     --no-grayscale \
#     --train-batch-size 256 \
#     --test-batch-size 4 \
#     --network UNet \
#     --network-kwargs "{'num_scales':3,'group_size':1,'num_layers_encoder_block':3,'num_layers_mid_block':3,'num_layers_decoder_block':3}" \
#     --model DenoiserModel \
#     --reparam-kwargs "{'residual':False}" \
#     --min-noise-level psnr=90 \
#     --max-noise-level psnr=-30 \
#     --noise-level-sampler UniformLog \
#     --mse-var-exponent -1 \
#     --train-noise-score 0 \
#     --noise-score-var-exponent 0 \
#     --lr 0.0002 \
#     --num-training-steps 300000 \
#     --lr-decay-every 100000 \
#     --warmup-steps 1000 \
#     --size-network large \
#     --no-frequency-component \
#     --spatial-component \
#     --noise-covariance 1 \

# sbatch --gres=gpu:3 -C "a100-80gb|h100" -t 7-00:00:00 job_fg.sbatch \
#     --name multigpu/combined_cov_tworandvars/denoiser_ImageNet_songSmall_score_anisoEmb_groupNorm_mult_lr1.5e-4_lrdecay50000_1000warmup \
#     --dataset ImageNet64 \
#     --no-grayscale \
#     --train-batch-size 256 \
#     --test-batch-size 4 \
#     --network UNet \
#     --network-kwargs "{'num_scales':3,'group_size':1,'num_layers_encoder_block':3,'num_layers_mid_block':3,'num_layers_decoder_block':3}" \
#     --model DenoiserModel \
#     --reparam-kwargs "{'residual':False}" \
#     --min-noise-level psnr=90 \
#     --max-noise-level psnr=-30 \
#     --noise-level-sampler UniformLog \
#     --mse-var-exponent -1 \
#     --train-noise-score 0 \
#     --noise-score-var-exponent 0 \
#     --lr 0.0002 \
#     --num-training-steps 200000 \
#     --lr-decay-every 50000 \
#     --noise-covariance 1 \
#     --warmup-steps 1000 \
#     --size-network small \


# --reparam-kwargs "{'output_scaling':1,'residual':True}" \
# export WORLD_SIZE=1
# eexport LOCAL_RANK=1
# torchrun --standalone --nnodes=1 --nproc-per-node=2 \
# python main.py \
#     --name multigpu/combined_cov_tworandvars/tests_denoiser \
#     --dataset CIFAR10 \
#     --no-grayscale \
#     --train-batch-size 128 \
#     --test-batch-size 32 \
#     --network UNet \
#     --network-kwargs "{'num_scales':3,'group_size':1,'num_layers_encoder_block':3,'num_layers_mid_block':3,'num_layers_decoder_block':3}" \
#     --model DenoiserModel \
#     --reparam-kwargs "{'residual':False}" \
#     --min-noise-level psnr=90 \
#     --max-noise-level psnr=-30 \
#     --noise-level-sampler UniformLog \
#     --mse-var-exponent -1 \
#     --train-noise-score 1 \
#     --noise-score-var-exponent 0 \
#     --lr 0.0003 \
#     --num-training-steps 200000 \
#     --lr-decay-every 10000 \
#     --noise-covariance 1 \
#     --warmup-steps 0 \
#     --size-network small \