# sbatch --gres=gpu:8 -C "a100-80gb|h100" -t 7-00:00:00 job_fg.sbatch \
#     --name multigpu/all_together/energy_song_dual_truncatedFreq_afhq_cat_192 \
#     --dataset AFHQ \
#     --no-grayscale \
#     --train-batch-size 32 \
#     --test-batch-size 4 \
#     --network UNet \
#     --network-kwargs "{'num_scales':3,'group_size':1,'num_layers_encoder_block':3,'num_layers_mid_block':3,'num_layers_decoder_block':3}" \
#     --model EnergyModel \
#     --reparam-kwargs "{'conversion':'inner_product'}" \
#     --min-noise-level psnr=90 \
#     --max-noise-level psnr=-30 \
#     --noise-level-sampler UniformLog \
#     --mse-var-exponent -1 \
#     --train-noise-score 1 \
#     --noise-score-var-exponent 1 \
#     --lr 0.00005 \
#     --num-training-steps 200000 \
#     --lr-decay-every 100000 \
#     --noise-covariance 1 \
#     --size-network large \
#     --no-adaptive-scale \
#     --num-workers 8 \
#     --num-acum-gradients 2 \
#     --warmup-steps 0 \
#     --use-ema \
#     --ema-decay 0.9999 \
#     --ema-update-every 1
    # vs 16

sbatch --gres=gpu:1 -C "a100-80gb|h100" -t 7-00:00:00 job_fg.sbatch \
    --name multigpu/all_together/energy_song_dual_truncatedFreq_gaussian_iso \
    --dataset Gaussian8x8 \
    --no-grayscale \
    --train-batch-size 256 \
    --test-batch-size 4 \
    --network UNet \
    --model EnergyModel \
    --reparam-kwargs "{'conversion':'inner_product'}" \
    --min-noise-level psnr=90 \
    --max-noise-level psnr=-30 \
    --noise-level-sampler UniformLog \
    --mse-var-exponent -1 \
    --train-noise-score 1 \
    --noise-score-var-exponent 1 \
    --lr 0.0002 \
    --num-training-steps 400000 \
    --lr-decay-every 100000 \
    --noise-covariance 1 \
    --warmup-steps 0 \
    --size-network small \
    --no-adaptive-scale \
    --num-workers 8

    ## MISSING TRAIN NOISE SCORE

# sbatch --gres=gpu:4 -C "a100-80gb|h100" -t 7-00:00:00 job_fg.sbatch \
#     --name multigpu/combined_cov_tworandvars/energy_song_anisoEmb_groupNorm_lambda1_mult_lr2e-4_nolrdecay_1000warmup_d2 \
#     --dataset CIFAR10 \
#     --no-grayscale \
#     --train-batch-size 512 \
#     --test-batch-size 4 \
#     --network UNet \
#     --network-kwargs "{'num_scales':3,'group_size':1,'num_layers_encoder_block':3,'num_layers_mid_block':3,'num_layers_decoder_block':3}" \
#     --model EnergyModel \
#     --reparam-kwargs "{'conversion':'inner_product'}" \
#     --min-noise-level psnr=90 \
#     --max-noise-level psnr=-30 \
#     --noise-level-sampler UniformLog \
#     --mse-var-exponent -1 \
#     --train-noise-score 1 \
#     --noise-score-var-exponent 1 \
#     --lr 0.0002 \
#     --num-training-steps 200000 \
#     --lr-decay-every 200000 \
#     --noise-covariance 1 \
#     --warmup-steps 1000 \
#     --size-network large \
#     --no-frequency-component \
#     --spatial-component \


# torchrun --standalone --nnodes=1 --nproc-per-node=2 \
#     
# python main.py \
#     --name multigpu/combined_cov_tworandvars/tests_debug_song_cifar \
#     --dataset CIFAR10 \
#     --no-grayscale \
#     --train-batch-size 4 \
#     --test-batch-size 2 \
#     --network UNet \
#     --network-kwargs "{'num_scales':3,'group_size':1,'num_layers_encoder_block':3,'num_layers_mid_block':3,'num_layers_decoder_block':3}" \
#     --model EnergyModel \
#     --reparam-kwargs "{'conversion':'inner_product'}" \
#     --min-noise-level psnr=90 \
#     --max-noise-level psnr=-30 \
#     --noise-level-sampler UniformLog \
#     --mse-var-exponent -1 \
#     --train-noise-score 1 \
#     --noise-score-var-exponent 0 \
#     --lr 0.0002 \
#     --num-training-steps 400000 \
#     --lr-decay-every 10000 \
#     --noise-covariance 1 \
#     --warmup-steps 0 \
#     --size-network small \

# srun --gres=gpu:1 -C "a100-80gb|h100" -t 7-00:00:00 job_fg.sbatch --name CIFAR10-energy-3scales-unetcat-withemb --dataset CIFAR10 --no-grayscale --train-batch-size 128 --test-batch-size 32 --network UNet --network-kwargs "{'num_scales':3,'group_size':1,'num_layers_encoder_block':3,'num_layers_mid_block':3,'num_layers_decoder_block':3}" --model ColoredEnergyModel --reparam-kwargs "{'conversion':'inner_product'}" --min-noise-level psnr=90 --max-noise-level psnr=-30 --noise-level-sampler UniformLog --mse-var-exponent -1 --train-noise-score 1 --noise-score-var-exponent 0 --lr 0.0002 --num-training-steps 200000 --lr-decay-every 10000 --noise-covariance 1 --type-cov pink_noise
