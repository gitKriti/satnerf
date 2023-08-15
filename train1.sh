#!/usr/bin/env bash

export project_dir="/home/myid/kg23166/satnerf"
start_time=$(date +"%Y-%m-%d %H:%M:%S")
nohup time python3 main.py --model sat-nerf \
                           --exp_name JAX_004_ds1_sat-nerf \
                           --root_dir $project_dir/datasets/root_dir/crops_rpcs_ba_v2/JAX_004 \
                           --img_dir $project_dir/datasets/DFC2019/Track3-RGB-crops/JAX_004 \
                           --gt_dir $project_dir/datasets/DFC2019/Track3-Truth \
                           --cache_dir $project_dir/cache/crops_rpcs_ba_v2/JAX_004_ds1 \
                           --logs_dir $project_dir/logs \
                           --ckpts_dir $project_dir/checkpoints \
                           --gpu_id 0 > $project_dir/training_log_JAX_004.txt 2>&1 &
wait $!
end_time=$(date +"%Y-%m-%d %H:%M:%S")

# Calculate the training duration
start_seconds=$(date -d "$start_time" +%s)
end_seconds=$(date -d "$end_time" +%s)
duration=$((end_seconds - start_seconds))

# Log the training time duration
echo "Training on JAX_004 dataset took $duration seconds." >> $project_dir/training_duration_004.txt
