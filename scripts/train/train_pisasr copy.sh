#!/bin/bash

# 设置日志文件路径
LOG_DIR="experiments/train-pisasr-uncertainty-new918/logs"
mkdir -p $LOG_DIR

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_log_${TIMESTAMP}.txt"

echo "=== PiSA-SR Training with Uncertainty Estimation ===" | tee -a $LOG_FILE
echo "Training started at: $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 启动训练
python train_pisasr.py \
--pretrained_model_path="/home/bd/super_resolution/sd/stable-diffusion-2-1-base" \
--pretrained_model_path_csd="/home/bd/super_resolution/sd/stable-diffusion-2-1-base" \
--dataset_txt_paths="preset/gt_path.txt" \
# --highquality_dataset_txt_paths="preset/gt_selected_path.txt" \
# 训练的时候看一下在测试集上效果什么样
# --dataset_test_folder="preset/testfolder_RealSR" \
--learning_rate=3e-5 \
--train_batch_size=1 \
--gradient_accumulation_steps=2 \
--enable_xformers_memory_efficient_attention \
--gradient_checkpointing \
--checkpointing_steps 500 \
--seed 123 \
--max_grad_norm=5.0 \
--output_dir="experiments/train-qusr-uncertainty-new918" \
--cfg_csd 7.5 \
--timesteps1 1 \
--lambda_lpips=2 \
--lambda_l2=0.5 \
--lambda_csd=2 \
--lora_rank_unet_sem=4 \
--min_dm_step_ratio=0.02 \
--max_dm_step_ratio=0.5 \
--null_text_ratio=0.0 \
--align_method="adain" \
--deg_file_path="params.yml" \
--tracker_project_name "QUSR_Uncertainty-new918" \
--max_train_steps 30001 \
--enable_uncertainty \
--uncertainty_hidden_channels=64 \
--uncertainty_channels=3 \
--min_noise=0.1 \
--un=0.3 \
--kappa=1.0 \
--lambda_uncertainty=0.3 \
--log_steps=10 \
2>&1 | tee -a $LOG_FILE

# 训练结束
echo "" | tee -a $LOG_FILE
echo "Training completed at: $(date)" | tee -a $LOG_FILE
echo "Final log saved to: $LOG_FILE" | tee -a $LOG_FILE
