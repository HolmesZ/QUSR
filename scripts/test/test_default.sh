
#!/bin/bash

# 配置路径（与 train_pisasr.sh 保持一致）
SD_DIR="/home/bd/super_resolution/sd/stable-diffusion-2-1-base"
DATASET_BASE_DIR="/home/bd/super_resolution/text-qusr/dataset/qusr-dataset-256"

# 实验目录（修改为你要测试的模型所在目录）
EXPERIMENT_DIR="experiments/train-qusr-uncertainty-0124-211151"
MODEL_STEP="30001"  # 修改为要测试的 checkpoint 步数

# 输出目录
OUTPUT_DIR="${EXPERIMENT_DIR}/test_step${MODEL_STEP}"

echo "=== PiSA-SR Testing ===" 
echo "Testing model: ${EXPERIMENT_DIR}/checkpoints/model_${MODEL_STEP}.pkl"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# 禁用 bitsandbytes triton 以避免兼容性问题
export BITSANDBYTES_NOWELCOME=1
export BNB_CUDA_VERSION=0

# 1. 运行超分辨率推理
CUDA_VISIBLE_DEVICES="0" accelerate launch test_pisasr.py \
    --pretrained_model_path "$SD_DIR" \
    --pretrained_path "${EXPERIMENT_DIR}/checkpoints/model_${MODEL_STEP}.pkl" \
    --process_size 256 \
    --upscale 4 \
    --input_image "$DATASET_BASE_DIR/testfolder/test_LR" \
    --output_dir "$OUTPUT_DIR" \
    --quality_prompt_path "$DATASET_BASE_DIR/lowlevel_prompt_q" \
    --default

# 2. 计算评估指标
python scripts/test/test_mertic.py \
    --inp_imgs "$OUTPUT_DIR" \
    --gt_imgs "$DATASET_BASE_DIR/testfolder/test_HR" \
    --log "${EXPERIMENT_DIR}/logs" \
    --log_name "test_step${MODEL_STEP}"

echo ""
echo "Testing completed!"