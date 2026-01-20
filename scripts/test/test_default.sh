
python test_pisasr.py \
--pretrained_model_path preset/models/stable-diffusion-2-1-base \
--pretrained_path experiments/train-pisasr-uncertainty-new918/checkpoints/model_15001.pkl \
--process_size 512 \
--upscale 4 \
--input_image preset/test_datasets/RealSR_test/test_SR_bicubic \
--output_dir experiments/test1w5_ur_real_f-new918 \
--quality_prompt_path preset/test_lowlevel_prompt_q_RealSR \
--default

python scripts/test/test_mertic.py \
    --inp_imgs /data2/Solar_Data/PiSA-SR/experiments/test1w5_ur_real_f-new918 \
    --gt_imgs /data2/Solar_Data/PiSA-SR/preset/test_datasets/RealSR_test/test_HR \
    --log /data2/Solar_Data/PiSA-SR/logs \
    --log_name PiSA_SR_Test_finally