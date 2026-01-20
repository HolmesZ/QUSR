import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import sys
import logging
from datetime import datetime

# 禁用SSL证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 添加IQA-PyTorch路径
sys.path.append('/data2/Solar_Data/PiSA-SR/IQA-PyTorch-main')
import pyiqa

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from pisasr import CSDLoss, PiSASR
from src.my_utils.training_utils import parse_args  
from src.datasets.dataset_pregen import PairedSRPreGenDataset

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import random

def main(args):
    # 设置日志记录
    log_dir = Path(args.output_dir, "logs")
    log_dir.mkdir(exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_{timestamp}.txt"
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== PiSA-SR Training with Uncertainty Estimation ===")
    logger.info(f"Training started at: {datetime.now()}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("")
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # Enable dual attention for enhanced training
    # 禁用双重注意力，只使用单一注意力
    args.enable_dual_attention = False

    
    net_pisasr = PiSASR(args)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pisasr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pisasr.unet.enable_gradient_checkpointing()
        
    # Re-setup dual attention processors after xformers setup
    if args.enable_dual_attention and hasattr(net_pisasr.unet, '_setup_dual_attention_processors'):
        net_pisasr.unet._setup_dual_attention_processors()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # init CSDLoss model
    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # 初始化IQA评估指标
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device='cuda')
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr', device='cuda')
    lpips_metric = pyiqa.create_metric('lpips', device='cuda')
    dists_metric = pyiqa.create_metric('dists', device='cuda')
    clipiqa_metric = pyiqa.create_metric('clipiqa', device='cuda')
    niqe_metric = pyiqa.create_metric('niqe', device='cuda')
    musiq_metric = pyiqa.create_metric('musiq', device='cuda')
    fid_metric = pyiqa.create_metric('fid', device='cuda')

    # # set gen adapter - pixel
    # net_pisasr.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix'])
    # net_pisasr.set_train_pix() # first to remove degradation
    
    # 双重注意力联合训练：从一开始就同时使用语义和质量提示
    if args.enable_dual_attention:
        pass
    
    # 质量提示训练（只使用质量相关的lora）
    net_pisasr.unet.set_adapter(['default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
    net_pisasr.set_train_sem()

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pisasr.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    
    # Add uncertainty estimator parameters to optimizer if enabled
    if hasattr(net_pisasr, 'enable_uncertainty') and net_pisasr.enable_uncertainty:
        for _p in net_pisasr.uncertainty_estimator.parameters():
            layers_to_opt.append(_p)
        print(f"Added {sum(p.numel() for p in net_pisasr.uncertainty_estimator.parameters())} uncertainty estimator parameters to optimizer")

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    
    # 如果从checkpoint恢复，设置学习率调度器的步数
    if args.resume_ckpt is not None:
        ckpt_filename = os.path.basename(args.resume_ckpt)
        if ckpt_filename.startswith("model_") and ckpt_filename.endswith(".pkl"):
            try:
                resume_step = int(ckpt_filename.replace("model_", "").replace(".pkl", ""))
                # 设置学习率调度器到正确的步数
                for _ in range(resume_step):
                    lr_scheduler.step()
            except ValueError:
                pass
    
    # initialize the dataset 
    if getattr(args, 'use_online_degradation', False):
        # 使用在线降质
        from src.datasets.dataset import PairedSROnlineTxtDataset
        dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
        dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    else:
        # 使用全尺寸预生成数据集
        from src.datasets.dataset_pregen import PairedSRPreGenDataset
        
        # 如果指定了固定裁剪图像列表，读取图像名称并设置到args中
        if args.consistent_crop_images and os.path.exists(args.consistent_crop_images):
            with open(args.consistent_crop_images, 'r') as f:
                consistent_crop_images = [line.strip() for line in f.readlines()]
            # 将图像列表设置到args中，供数据集使用
            args.consistent_crop_images_list = consistent_crop_images
        
        dataset_train = PairedSRPreGenDataset(split="train", args=args)
        dataset_val = PairedSRPreGenDataset(split="test", args=args)
        
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    

    # init RAM for text prompt extractor - COMMENTED OUT FOR QUALITY-ONLY TRAINING
    # from ram.models.ram_lora import ram
    # from ram import inference_ram as inference
    # ram_transforms = transforms.Compose([
    #     transforms.Resize((384, 384)),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # RAM = ram(pretrained='src/ram_pretrain_model/ram_swin_large_14m.pth',
    #         pretrained_condition=None,
    #         image_size=384,
    #         vit='swin_l')
    # RAM.eval()
    # RAM.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    net_pisasr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_pisasr, optimizer, dl_train, lr_scheduler
    )
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # Add dual attention configuration to tracking
        tracker_config["dual_attention_enabled"] = False
        tracker_config["training_strategy"] = "quality_only_training"
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",disable=not accelerator.is_local_main_process,)
    # 初始化global_step
    global_step = 0
    if args.resume_ckpt is not None:
        # 从checkpoint文件名中提取步数
        ckpt_filename = os.path.basename(args.resume_ckpt)
        if ckpt_filename.startswith("model_") and ckpt_filename.endswith(".pkl"):
            try:
                global_step = int(ckpt_filename.replace("model_", "").replace(".pkl", ""))
            except ValueError:
                global_step = 0

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    # global_step = 0
    should_stop = False
    lambda_l2 = args.lambda_l2
    # lambda_lpips = 0
    # lambda_csd = 0
    # 语义训练
    lambda_lpips = args.lambda_lpips
    lambda_csd = args.lambda_csd
    
    # 日志记录质量提示策略
    
    # 根据当前步数决定是否应该继续pixel训练
    # if args.resume_ckpt is not None:
    #     if global_step < args.pix_steps:
    #         # 如果恢复的步数小于pix_steps，继续pixel训练
    #         print(f"从步数 {global_step} 恢复，继续pixel训练直到步数 {args.pix_steps}")
    #     else:
    #         # 如果恢复的步数已经超过pix_steps，直接进入semantic训练
    #         print(f"从步数 {global_step} 恢复，直接进入semantic训练")
    #         lambda_lpips = args.lambda_lpips
    #         lambda_csd = args.lambda_csd
    #         # 设置semantic训练模式
    #         if hasattr(net_pisasr, 'module'):
    #             net_pisasr.module.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix','default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
    #             net_pisasr.module.set_train_sem() 
    #         else:
    #             net_pisasr.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix','default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
    #             net_pisasr.set_train_sem()

    for epoch in range(0, args.num_training_epochs):
        if should_stop:
            break
        for step, batch in enumerate(dl_train):
                with accelerator.accumulate(net_pisasr):
                    x_src = batch["conditioning_pixel_values"]
                    x_tgt = batch["output_pixel_values"]

                    # get text prompts from GT - COMMENTED OUT FOR QUALITY-ONLY TRAINING
                    # Use default semantic prompt instead of RAM-generated captions
                    # x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                    # caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                    # batch["prompt"] = [f'{each_caption}, {args.pos_prompt_csd}' for each_caption in caption]
                    batch_size = x_src.shape[0]  # Get batch size from input tensor
                    batch["prompt"] = ["A high quality image with good details and clarity." for _ in range(batch_size)]
                    
                # if global_step == args.pix_steps:
                #     # begin the semantic optimization
                #     if hasattr(net_pisasr, 'module'):
                #         net_pisasr.module.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix','default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
                #         net_pisasr.module.set_train_sem() 
                #     else:
                #         net_pisasr.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix','default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
                #         net_pisasr.set_train_sem()
                #     
                #     lambda_l2 = args.lambda_l2
                #     lambda_lpips = args.lambda_lpips
                #     lambda_csd = args.lambda_csd
                    
                # set runtime flag for warmup into args (consumed in model.forward)
                setattr(args, "use_uncertainty_now", global_step >= 0)
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds, uncertainty_map = net_pisasr(x_src, x_tgt, batch=batch, args=args)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * lambda_lpips
                loss = loss_l2 + loss_lpips
                
                # Uncertainty warmup: disable uncertainty for first 1000 steps
                # NOTE: global_step is updated after optimizer step; here it reflects the previous step count
                use_uncertainty = (
                    hasattr(net_pisasr.module, 'enable_uncertainty') and
                    net_pisasr.module.enable_uncertainty and
                    uncertainty_map is not None and
                    global_step >= 0
                )

                loss_uncertainty = torch.tensor(0.0, device=x_tgt_pred.device)
                if use_uncertainty:
                    loss_uncertainty = net_pisasr.module.compute_uncertainty_loss(x_tgt_pred, x_tgt, uncertainty_map)
                    # Apply uncertainty loss weight (matching UDR-S2Former: weight=1.0)
                    loss = loss + loss_uncertainty * args.lambda_uncertainty
                
                # reg loss
                loss_csd = net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args, ) * lambda_csd
                loss = loss + loss_csd
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # 初始化logs变量（所有进程都需要）
                    logs = {}
                    
                    if accelerator.is_main_process:
                        # log all the losses
                        logs["loss_csd"] = loss_csd.detach().item()
                        logs["loss_l2"] = loss_l2.detach().item()
                        logs["loss_lpips"] = loss_lpips.detach().item()
                        
                        # Log uncertainty loss only when enabled after warmup
                        if use_uncertainty:
                            logs["loss_uncertainty"] = loss_uncertainty.detach().item()
                        
                        progress_bar.set_postfix(**logs)
                        
                        # 记录每个step的详细信息到日志文件
                        step_log = f"Step {global_step}: " + " | ".join([f"{k}: {v:.6f}" for k, v in logs.items()])
                        logger.info(step_log)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pisasr).save_model(outf)

                    # test
                    if global_step % args.eval_freq == 1:
                        os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        
                        # 用于累积所有指标值
                        psnr_values = []
                        ssim_values = []
                        lpips_values = []
                        dists_values = []
                        clipiqa_values = []
                        niqe_values = []
                        musiq_values = []
                        
                        for step, batch_val in enumerate(dl_val):
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            x_basename = batch_val["base_name"][0]
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # Set model to eval mode for validation
                                net_pisasr.eval()
                                
                                # get text prompts from LR - COMMENTED OUT FOR QUALITY-ONLY TRAINING
                                # Use default semantic prompt instead of RAM-generated captions
                                # x_src_ram = ram_transforms(x_src * 0.5 + 0.5)
                                # caption = inference(x_src_ram.to(dtype=torch.float16), RAM)
                                # batch_val["prompt"] = caption
                                batch_val["prompt"] = ["A high quality image with good details and clarity."]
                                # forward pass
                                x_tgt_pred, latents_pred, _, _, uncertainty_map = accelerator.unwrap_model(net_pisasr)(x_src, x_tgt,
                                                                                                      batch=batch_val,
                                                                                                      args=args)
                                
                                # 计算所有指标 (输入范围[-1,1]，转换为[0,1])
                                pred_norm = x_tgt_pred * 0.5 + 0.5
                                gt_norm = x_tgt * 0.5 + 0.5

                                # 计算需要参考图像的指标
                                psnr_val = psnr_metric(pred_norm, gt_norm).item()
                                ssim_val = ssim_metric(pred_norm, gt_norm).item()
                                lpips_val = lpips_metric(pred_norm, gt_norm).item()
                                dists_val = dists_metric(pred_norm, gt_norm).item()

                                # 计算只需要预测图像的指标
                                clipiqa_val = clipiqa_metric(pred_norm).item()
                                niqe_val = niqe_metric(pred_norm).item()
                                musiq_val = musiq_metric(pred_norm).item()

                                # 累积指标值
                                psnr_values.append(psnr_val)
                                ssim_values.append(ssim_val)
                                lpips_values.append(lpips_val)
                                dists_values.append(dists_val)
                                clipiqa_values.append(clipiqa_val)
                                niqe_values.append(niqe_val)
                                musiq_values.append(musiq_val)
                                
                                # save the output
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                input_image = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                                if args.align_method == 'adain':
                                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                                elif args.align_method == 'wavelet':
                                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                                else:
                                    pass
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}")
                                output_pil.save(outf)
                                
                                # Save uncertainty map if available
                                if uncertainty_map is not None:
                                    # Convert uncertainty map to visualization
                                    uncertainty_vis = uncertainty_map[0].cpu()  # Take first sample
                                    
                                    # For visualization: use raw uncertainty values (no sigmoid)
                                    # Training still uses sigmoid in compute_uncertainty_loss
                                    uncertainty_vis = uncertainty_vis  # Keep raw values for better visualization
                                    
                                    # Convert multi-channel uncertainty to single channel heatmap
                                    if uncertainty_vis.shape[0] > 1:  # Multi-channel
                                        # Take the mean across channels to get overall uncertainty
                                        uncertainty_vis = uncertainty_vis.mean(0, keepdim=True)
                                    
                                    # Convert to single channel heatmap
                                    uncertainty_vis = uncertainty_vis.squeeze(0)  # Remove channel dim: [H, W]
                                    
                                    # Normalize to [0, 1] for colormap
                                    uncertainty_min = uncertainty_vis.min()
                                    uncertainty_max = uncertainty_vis.max()
                                    uncertainty_std = uncertainty_vis.std()
                                    
                                    # Debug: Print uncertainty statistics
                                    print(f"Uncertainty Debug - Min: {uncertainty_min:.6f}, Max: {uncertainty_max:.6f}, Std: {uncertainty_std:.6f}, Range: {uncertainty_max - uncertainty_min:.6f}")
                                    
                                    if uncertainty_max > uncertainty_min:
                                        uncertainty_vis = (uncertainty_vis - uncertainty_min) / (uncertainty_max - uncertainty_min)
                                    else:
                                        # If all values are the same, create a small variation for visualization
                                        print("Warning: All uncertainty values are identical!")
                                        uncertainty_vis = torch.ones_like(uncertainty_vis) * 0.5
                                    
                                    # Apply colormap (using matplotlib-style colormap)
                                    import matplotlib.pyplot as plt
                                    import matplotlib.cm as cm
                                    
                                    # Convert to numpy for colormap
                                    uncertainty_np = uncertainty_vis.cpu().numpy()
                                    
                                    # Choose colormap (jet is classic for uncertainty visualization)
                                    # Options: 'jet', 'hot', 'viridis', 'plasma', 'inferno'
                                    colormap = cm.jet  # Classic blue->green->red heatmap
                                    
                                    # Apply colormap with better contrast
                                    colored = colormap(uncertainty_np)  # Returns RGBA
                                    colored_rgb = colored[:, :, :3]     # Take only RGB, drop alpha
                                    
                                    # Enhance contrast for better visualization
                                    colored_rgb = np.clip(colored_rgb * 1.2, 0, 1)  # Slightly enhance contrast
                                    
                                    # Convert back to PIL Image
                                    colored_rgb = (colored_rgb * 255).astype(np.uint8)
                                    uncertainty_pil = transforms.ToPILImage()(torch.from_numpy(colored_rgb).permute(2, 0, 1))
                                    
                                    # Save uncertainty map
                                    uncertainty_path = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename.split('.')[0]}_uncertainty.png")
                                    uncertainty_pil.save(uncertainty_path)
                                    
                        # 计算所有指标的平均值
                        avg_psnr = np.mean(psnr_values)
                        avg_ssim = np.mean(ssim_values)
                        avg_lpips = np.mean(lpips_values)
                        avg_dists = np.mean(dists_values)
                        avg_clipiqa = np.mean(clipiqa_values)
                        avg_niqe = np.mean(niqe_values)
                        avg_musiq = np.mean(musiq_values)

                        # 计算FID（需要整个目录）
                        eval_output_dir = os.path.join(args.output_dir, "eval", f"fid_{global_step}")
                        # 获取验证集的ground truth路径（需要根据实际情况调整）
                        gt_dir = args.dataset_test_folder if hasattr(args, 'dataset_test_folder') else "preset/testfolder"

                        try:
                            fid_value = fid_metric(gt_dir, eval_output_dir).item()
                        except Exception as e:
                            fid_value = 0.0

                        # 添加到日志中
                        logs["avg_psnr"] = avg_psnr
                        logs["avg_ssim"] = avg_ssim
                        logs["avg_lpips"] = avg_lpips
                        logs["avg_dists"] = avg_dists
                        logs["avg_clipiqa"] = avg_clipiqa
                        logs["avg_niqe"] = avg_niqe
                        logs["avg_musiq"] = avg_musiq
                        logs["fid"] = fid_value

                        if accelerator.is_main_process:
                            # 控制台输出
                            print(f"验证结果 - 步骤 {global_step}:")
                            print(f"  PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
                            print(f"  LPIPS: {avg_lpips:.4f}, DISTS: {avg_dists:.4f}")
                            print(f"  CLIPIQA: {avg_clipiqa:.4f}, NIQE: {avg_niqe:.4f}")
                            print(f"  MUSIQ: {avg_musiq:.4f}, FID: {fid_value:.4f}")
                            
                            # 记录到日志文件
                            logger.info(f"Validation Results - Step {global_step}:")
                            logger.info(f"  PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
                            logger.info(f"  LPIPS: {avg_lpips:.4f}, DISTS: {avg_dists:.4f}")
                            logger.info(f"  CLIPIQA: {avg_clipiqa:.4f}, NIQE: {avg_niqe:.4f}")
                            logger.info(f"  MUSIQ: {avg_musiq:.4f}, FID: {fid_value:.4f}")
                        
                        # Reset model to training mode
                        net_pisasr.train()
                        
                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

                # stop when reaching max_train_steps
                if global_step >= args.max_train_steps:
                    should_stop = True
                    break
    
    # 训练结束日志记录
    logger.info("Training completed successfully!")
    logger.info(f"Final global step: {global_step}")
    logger.info(f"Training completed at: {datetime.now()}")
    logger.info(f"Final log saved to: {log_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    