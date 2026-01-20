#!/usr/bin/env python3
"""
使用预生成LR图像的数据集类，节省训练时的内存
"""

import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

class PairedRandomTransform:
    """确保LR和HR图像使用相同随机参数的变换类"""
    def __init__(self, crop_size, target_size, flip_prob=0.5, use_consistent_crop=False, 
                 consistent_crop_images=None):
        self.crop_size = crop_size
        self.target_size = target_size
        self.flip_prob = flip_prob
        self.use_consistent_crop = use_consistent_crop
        # 指定需要固定裁剪的图像列表（图像名称集合）
        self.consistent_crop_images = set(consistent_crop_images) if consistent_crop_images else set()
    
    def __call__(self, lr_img, hr_img, image_name=None):
        # 获取图像尺寸
        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size
        
        # 生成裁剪参数
        if (self.use_consistent_crop and image_name is not None and 
            (not self.consistent_crop_images or image_name in self.consistent_crop_images)):
            # 使用与偏好数据生成相同的固定裁剪策略
            # 只有当图像在指定列表中时才使用固定裁剪
            
            # 保存当前随机状态
            current_state = random.getstate()
            
            # 使用图像特定的种子
            seed = hash(image_name) % (2**32)
            random.seed(seed)
            
            max_left = max(0, min(lr_w, hr_w) - self.crop_size)
            max_top = max(0, min(lr_h, hr_h) - self.crop_size)
            
            if max_left > 0 and max_top > 0:
                crop_left = random.randint(0, max_left)
                crop_top = random.randint(0, max_top)
            else:
                # 如果图像太小，使用中心裁剪
                crop_left = max(0, (min(lr_w, hr_w) - self.crop_size) // 2)
                crop_top = max(0, (min(lr_h, hr_h) - self.crop_size) // 2)
            
            # 恢复原始随机状态，避免影响其他随机操作
            random.setstate(current_state)
        else:
            # 原有的随机裁剪策略
            crop_left = random.randint(0, max(0, min(lr_w, hr_w) - self.crop_size))
            crop_top = random.randint(0, max(0, min(lr_h, hr_h) - self.crop_size))
        
        # 对LR和HR图像应用相同的裁剪
        if lr_w >= self.crop_size and lr_h >= self.crop_size:
            lr_img = lr_img.crop((crop_left, crop_top, crop_left + self.crop_size, crop_top + self.crop_size))
        else:
            lr_img = lr_img.resize((self.crop_size, self.crop_size), Image.LANCZOS)
            
        if hr_w >= self.crop_size and hr_h >= self.crop_size:
            hr_img = hr_img.crop((crop_left, crop_top, crop_left + self.crop_size, crop_top + self.crop_size))
        else:
            hr_img = hr_img.resize((self.crop_size, self.crop_size), Image.LANCZOS)
        
        # 生成随机翻转决策（一次性生成，两个图像使用相同决策）
        should_flip = random.random() < self.flip_prob
        if should_flip:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Resize到目标尺寸
        lr_img = lr_img.resize((self.target_size, self.target_size), Image.LANCZOS)
        hr_img = hr_img.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        return lr_img, hr_img

class PairedSRPreGenDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        
        # 添加XPSR风格的质量提示选择参数
        self.gt_quality_prompt_ratio = getattr(args, 'gt_quality_prompt_ratio', 0.0)
        self.use_gt_quality_prompt = getattr(args, 'use_gt_quality_prompt', False)
        
        # 质量提示路径
        self.quality_prompt_path = getattr(args, 'quality_prompt_path', '/data2/Solar_Data/PiSA-SR/preset/lowlevel_prompt_q')
        self.quality_prompt_gt_path = getattr(args, 'quality_prompt_gt_path', '/data2/Solar_Data/PiSA-SR/preset/lowlevel_prompt_q_GT')
        
        if split == 'train':
            # 预生成的全尺寸LR图像目录
            self.lr_dir = "/data2/Solar_Data/PiSA-SR/preset/pre_generated_lr"
            
            # 读取原始GT图像路径（用作HR）
            with open(args.dataset_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]
            
            if args.highquality_dataset_txt_paths is not None:
                with open(args.highquality_dataset_txt_paths, 'r') as f:
                    self.hq_gt_list = [line.strip() for line in f.readlines()]
            
            # 检查预生成LR目录是否存在
            if os.path.exists(self.lr_dir):
                print(f"训练集使用预生成的全尺寸LR图像: {self.lr_dir}")
                print(f"训练集使用原始GT图像作为HR: {len(self.gt_list)} 个")
            else:
                print(f"警告：全尺寸预生成LR目录不存在: {self.lr_dir}")
                print("请先运行 python pre_generate_lr.py 生成全尺寸LR图像")
            
            # 获取需要固定裁剪的图像列表（从args中获取，已在训练脚本中读取）
            consistent_crop_images = getattr(args, 'consistent_crop_images_list', None)
            
            # 使用配对变换确保LR和HR图像的对应关系
            self.paired_transform = PairedRandomTransform(
                crop_size=args.resolution_ori,
                target_size=args.resolution_tgt,
                flip_prob=0.5,
                use_consistent_crop=getattr(args, 'use_consistent_crop', False),
                consistent_crop_images=consistent_crop_images
            )

        elif split == 'test':
            # 验证集使用预生成的LR-HR图像对
            self.input_folder = os.path.join(args.dataset_test_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(args.dataset_test_folder, "test_HR")
            self.lr_list = []
            self.gt_list = []
            
            # 获取并排序LR文件名
            lr_names = sorted(os.listdir(os.path.join(self.input_folder)))
            
            # 为每个LR文件找到对应的HR文件
            for lr_name in lr_names:
                if lr_name.endswith('.png') or lr_name.endswith('.jpg') or lr_name.endswith('.jpeg'):
                    # 从LR文件名中提取基础名称
                    base_name = os.path.splitext(lr_name)[0]
                    
                    # 构造可能的HR文件名列表
                    possible_hr_names = [
                        f"{base_name}_gt.png",
                        f"{base_name}_gt.jpg",
                        f"{base_name}_gt.jpeg",
                        f"{base_name}.png", # 尝试与LR同名
                        f"{base_name}.jpg",
                        f"{base_name}.jpeg"
                    ]
                    
                    # 查找存在的HR文件
                    found_hr_path = None
                    for hr_name in possible_hr_names:
                        hr_path = os.path.join(self.output_folder, hr_name)
                        if os.path.exists(hr_path):
                            found_hr_path = hr_path
                            break
                    
                    # 检查HR文件是否存在
                    if found_hr_path:
                        self.lr_list.append(os.path.join(self.input_folder, lr_name))
                        self.gt_list.append(found_hr_path)
                    else:
                        print(f"Warning: HR file not found for {lr_name}, expected one of {possible_hr_names}")
            
            print(f"Found {len(self.lr_list)} LR-HR pairs for validation")
            assert len(self.lr_list) == len(self.gt_list)
            
            # 验证集使用确定性变换（中心裁剪或直接resize）
            val_transforms = []
            if hasattr(args, 'resolution_ori') and args.resolution_ori:
                val_transforms.append(transforms.CenterCrop((args.resolution_ori, args.resolution_ori)))
            val_transforms.append(transforms.Resize((args.resolution_tgt, args.resolution_tgt)))
            self.val_transform = transforms.Compose(val_transforms)

    def __len__(self):
        return len(self.gt_list)

    def get_quality_prompt(self, image_name_or_index, use_gt=False):
        """
        Get quality prompt text for a given image.
        
        Args:
            image_name_or_index: Image filename (without extension) or index
            use_gt: Whether to use GT quality prompts (more detailed)
            
        Returns:
            Quality prompt text string
        """
        try:
            # Determine base path
            base_path = self.quality_prompt_gt_path if use_gt else self.quality_prompt_path
            
            # Handle both filename and index inputs
            if isinstance(image_name_or_index, (int, str)) and str(image_name_or_index).isdigit():
                # Numeric index - format to match file naming
                prompt_file = f"{int(image_name_or_index):05d}.txt"
            else:
                # String filename - ensure .txt extension
                prompt_file = str(image_name_or_index)
                if not prompt_file.endswith('.txt'):
                    prompt_file += '.txt'
            
            prompt_path = os.path.join(base_path, prompt_file)
            
            # Read quality prompt
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    quality_prompt = f.read().strip()
                return quality_prompt
            else:
                # Fallback to default quality prompt
                return "This image shows average quality with some details visible but could be improved."
                
        except Exception as e:
            return "This image shows average quality with some details visible but could be improved."

    def __getitem__(self, idx):
        if self.split == 'train':
            # 选择GT图像（用作HR）
            if hasattr(self, 'hq_gt_list') and self.hq_gt_list:
                if np.random.uniform() < self.args.prob:
                    gt_img_path = self.gt_list[idx]
                else:
                    idx = random.sample(range(0, len(self.hq_gt_list)), 1)
                    gt_img_path = self.hq_gt_list[idx[0]]
            else:
                gt_img_path = self.gt_list[idx]
            
            # 读取原始GT图像作为HR
            hr_img = Image.open(gt_img_path).convert('RGB')
            
            # 读取对应的预生成全尺寸LR图像
            img_name = os.path.basename(gt_img_path)
            lr_path = os.path.join(self.lr_dir, img_name)
            
            if os.path.exists(lr_path):
                lr_img = Image.open(lr_path).convert('RGB')
            else:
                print(f"警告: 预生成的LR图像不存在 {lr_path}，使用原始图像")
                lr_img = hr_img  # 使用原始图像作为fallback
            
            # 使用配对变换确保LR和HR的对应关系
            # 传递图像名称以支持一致裁剪
            img_name = os.path.basename(gt_img_path)
            lr_img, gt_img = self.paired_transform(lr_img, hr_img, img_name)
            
            # 转换为tensor并归一化
            img_t = F.to_tensor(lr_img)
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
            output_t = F.to_tensor(gt_img)
            output_t = F.normalize(output_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            # 质量提示：始终使用LR质量
            use_gt_quality = False  
            
            # 获取LR质量提示
            quality_prompt = self.get_quality_prompt(idx, use_gt=use_gt_quality)
            
            # 将质量提示添加到batch中
            example["quality_prompts"] = [quality_prompt]

            return example
            
        elif self.split == 'test':
            # 验证集使用确定性变换，确保LR和HR对应
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            
            # 对LR和HR图像应用相同的确定性变换
            img_t = self.val_transform(input_img)
            output_t = self.val_transform(output_img)
            
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            return example 