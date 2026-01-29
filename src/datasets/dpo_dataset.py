#!/usr/bin/env python3
"""
DPO偏好数据集加载器
用于加载偏好对比数据进行DPO微调训练
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..utils.crop_params_parser import CropParamsParser

class DPOPreferenceDataset(Dataset):
    """DPO偏好对比数据集"""
    
    def __init__(self, 
                 preference_pairs_file: str,
                 lr_dpo_dir: str,
                 crop_params_file: str,
                 target_size: int = 512,
                 crop_size: int = 512):
        """
        初始化DPO偏好数据集
        
        Args:
            preference_pairs_file: 偏好对比数据JSON文件路径
            lr_dpo_dir: DPO LR图像目录
            crop_params_file: 裁剪参数文件路径
            target_size: 目标图像尺寸
            crop_size: 裁剪尺寸
        """
        self.preference_pairs_file = preference_pairs_file
        self.lr_dpo_dir = lr_dpo_dir
        self.target_size = target_size
        self.crop_size = crop_size
        
        # 初始化裁剪参数解析器
        self.crop_parser = CropParamsParser(crop_params_file)
        
        # 加载偏好对比数据
        self.preference_pairs = self._load_preference_pairs()
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
        
        print(f"DPO数据集初始化完成:")
        print(f"  偏好对数量: {len(self.preference_pairs)}")
        print(f"  裁剪参数数量: {len(self.crop_parser)}")
        print(f"  LR图像目录: {lr_dpo_dir}")
    
    def _load_preference_pairs(self) -> List[Dict]:
        """加载偏好对比数据"""
        try:
            with open(self.preference_pairs_file, 'r', encoding='utf-8') as f:
                pairs = json.load(f)
            
            # 验证数据格式
            valid_pairs = []
            for i, pair in enumerate(pairs):
                if self._validate_preference_pair(pair):
                    valid_pairs.append(pair)
                else:
                    print(f"警告：跳过无效的偏好对 {i}")
            
            print(f"成功加载 {len(valid_pairs)} 个有效偏好对")
            return valid_pairs
            
        except Exception as e:
            print(f"加载偏好数据时出错: {e}")
            return []
    
    def _validate_preference_pair(self, pair: Dict) -> bool:
        """验证偏好对数据格式"""
        required_keys = ['chosen', 'rejected']
        for key in required_keys:
            if key not in pair:
                return False
            if 'image_path' not in pair[key]:
                return False
        return True
    
    def _extract_image_name_from_path(self, image_path: str) -> str:
        """从图像路径中提取图像名称"""
        return os.path.splitext(os.path.basename(image_path))[0]
    
    def _load_and_crop_lr_image(self, image_name: str) -> Optional[Image.Image]:
        """加载并裁剪LR图像"""
        try:
            # 构建LR图像路径
            lr_path = os.path.join(self.lr_dpo_dir, f"{image_name}.png")
            if not os.path.exists(lr_path):
                # 尝试其他扩展名
                for ext in ['.jpg', '.jpeg']:
                    lr_path = os.path.join(self.lr_dpo_dir, f"{image_name}{ext}")
                    if os.path.exists(lr_path):
                        break
                else:
                    print(f"警告：未找到LR图像: {image_name}")
                    return None
            
            # 加载图像
            lr_image = Image.open(lr_path).convert('RGB')
            
            # 获取裁剪参数
            crop_params = self.crop_parser.get_crop_params(image_name)
            if crop_params is None:
                print(f"警告：未找到裁剪参数: {image_name}")
                # 使用中心裁剪作为备选
                w, h = lr_image.size
                crop_x = max(0, (w - self.crop_size) // 2)
                crop_y = max(0, (h - self.crop_size) // 2)
            else:
                crop_x, crop_y = crop_params
            
            # 执行裁剪
            cropped = lr_image.crop((
                crop_x, crop_y, 
                crop_x + self.crop_size, 
                crop_y + self.crop_size
            ))
            
            return cropped
            
        except Exception as e:
            print(f"加载LR图像时出错 {image_name}: {e}")
            return None
    
    def _load_sr_image(self, image_path: str) -> Optional[Image.Image]:
        """加载SR图像"""
        try:
            if not os.path.exists(image_path):
                print(f"警告：SR图像不存在: {image_path}")
                return None
            
            sr_image = Image.open(image_path).convert('RGB')
            return sr_image
            
        except Exception as e:
            print(f"加载SR图像时出错 {image_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.preference_pairs)
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        """获取一个偏好对比样本"""
        try:
            pair = self.preference_pairs[idx]
            
            # 提取图像名称
            chosen_path = pair['chosen']['image_path']
            rejected_path = pair['rejected']['image_path']
            
            chosen_name = self._extract_image_name_from_path(chosen_path)
            rejected_name = self._extract_image_name_from_path(rejected_path)
            
            # 确保chosen和rejected来自同一张LR图像
            if chosen_name != rejected_name:
                print(f"警告：偏好对图像名称不匹配: {chosen_name} vs {rejected_name}")
                return None
            
            # 加载LR图像
            lr_image = self._load_and_crop_lr_image(chosen_name)
            if lr_image is None:
                return None
            
            # 加载chosen和rejected SR图像
            chosen_sr = self._load_sr_image(chosen_path)
            rejected_sr = self._load_sr_image(rejected_path)
            
            if chosen_sr is None or rejected_sr is None:
                return None
            
            # 应用变换
            lr_tensor = self.transform(lr_image)
            chosen_tensor = self.transform(chosen_sr)
            rejected_tensor = self.transform(rejected_sr)
            
            # 标准化到[-1, 1]
            lr_tensor = lr_tensor * 2.0 - 1.0
            chosen_tensor = chosen_tensor * 2.0 - 1.0
            rejected_tensor = rejected_tensor * 2.0 - 1.0
            
            # 验证chosen和rejected张量是否相同
            if torch.equal(chosen_tensor, rejected_tensor):
                print(f"警告：在样本索引 {idx} (图像: {chosen_name}) 中，chosen 和 rejected 的图像内容完全相同。")
            
            # 获取质量分数
            chosen_score = pair['chosen'].get('composite_score', 0.0)
            rejected_score = pair['rejected'].get('composite_score', 0.0)
            margin = pair.get('margin', chosen_score - rejected_score)
            
            return {
                'lr': lr_tensor,
                'chosen': chosen_tensor,
                'rejected': rejected_tensor,
                'chosen_score': float(chosen_score),
                'rejected_score': float(rejected_score),
                'margin': float(margin),
                'image_name': chosen_name,
                'chosen_config': pair['chosen'].get('config', ''),
                'rejected_config': pair['rejected'].get('config', '')
            }
            
        except Exception as e:
            print(f"获取样本时出错 (idx={idx}): {e}")
            return None

def collate_dpo_batch(batch):
    """DPO批次整理函数，过滤无效样本"""
    # 过滤None样本
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        return None
    
    # 整理批次数据
    batch_data = {
        'lr': torch.stack([item['lr'] for item in valid_batch]),
        'chosen': torch.stack([item['chosen'] for item in valid_batch]),
        'rejected': torch.stack([item['rejected'] for item in valid_batch]),
        'chosen_score': torch.tensor([item['chosen_score'] for item in valid_batch]),
        'rejected_score': torch.tensor([item['rejected_score'] for item in valid_batch]),
        'margin': torch.tensor([item['margin'] for item in valid_batch]),
        'image_names': [item['image_name'] for item in valid_batch],
        'chosen_configs': [item['chosen_config'] for item in valid_batch],
        'rejected_configs': [item['rejected_config'] for item in valid_batch]
    }
    
    return batch_data
