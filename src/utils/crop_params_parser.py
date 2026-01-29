#!/usr/bin/env python3
"""
裁剪参数解析器，用于解析processing_log.txt文件中的裁剪参数
确保DPO训练时使用与偏好数据生成时相同的裁剪区域
"""

import os
import re
from typing import Dict, Tuple, Optional

class CropParamsParser:
    """解析裁剪参数日志文件"""
    
    def __init__(self, log_file_path: str):
        """
        初始化裁剪参数解析器
        
        Args:
            log_file_path: processing_log.txt文件路径
        """
        self.log_file_path = log_file_path
        self.crop_params = {}
        self._parse_log_file()
    
    def _parse_log_file(self):
        """解析日志文件，提取裁剪参数"""
        if not os.path.exists(self.log_file_path):
            print(f"警告：裁剪参数文件不存在: {self.log_file_path}")
            return
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 解析格式：成功: 图像名.png (裁剪位置: x,y)
            pattern = r'成功:\s*([^(]+)\s*\(裁剪位置:\s*(\d+),(\d+)\)'
            
            for line in lines:
                line = line.strip()
                match = re.search(pattern, line)
                if match:
                    image_name = match.group(1).strip()
                    crop_x = int(match.group(2))
                    crop_y = int(match.group(3))
                    
                    # 移除扩展名，统一使用基础名称
                    base_name = os.path.splitext(image_name)[0]
                    self.crop_params[base_name] = (crop_x, crop_y)
            
            print(f"成功解析 {len(self.crop_params)} 个图像的裁剪参数")
            
        except Exception as e:
            print(f"解析裁剪参数文件时出错: {e}")
    
    def get_crop_params(self, image_name: str) -> Optional[Tuple[int, int]]:
        """
        获取指定图像的裁剪参数
        
        Args:
            image_name: 图像名称（可带或不带扩展名）
            
        Returns:
            (crop_x, crop_y) 或 None（如果未找到）
        """
        # 移除扩展名，统一使用基础名称
        base_name = os.path.splitext(image_name)[0]
        return self.crop_params.get(base_name)
    
    def has_crop_params(self, image_name: str) -> bool:
        """检查是否有指定图像的裁剪参数"""
        return self.get_crop_params(image_name) is not None
    
    def get_all_images(self) -> list:
        """获取所有有裁剪参数的图像名称列表"""
        return list(self.crop_params.keys())
    
    def __len__(self):
        """返回有裁剪参数的图像数量"""
        return len(self.crop_params)
    
    def print_stats(self):
        """打印统计信息"""
        print(f"裁剪参数统计:")
        print(f"  总图像数: {len(self.crop_params)}")
        if self.crop_params:
            # 示例显示前5个
            example_items = list(self.crop_params.items())[:5]
            print(f"  示例:")
            for name, (x, y) in example_items:
                print(f"    {name}: ({x}, {y})")
