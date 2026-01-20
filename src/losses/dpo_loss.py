#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) 损失函数实现
用于基于偏好数据优化模型生成结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from typing import Dict, Optional, Tuple

class DPOLoss(nn.Module):
    """DPO损失函数"""
    
    def __init__(self, 
                 beta: float = 0.1,
                 use_reference_model: bool = False,
                 reward_type: str = 'lpips',
                 device: str = 'cuda'):
        """
        初始化DPO损失函数
        
        Args:
            beta: DPO温度参数，控制偏好强度
            use_reference_model: 是否使用参考模型计算KL散度
            reward_type: 奖励函数类型 ('lpips', 'mse', 'composite')
            device: 设备
        """
        super().__init__()
        self.beta = beta
        self.use_reference_model = use_reference_model
        self.reward_type = reward_type
        self.device = device
        
        # 初始化奖励函数
        self._init_reward_function()
        
        print(f"DPO损失函数初始化:")
        print(f"  Beta: {beta}")
        print(f"  奖励类型: {reward_type}")
        print(f"  使用参考模型: {use_reference_model}")
    
    def _init_reward_function(self):
        """初始化奖励函数"""
        if self.reward_type == 'lpips':
            self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_loss.requires_grad_(False)
        elif self.reward_type == 'mse':
            self.mse_loss = nn.MSELoss()
        elif self.reward_type == 'composite':
            # 复合奖励函数，结合多个指标
            self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_loss.requires_grad_(False)
            self.mse_loss = nn.MSELoss()
    
    def compute_reward(self, 
                      generated: torch.Tensor, 
                      reference: torch.Tensor) -> torch.Tensor:
        """
        计算奖励分数
        
        Args:
            generated: 生成的图像 [B, C, H, W]
            reference: 参考图像 [B, C, H, W]
            
        Returns:
            奖励分数 [B]
        """
        if self.reward_type == 'lpips':
            # LPIPS距离越小，奖励越高
            lpips_dist = self.lpips_loss(generated, reference)
            reward = -lpips_dist.squeeze()  # 转换为奖励（距离越小奖励越高）
            
        elif self.reward_type == 'mse':
            # MSE损失越小，奖励越高
            mse_dist = self.mse_loss(generated, reference)
            reward = -mse_dist  # 转换为奖励
            
        elif self.reward_type == 'composite':
            # 复合奖励：结合LPIPS和MSE
            lpips_dist = self.lpips_loss(generated, reference).squeeze()
            mse_dist = F.mse_loss(generated, reference, reduction='none')
            mse_dist = mse_dist.mean(dim=[1, 2, 3])  # [B]
            
            # 归一化并组合
            lpips_reward = -lpips_dist
            mse_reward = -mse_dist
            reward = 0.7 * lpips_reward + 0.3 * mse_reward
            
        else:
            raise ValueError(f"不支持的奖励类型: {self.reward_type}")
        
        return reward
    
    def forward(self, 
                model_output: torch.Tensor,
                chosen: torch.Tensor,
                rejected: torch.Tensor,
                reference_chosen_reward: Optional[torch.Tensor] = None,
                reference_rejected_reward: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算DPO损失
        
        Args:
            model_output: 模型生成的图像 [B, C, H, W]
            chosen: 偏好的参考图像 [B, C, H, W]
            rejected: 不偏好的参考图像 [B, C, H, W]
            reference_chosen_reward: 参考模型对chosen的奖励（可选）
            reference_rejected_reward: 参考模型对rejected的奖励（可选）
            
        Returns:
            包含损失和统计信息的字典
        """
        batch_size = model_output.size(0)
        
        # 计算当前模型的奖励
        chosen_reward = self.compute_reward(model_output, chosen)
        rejected_reward = self.compute_reward(model_output, rejected)
        
        # 计算奖励差异
        reward_diff = chosen_reward - rejected_reward
        
        # 如果使用参考模型，计算KL散度正则化项
        if self.use_reference_model and reference_chosen_reward is not None:
            ref_reward_diff = reference_chosen_reward - reference_rejected_reward
            # KL散度正则化
            kl_penalty = reward_diff - ref_reward_diff
        else:
            kl_penalty = torch.zeros_like(reward_diff)
        
        # 计算DPO损失
        # loss = -log(sigmoid(beta * (chosen_reward - rejected_reward)))
        logits = self.beta * (reward_diff - kl_penalty)
        dpo_loss = -F.logsigmoid(logits).mean()
        
        # 计算偏好准确率（模型是否正确偏好chosen）
        preference_accuracy = (reward_diff > 0).float().mean()
        
        # 统计信息
        stats = {
            'dpo_loss': dpo_loss,
            'chosen_reward': chosen_reward.mean(),
            'rejected_reward': rejected_reward.mean(),
            'reward_diff': reward_diff.mean(),
            'preference_accuracy': preference_accuracy,
            'reward_margin': reward_diff.abs().mean()
        }
        
        return stats

class AdaptiveDPOLoss(DPOLoss):
    """自适应DPO损失，动态调整beta参数"""
    
    def __init__(self, 
                 beta_init: float = 0.1,
                 beta_min: float = 0.01,
                 beta_max: float = 1.0,
                 adaptation_rate: float = 0.01,
                 **kwargs):
        super().__init__(beta=beta_init, **kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.adaptation_rate = adaptation_rate
        self.step_count = 0
        
    def update_beta(self, preference_accuracy: float):
        """根据偏好准确率动态调整beta"""
        self.step_count += 1
        
        # 如果准确率太高，降低beta；如果太低，提高beta
        target_accuracy = 0.7  # 目标准确率
        accuracy_diff = preference_accuracy - target_accuracy
        
        # 动态调整beta
        beta_adjustment = -self.adaptation_rate * accuracy_diff
        self.beta = torch.clamp(
            torch.tensor(self.beta + beta_adjustment),
            self.beta_min, 
            self.beta_max
        ).item()
    
    def forward(self, *args, **kwargs):
        """前向传播，包含beta自适应调整"""
        stats = super().forward(*args, **kwargs)
        
        # 更新beta
        self.update_beta(stats['preference_accuracy'].item())
        stats['current_beta'] = self.beta
        
        return stats

def create_dpo_loss(loss_type: str = 'standard', **kwargs) -> nn.Module:
    """
    创建DPO损失函数
    
    Args:
        loss_type: 损失类型 ('standard', 'adaptive')
        **kwargs: 其他参数
        
    Returns:
        DPO损失函数实例
    """
    if loss_type == 'standard':
        return DPOLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveDPOLoss(**kwargs)
    else:
        raise ValueError(f"不支持的DPO损失类型: {loss_type}")
