import os
import sys
import time
import random
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules, ModulesToSaveWrapper
from peft.utils.other import transpose

sys.path.append(os.getcwd())
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.dual_unet_2d_condition import DualUNet2DConditionModel, create_dual_unet_from_pretrained
from src.my_utils.vaehook import VAEHook


import glob
def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None


import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# def initialize_unet(rank_sem, return_lora_module_names=False, pretrained_model_path=None):
#     unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
def initialize_unet(rank_sem, return_lora_module_names=False, pretrained_model_path=None, enable_dual_attention=True):
    if enable_dual_attention:
        # Create dual UNet with dual attention mechanism
        unet = create_dual_unet_from_pretrained(
            pretrained_model_path=pretrained_model_path,
            enable_dual_attention=True
        )
    else:
        # Use standard UNet
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    
    unet.requires_grad_(False)
    unet.train()

    # l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix = [], [], []
    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                # l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                # l_target_modules_decoder_pix.append(n.replace(".weight",""))
                l_target_modules_decoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n:
                # l_modules_others_pix.append(n.replace(".weight",""))
                l_modules_others_sem.append(n.replace(".weight",""))
                break

    # 注释掉pixel lora配置
    # lora_conf_encoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_pix)
    # lora_conf_decoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_pix)
    # lora_conf_others_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_modules_others_pix)
    lora_conf_encoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_sem)
    lora_conf_decoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_sem)
    lora_conf_others_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_modules_others_sem)

    # 注释掉pixel lora适配器
    # unet.add_adapter(lora_conf_encoder_pix, adapter_name="default_encoder_pix")
    # unet.add_adapter(lora_conf_decoder_pix, adapter_name="default_decoder_pix")
    # unet.add_adapter(lora_conf_others_pix, adapter_name="default_others_pix")
    # Avoid duplicate adapter addition
    if not hasattr(unet, "peft_config") or "default_encoder_sem" not in getattr(unet, "peft_config", {}):
        unet.add_adapter(lora_conf_encoder_sem, adapter_name="default_encoder_sem")
    if not hasattr(unet, "peft_config") or "default_decoder_sem" not in getattr(unet, "peft_config", {}):
        unet.add_adapter(lora_conf_decoder_sem, adapter_name="default_decoder_sem")
    if not hasattr(unet, "peft_config") or "default_others_sem" not in getattr(unet, "peft_config", {}):
        unet.add_adapter(lora_conf_others_sem, adapter_name="default_others_sem")

    if return_lora_module_names:
        # return unet, l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix, l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem
        return unet, l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem
    else:
        return unet


class UncertaintyEstimator(nn.Module):
    """Simple uncertainty estimator that generates a single uncertainty map"""

    def __init__(self, in_channels=3, hidden_channels=64, out_channels=3):
        super().__init__()

        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
            # No activation for final layer to allow negative values
        )

        # UDR-inspired ranking parameters
        self.topk_ratio = 0.8
        self.enable_ranking = True

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Xavier initialization for better numerical stability
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # Small positive bias

    def forward(self, lr_image):
        """
        Forward pass to generate single uncertainty map
        Args:
            lr_image: Input low-resolution image [B, 3, H, W]
        Returns:
            uncertainty_map: Single uncertainty map [B, 1, H, W]
        """
        # Simple encoder-decoder
        features = self.encoder(lr_image)
        uncertainty_map = self.decoder(features)

        return uncertainty_map
    
    def get_topk_mask(self, uncertainty, topk_ratio=None):
        """Get mask for top-k uncertain pixels using the final uncertainty map"""
        if topk_ratio is None:
            topk_ratio = self.topk_ratio
            
        # uncertainty is now directly the final uncertainty map
        b, c, h, w = uncertainty.shape
        uncertainty_flat = uncertainty.view(b, c, h * w)
        
        k = int(topk_ratio * h * w)
        topk_values, topk_indices = torch.topk(uncertainty_flat, k=k, dim=-1, largest=True)
        
        mask = torch.zeros_like(uncertainty_flat)
        mask.scatter_(-1, topk_indices, 1.0)
        mask = mask.view(b, c, h, w)
        
        return mask, topk_indices


class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        
        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)
            noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)
            

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))

        return loss

    def stopgrad(self, x):
        return x.detach()


class PiSASR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args

        self.enable_dual_attention = False
        
        # Uncertainty estimation parameters
        self.enable_uncertainty = getattr(args, 'enable_uncertainty', False)
        if self.enable_uncertainty:
            # Use simple uncertainty estimator
            self.uncertainty_estimator = UncertaintyEstimator(
                in_channels=3,
                hidden_channels=getattr(args, 'uncertainty_hidden_channels', 64),
                out_channels=3  # 3-channel output like UPSR
            ).cuda()
            
            # Update ranking parameters from args
            self.uncertainty_estimator.topk_ratio = getattr(args, 'topk_ratio', 0.8)
            self.uncertainty_estimator.enable_ranking = getattr(args, 'enable_ranking', True)
            
            # Uncertainty parameters
            self.min_noise_level = getattr(args, 'min_noise', 0.1)
            self.uncertainty_weight = getattr(args, 'un', 0.1)
            self.kappa = getattr(args, 'kappa', 2.5)
            
            # Loss weights
            self.lambda_uncertainty = getattr(args, 'lambda_uncertainty', 0.1)
            self.lambda_consistency = getattr(args, 'lambda_consistency', 0.05)

        # Quality prompt paths - use relative paths for portability
        self.quality_prompt_path = getattr(args, 'quality_prompt_path', 'preset/lowlevel_prompt_q')

        if args.resume_ckpt is None:
            self.unet, lora_unet_modules_encoder_sem, lora_unet_modules_decoder_sem, lora_unet_others_sem, =\
                    initialize_unet(
                        rank_sem=args.lora_rank_unet_sem, 
                        pretrained_model_path=args.pretrained_model_path, 
                        return_lora_module_names=True,
                        enable_dual_attention=False
                    )
            
            self.lora_rank_unet_sem = args.lora_rank_unet_sem
            self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
                lora_unet_modules_encoder_sem, lora_unet_modules_decoder_sem, lora_unet_others_sem
        else:
            print(f'====> resume from {args.resume_ckpt}')
            stage1_yaml = find_filepath(args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = read_yaml(stage1_yaml)
            stage1_args = SimpleNamespace(**stage1_args)
            
            # 强制使用单一注意力，忽略checkpoint中的设置
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
            
            self.lora_rank_unet_sem = stage1_args.lora_rank_unet_sem
            pisasr = torch.load(args.resume_ckpt, map_location='cpu')
            self.load_ckpt_from_state_dict(pisasr)
        # unet.enable_xformers_memory_efficient_attention()
        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.vae_fix.to('cuda')

        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()

    # 注释掉pixel训练函数，因为不再使用pixel lora
    # def set_train_pix(self):
    #     self.unet.train()
    #     for n, _p in self.unet.named_parameters():
    #         if "pix" in n:
    #             _p.requires_grad = True
    #         if "sem" in n:
    #             _p.requires_grad = False
    
    def set_train_sem(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sem" in n:
                _p.requires_grad = True
            # 注释掉pixel相关代码
            # if "pix" in n:
            #     _p.requires_grad = False
        
        # Set uncertainty estimator to training mode if enabled
        if self.enable_uncertainty:
            self.uncertainty_estimator.train()
            for _p in self.uncertainty_estimator.parameters():
                _p.requires_grad = True

    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        # Avoid duplicate adapter addition when loading checkpoint
        peft_cfg = getattr(self.unet, "peft_config", {})
        if "default_encoder_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        if "default_decoder_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        if "default_others_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
        sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])
        
        # Load uncertainty estimator if available
        if sd.get("enable_uncertainty", False) and self.enable_uncertainty:
            if "state_dict_uncertainty" in sd:
                self.uncertainty_estimator.load_state_dict(sd["state_dict_uncertainty"])
            if "uncertainty_params" in sd:
                params = sd["uncertainty_params"]
                self.min_noise_level = params.get("min_noise_level", self.min_noise_level)
                self.uncertainty_weight = params.get("uncertainty_weight", self.uncertainty_weight)
                self.kappa = params.get("kappa", self.kappa)
                self.lambda_uncertainty = params.get("lambda_uncertainty", self.lambda_uncertainty)
                self.lambda_consistency = params.get("lambda_consistency", self.lambda_consistency)

    # Adopted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)
    
    def get_quality_prompt(self, image_name_or_index):
        """
        Get quality prompt text for a given image.
        
        Args:
            image_name_or_index: Image filename (without extension) or index
            
        Returns:
            Quality prompt text string
        """
        try:
            # Determine base path
            base_path = self.quality_prompt_path
            
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
                print(f"Warning: Quality prompt file {prompt_path} not found, using default")
                return "This image shows average quality with some details visible but could be improved."
                
        except Exception as e:
            print(f"Error reading quality prompt: {e}")
            return "This image shows average quality with some details visible but could be improved."

    def compute_uncertainty_guided_diffusion(self, latents, uncertainty_map):
        """Apply uncertainty-guided diffusion following UPSR principles"""
        if not self.enable_uncertainty:
            return latents

        # UPSR style: Encode uncertainty map to latent space using VAE
        # uncertainty_map is [B, 3, H, W] in image space

        uncertainty_latent = self.vae_fix.encode(uncertainty_map).latent_dist.sample() * self.vae_fix.config.scaling_factor
        uncertainty_latent = uncertainty_latent.to(dtype=torch.float32)
        
        # Apply minimum noise level constraint (UPSR principle)
        final_uncertainty = self.min_noise_level + (1 - self.min_noise_level) * uncertainty_latent
        
        # UPSR principle: variance = eta * kappa^2 * un^2
        # Here we approximate eta as a constant since we're in latent space
        noise_variance = final_uncertainty.pow(2) * (self.kappa ** 2)
        
        # Generate uncertainty-modulated noise
        # Standard deviation = sqrt(variance), following Gaussian distribution principles
        noise = torch.randn_like(latents)
        noise_scale = torch.sqrt(torch.abs(uncertainty_latent) + 1e-8)  # Add epsilon for numerical stability
        guided_noise = noise * noise_scale
        
        # Apply uncertainty-guided perturbation to latents with reduced strength
        perturbation_strength = 0.5  # Reduce perturbation strength
        guided_latents = latents + guided_noise * perturbation_strength
        
        return guided_latents

    def compute_uncertainty_loss(self, sr_pred, sr_gt, uncertainty_map):
        """Compute improved uncertainty-based loss with UDR-S2Former inspired ranking"""
        if not self.enable_uncertainty:
            return torch.tensor(0.0, device=sr_pred.device)

        # Normalize uncertainty map to [0,1] range (since we removed sigmoid)
        uncertainty_normalized = torch.sigmoid(uncertainty_map)
        b, c_u, h_u, w_u = uncertainty_normalized.shape
        b, c_p, h_p, w_p = sr_pred.shape

        # Resize uncertainty to match prediction resolution if needed
        if (h_u, w_u) != (h_p, w_p):
            uncertainty_resized = F.interpolate(uncertainty_normalized, size=(h_p, w_p), mode='bilinear', align_corners=False)
        else:
            uncertainty_resized = uncertainty_normalized
        
        # UDR-S2Former style: Use 3-channel uncertainty directly for 3-channel loss computation
        # Keep 3-channel uncertainty for 3-channel loss (consistent with UDR-S2Former)
        if uncertainty_resized.shape[1] != sr_pred.shape[1]:
            if uncertainty_resized.shape[1] == 1:
                # Single channel uncertainty -> repeat to match prediction channels
                uncertainty_resized = uncertainty_resized.repeat(1, sr_pred.shape[1], 1, 1)
            else:
                # Multi-channel uncertainty -> use as is (UDR-S2Former approach)
                pass

        # UDR-S2Former exact implementation: uncertainty-weighted loss
        # s = exp(-var) for uncertainty weighting
        s = torch.exp(-uncertainty_resized)
        sr_weighted = torch.mul(sr_pred, s)  # Weighted prediction
        hr_weighted = torch.mul(sr_gt, s)    # Weighted ground truth

        # Exact UDR-S2Former loss: L1(weighted_pred, weighted_gt) + 2 * mean(uncertainty)
        uncertainty_loss = F.l1_loss(sr_weighted, hr_weighted) + 2 * torch.mean(uncertainty_resized)

        return uncertainty_loss

    def forward(self, c_t, c_tgt, batch=None, args=None):

        bs = c_t.shape[0]

        # Encode to latent space first
        encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        encoded_control = encoded_control.to(dtype=torch.float32)

        # Compute uncertainty map (UDR-S2Former style) if enabled
        uncertainty_map = None
        if self.enable_uncertainty:
            uncertainty_map = self.uncertainty_estimator(c_t)  # [B, 3, H, W] - final uncertainty map

        # Apply uncertainty-guided diffusion if enabled (runtime warmup controlled in train script)
        if self.enable_uncertainty and uncertainty_map is not None and getattr(args, "use_uncertainty_now", True):
            encoded_control = self.compute_uncertainty_guided_diffusion(encoded_control, uncertainty_map)
        
        # Calculate prompt embeddings - COMMENTED OUT SEMANTIC PROMPTS FOR QUALITY-ONLY TRAINING
        # Use a simple default embedding for semantic prompts
        # prompt_embeds = self.encode_prompt(batch["prompt"])
        # neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        # null_prompt_embeds = self.encode_prompt(batch["null_prompt"])
        default_prompt = "A high quality image with good details and clarity."
        default_neg_prompt = "low quality, blurry, noisy, distorted"
        default_null_prompt = "image"

        prompt_embeds = self.encode_prompt([default_prompt] * bs)
        neg_prompt_embeds = self.encode_prompt([default_neg_prompt] * bs)
        null_prompt_embeds = self.encode_prompt([default_null_prompt] * bs)

        if random.random() < args.null_text_ratio:
            pos_caption_enc = null_prompt_embeds
        else:
            pos_caption_enc = prompt_embeds

        # 使用质量提示进行训练（单一注意力）
        # 从batch中获取预选择的质量提示，或者根据图像名称动态读取
        if "quality_prompts" in batch:
            # 数据加载器已经提供了LR质量提示，直接使用
            quality_prompts = batch["quality_prompts"]
        else:
            # 使用默认的质量提示
            quality_prompts = []
            for i in range(bs):
                quality_prompt = self.get_quality_prompt(i)
                quality_prompts.append(quality_prompt)

        # Encode quality prompts
        quality_prompt_embeds = self.encode_prompt(quality_prompts)

        # Forward pass with quality prompts
        model_pred = self.unet(
            encoded_control,
            self.timesteps1,
            encoder_hidden_states=quality_prompt_embeds.to(torch.float32),
        ).sample
        
        x_denoised = encoded_control - model_pred
        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)
        
        # Debug: 检查训练输出是否包含NaN或无穷大
        if torch.isnan(output_image).any() or torch.isinf(output_image).any():
            print(f"Warning: Training output contains NaN or Inf values!")
            print(f"Output range: [{output_image.min().item():.4f}, {output_image.max().item():.4f}]")
            # 替换NaN和Inf值
            output_image = torch.nan_to_num(output_image, nan=0.0, posinf=1.0, neginf=-1.0)
            output_image = output_image.clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds, uncertainty_map


    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"] =\
            self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem
        sd["lora_rank_unet_sem"] = self.lora_rank_unet_sem
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        
        # Save uncertainty estimator if enabled
        sd["enable_uncertainty"] = self.enable_uncertainty
        if self.enable_uncertainty:
            sd["state_dict_uncertainty"] = self.uncertainty_estimator.state_dict()
            sd["uncertainty_params"] = {
                "min_noise_level": self.min_noise_level,
                "uncertainty_weight": self.uncertainty_weight,
                "kappa": self.kappa,
                "lambda_uncertainty": self.lambda_uncertainty,
                "lambda_consistency": self.lambda_consistency
            }
        
        torch.save(sd, outf)


class PiSASR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda"
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args = args
        
        # 使用单一注意力
        self.enable_dual_attention = False
        print("PiSASR_eval dual attention: Disabled (using single attention)")
        
        # Uncertainty estimation support
        self.enable_uncertainty = getattr(args, 'enable_uncertainty', False)
        if self.enable_uncertainty:
            # Use simple uncertainty estimator
            self.uncertainty_estimator = UncertaintyEstimator(
                in_channels=3,
                hidden_channels=getattr(args, 'uncertainty_hidden_channels', 64),
                out_channels=3  # 3-channel output like UPSR
            ).to(self.device)

            # Update ranking parameters from args
            self.uncertainty_estimator.topk_ratio = getattr(args, 'topk_ratio', 0.8)
            self.uncertainty_estimator.enable_ranking = getattr(args, 'enable_ranking', True)

        # Quality prompt paths - use relative paths for portability
        # For inference/testing, use test_lowlevel_prompt_q directory which contains generated test prompts
        self.quality_prompt_path = getattr(args, 'quality_prompt_path', 'preset/test_lowlevel_prompt_q_RealSR')
        # self.quality_prompt_path = getattr(args, 'quality_prompt_path', '')

        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        
        # 强制使用单一注意力
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        # Load pretrained weights
        self._load_pretrained_weights(args.pretrained_path)

        # Initialize VAE tiling
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size
        )

        # 简化适配器设置：只使用语义LoRA
        set_weights_and_activate_adapters(self.unet, ["default_encoder_sem", "default_decoder_sem", "default_others_sem"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()

        # Move models to device and precision
        self._move_models_to_device_and_dtype()

        # Set parameters
        self.timesteps1 = torch.tensor([1], device=self.device).long()

    def _get_dtype(self, precision):
        """Get the appropriate data type based on precision."""
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def _move_models_to_device_and_dtype(self):
        """Move models to the correct device and precision."""
        for model in [self.vae, self.unet, self.text_encoder]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights and initialize LoRA adapters."""
        sd = torch.load(pretrained_path, map_location='cpu')
        self._load_and_save_ckpt_from_state_dict(sd)


    def _load_and_save_ckpt_from_state_dict(self, sd):
        """Load checkpoint and initialize LoRA adapters."""
        # Define LoRA configurations
        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        # Add and load adapters
        # Avoid duplicate adapter addition when loading and merging
        peft_cfg = getattr(self.unet, "peft_config", {})
        if "default_encoder_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        if "default_decoder_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        if "default_others_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        # 简化权重加载：直接合并LoRA权重
        set_weights_and_activate_adapters(self.unet, ["default_encoder_sem", "default_decoder_sem", "default_others_sem"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()
        
        # Load uncertainty estimator if available
        if sd.get("enable_uncertainty", False) and self.enable_uncertainty:
            if "state_dict_uncertainty" in sd:
                self.uncertainty_estimator.load_state_dict(sd["state_dict_uncertainty"])
                self.uncertainty_estimator.eval()
                self.uncertainty_estimator.requires_grad_(False)


    def set_eval(self):
        """Set models to evaluation mode."""
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        if self.enable_uncertainty:
            self.uncertainty_estimator.eval()
            self.uncertainty_estimator.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)
    
    def get_quality_prompt(self, image_name_or_index):
        """
        Get quality prompt text for a given image.
        
        Args:
            image_name_or_index: Image filename (without extension) or index
            
        Returns:
            Quality prompt text string
        """
        try:
            # Determine base path
            base_path = self.quality_prompt_path
            
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
                print(f"Warning: Quality prompt file {prompt_path} not found, using default")
                return "This image shows average quality with some details visible but could be improved."
                
        except Exception as e:
            print(f"Error reading quality prompt: {e}")
            return "This image shows average quality with some details visible but could be improved."

    def _apply_uncertainty_guidance_inference(self, encoded_control, uncertainty_map):
        """Apply uncertainty guidance during inference (consistent with training and UPSR)"""
        # UPSR style: Encode uncertainty map to latent space using VAE
        uncertainty_latent = self.vae.encode(uncertainty_map).latent_dist.sample() * self.vae.config.scaling_factor
        uncertainty_latent = uncertainty_latent.to(dtype=self.weight_dtype)
        
        # Apply minimum noise level constraint (using same parameters as training)
        min_noise_level = getattr(self.args, 'min_noise', 0.1)
        final_uncertainty = min_noise_level + (1 - min_noise_level) * uncertainty_latent
        
        # variance = eta * kappa^2 * un^2 (adapted for single-step)
        kappa = getattr(self.args, 'kappa', 1.0)
        noise_variance = final_uncertainty.pow(2) * (kappa ** 2)
        
        # Generate uncertainty-modulated noise
        noise = torch.randn_like(encoded_control)
        noise_scale = torch.sqrt(torch.abs(uncertainty_latent) + 1e-8)
        guided_noise = noise * noise_scale
        
        # Apply uncertainty-guided perturbation to latents with reduced strength
        perturbation_strength = 0.5  # Reduce perturbation strength
        guided_latents = encoded_control + guided_noise * perturbation_strength
        
        return guided_latents

    def count_parameters(self, model):
        """Count the number of parameters in a model."""
        return sum(p.numel() for p in model.parameters()) / 1e9

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None, image_name=None):
        """Forward pass for inference."""
        torch.cuda.synchronize()
        start_time = time.time()

        c_t = c_t.to(dtype=self.weight_dtype)

        # Try to get quality prompt from generated files if image_name is provided
        if image_name is not None:
            try:
                # Extract image name without extension and path
                base_name = os.path.basename(image_name)
                image_name_no_ext = os.path.splitext(base_name)[0]

                # Get quality prompt from test_lowlevel_prompt_q directory
                quality_prompt = self.get_quality_prompt(image_name_no_ext)

                # Use the quality prompt as semantic prompt
                semantic_prompt = quality_prompt
                print(f"Using quality prompt for {image_name_no_ext}: {quality_prompt[:100]}...")
            except Exception as e:
                print(f"Warning: Could not load quality prompt for {image_name}, using default: {e}")
                semantic_prompt = "A high quality image with good details and clarity."
        else:
            # Use default semantic prompt for quality-only training
            semantic_prompt = "A high quality image with good details and clarity."

        prompt_embeds = self.encode_prompt([semantic_prompt]).to(dtype=self.weight_dtype)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        encoded_control = encoded_control.to(dtype=self.weight_dtype)

        # Apply uncertainty-guided diffusion in inference (UDR-S2Former style)
        if self.enable_uncertainty:
            uncertainty_map = self.uncertainty_estimator(c_t)  # [B, 3, H, W] - final uncertainty map

            # Apply the same uncertainty guidance as in training
            encoded_control = self._apply_uncertainty_guidance_inference(encoded_control, uncertainty_map)

        # Tile and process latents if necessary
        model_pred = self._process_latents(encoded_control, prompt_embeds, default)

        # Decode output
        x_denoised = encoded_control - model_pred
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        # Debug: 检查输出是否包含NaN或无穷大
        if torch.isnan(output_image).any() or torch.isinf(output_image).any():
            print(f"Warning: Output contains NaN or Inf values!")
            print(f"Output range: [{output_image.min().item():.4f}, {output_image.max().item():.4f}]")
            # 替换NaN和Inf值
            output_image = torch.nan_to_num(output_image, nan=0.0, posinf=1.0, neginf=-1.0)
            output_image = output_image.clamp(-1, 1)

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        return total_time, output_image

    def _process_latents(self, encoded_control, prompt_embeds, default):
        """Process latents with or without tiling."""
        h, w = encoded_control.size()[-2:]
        tile_size, tile_overlap = self.args.latent_tiled_size, self.args.latent_tiled_overlap

        if h * w <= tile_size * tile_size:
            print("[Tiled Latent]: Input size is small, no tiling required.")
            return self._predict_no_tiling(encoded_control, prompt_embeds, default)

        print(f"[Tiled Latent]: Input size {h}x{w}, tiling required.")
        return self._predict_with_tiling(encoded_control, prompt_embeds, default, tile_size, tile_overlap)

    def _predict_no_tiling(self, encoded_control, prompt_embeds, default):
        """Predict on the entire latent without tiling."""
        # Get quality prompt if dual attention is enabled
        quality_prompt_embeds = None
        if self.enable_dual_attention:
            # 始终使用LR质量提示，与训练保持一致
            quality_prompt = self.get_quality_prompt(0)
            print(f"Debug - Quality prompt: {quality_prompt[:100]}...")  # 调试信息
            quality_prompt_embeds = self.encode_prompt([quality_prompt]).to(dtype=self.weight_dtype)

        if default:
            if self.enable_dual_attention:
                return self.unet(
                    sample=encoded_control, 
                    timestep=self.timesteps1, 
                    encoder_hidden_states=prompt_embeds,
                    quality_prompt_embeds=quality_prompt_embeds
                ).sample
            else:
                return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

        # 使用质量提示进行推理（单一注意力）
        return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

    def _predict_with_tiling(self, encoded_control, prompt_embeds, default, tile_size, tile_overlap):
        """Predict on the latent with tiling."""
        _, _, h, w = encoded_control.size()
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        tile_size = min(tile_size, min(h, w))
        grid_rows = 0
        cur_x = 0
        while cur_x < encoded_control.size(-1):
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < encoded_control.size(-2):
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            noise_preds_row = []
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                # input tile dimensions
                input_tile = encoded_control[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols-1:
                    input_list_t = torch.cat(input_list, dim=0)
                    # predict the noise residual
                    # 使用质量提示
                    model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
                    # model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample
                    input_list = []
                noise_preds.append(model_out)

        # Stitch noise predictions for all tiles
        noise_pred = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        # Average overlapping areas with more than 1 contributor
        noise_pred /= contributors
        model_pred = noise_pred
        return model_pred



    

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generate a Gaussian mask for tile contributions."""
        from numpy import pi, exp, sqrt
        import numpy as np

        midpoint_x = (tile_width - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [exp(-(x - midpoint_x) ** 2 / (2 * (tile_width ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for x in range(tile_width)]
        y_probs = [exp(-(y - midpoint_y) ** 2 / (2 * (tile_height ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for y in range(tile_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tensor(weights, device=self.device).repeat(nbatches, self.unet.config.in_channels, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        """Initialize VAE with tiled encoding/decoding."""
        encoder, decoder = self.vae.encoder, self.vae.decoder

        if not hasattr(encoder, 'original_forward'):
            encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'):
            decoder.original_forward = decoder.forward

        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)