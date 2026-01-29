# 超分

## 通用命令

### 生成`gt_path.txt`

```bash
cd /home/bd/QUSR/preset/your-dataset-256
find $(pwd)/HR -name "*.jpg" -o -name "*.png" -o -name "*.tif" > gt_path.txt
```

## 数据集 256 训练 x4 超分模型

### 准备数据集

```
preset/your-dataset-256/
├── HR/                    # 256×256 的高分辨率图像
├── LR/                    # 64×64 的低分辨率图像
├── SR_Bicubic/            # 64×64 bicubic上采样到 256×256 的图像（训练输入）
├── gt_path.txt            # HR 图像路径列表
├── lowlevel_prompt_q/     # LR 图像质量提示 txt 目录
├── lowlevel_prompt_q_GT/  # HR 图像质量提示 txt 目录
└── testfolder/
    ├── test_LR/           # 测试集 64×64 图像
    └── test_HR/           # 测试集 256×256 图像
```

## 修改 `train_pisasr.sh`

1. 调整 `DATASET_BASE_DIR` 变量
2. `--resolution_ori=256` 和 `--resolution_tgt=256` 调整分辨率
