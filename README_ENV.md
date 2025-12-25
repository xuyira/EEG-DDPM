# EEG-DDPM 环境配置指南

## 快速开始

### 方法 1：使用 requirements.txt（推荐）

```bash
# 1. 创建虚拟环境（推荐）
conda create -n timedp python=3.8
conda activate timedp

# 或者使用 venv
python -m venv timedp
source timedp/bin/activate  # Linux/Mac
# 或
timedp\Scripts\activate  # Windows

# 2. 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.7
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu118

# CPU 版本
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

# 3. 安装其他依赖
pip install -r requirements.txt
```

### 方法 2：使用自动配置脚本

```bash
# 给脚本添加执行权限
chmod +x setup_env.sh

# 运行脚本
./setup_env.sh
```

## 版本兼容性说明

### 关键版本约束

- **Python**: 3.8（推荐）
- **PyTorch**: 1.13.1（与 Python 3.8 兼容）
- **Diffusers**: 0.21.4（避免 `huggingface_hub` 兼容性问题）
- **HuggingFace Hub**: <0.20.0（与 diffusers 0.21.4 兼容）

### 已知问题修复

1. **`AttributeError: module 'torch' has no attribute 'xpu'`**
   - 原因：diffusers 新版本需要支持 Intel XPU 的 PyTorch
   - 解决：使用 diffusers==0.21.4

2. **`ImportError: cannot import name 'cached_download' from 'huggingface_hub'`**
   - 原因：huggingface_hub 新版本移除了 `cached_download`
   - 解决：使用 huggingface_hub<0.20.0

## 依赖包说明

### 核心依赖

- **torch, torchvision**: 深度学习框架
- **diffusers**: DDPM 模型实现
- **accelerate**: 训练加速
- **transformers**: HuggingFace 模型库

### 数据处理

- **numpy, scipy**: 数值计算
- **pandas**: 数据处理和 Excel 导出
- **scikit-learn**: 机器学习工具

### 可视化

- **matplotlib**: 绘图
- **Pillow**: 图像处理

### 其他工具

- **einops**: Tensor 操作
- **tqdm**: 进度条
- **openpyxl**: Excel 文件处理
- **torchsummary**: 模型摘要

## 验证安装

运行以下命令验证环境是否正确配置：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "from diffusers import DDPMScheduler; print('Diffusers 导入成功')"
```

## 常见问题

### Q: 安装 diffusers 时出现版本冲突
A: 先卸载旧版本，再安装指定版本：
```bash
pip uninstall diffusers huggingface_hub -y
pip install diffusers==0.21.4 huggingface_hub==0.16.4
```

### Q: CUDA 版本不匹配
A: 访问 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/) 查看对应 CUDA 版本的安装命令。

### Q: 内存不足
A: 可以减小 `train_batch_size` 或使用梯度累积（`gradient_accumulation_steps`）。

## 更新日志

- 2024: 添加 diffusers 0.21.4 兼容性修复
- 2024: 添加 Python 3.8 支持说明

