#!/bin/bash
# EEG-DDPM 环境配置脚本
# 适用于 Python 3.8 和 CUDA 环境

echo "=========================================="
echo "EEG-DDPM 环境配置脚本"
echo "=========================================="

# 检查 Python 版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "当前 Python 版本: $python_version"

# 检查是否有 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "未检测到 NVIDIA GPU，将安装 CPU 版本的 PyTorch"
fi

echo ""
echo "开始安装依赖包..."
echo "=========================================="

# 升级 pip
pip install --upgrade pip

# 安装 PyTorch（根据是否有 CUDA 选择版本）
# 注意：请根据你的 CUDA 版本调整 PyTorch 安装命令
# 访问 https://pytorch.org/get-started/previous-versions/ 查看对应版本

# 如果有 CUDA 11.8
if command -v nvidia-smi &> /dev/null; then
    echo "安装 CUDA 版本的 PyTorch..."
    pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
else
    echo "安装 CPU 版本的 PyTorch..."
    pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu
fi

# 安装其他依赖
echo "安装其他依赖包..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "验证安装："
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers 版本: {diffusers.__version__}')"
python -c "import numpy; print(f'NumPy 版本: {numpy.__version__}')"

