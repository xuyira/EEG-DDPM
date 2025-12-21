# EEG DDPM 项目

使用 DDPM (Denoising Diffusion Probabilistic Model) 生成 EEG 数据的 CWT (Continuous Wavelet Transform) 表示。

## 项目结构

```
EEG_DDPM/
├── main.py              # 主入口文件
├── config.py            # 配置文件
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── model/               # 模型相关
│   ├── __init__.py
│   └── unet.py         # UNet 模型定义
├── utils/               # 工具函数
│   └── data_preprocessing.py  # 数据预处理
└── README.md           # 本文件
```

## 安装依赖

```bash
pip install torch torchvision diffusers accelerate transformers matplotlib scipy pywavelets
```

## 使用方法

### 1. 配置参数

编辑 `config.py` 文件，修改训练参数：

- `image_size`: 图像尺寸
- `train_batch_size`: 训练批次大小
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `data_root`: 数据根目录
- `leave_sub`: 留一受试者编号
- `subjects`: 所有受试者编号列表

### 2. 运行训练

```bash
python main.py
```

## 主要功能

1. **数据预处理** (`utils/data_preprocessing.py`)
   - `bci2a_preprcessing_method`: 加载数据并进行 CWT 变换
   - `build_loso_train_npy`: 构建留一受试者训练数据

2. **模型定义** (`model/unet.py`)
   - `create_unet_model`: 创建 UNet2DModel

3. **训练** (`train.py`)
   - `train_loop`: 训练循环

4. **评估** (`evaluate.py`)
   - `evaluate`: 生成样本图像并保存

## 输出

- 模型检查点保存在 `./model_checkpoints/ckpt-DDPM/suj{leave_sub}/`
- 生成的图像保存在 `./model_pic/DDPM/suj{leave_sub}/`
- TensorBoard 日志保存在 `./model_checkpoints/ckpt-DDPM/suj{leave_sub}/logs/`

