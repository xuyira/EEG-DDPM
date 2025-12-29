from dataclasses import dataclass
from pathlib import Path
import os
import torch


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 图像尺寸
    image_size: tuple = (64, 64)
    # 训练批次大小
    train_batch_size: int = 2
    # 评估批次大小
    eval_batch_size: int = 2
    # 训练轮数
    num_epochs: int = 500
    # 梯度累积步数（累计几次梯度更新一次参数）
    gradient_accumulation_steps: int = 1
    # 学习率
    learning_rate: float = 1e-4
    # 学习率衰减
    lr_warmup_steps: int = 500
    
    save_image_epochs: int = 50
    save_model_epochs: int = 20
    
    mixed_precision: str = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "./model_checkpoints/DDPM_attencond/" #模型保存目录
    # 是否上传模型到HF Hub
    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_model_id: str = "hibiscus/test_model"  # the name of the repository to create on the HF Hub
    hub_private_repo: bool = None
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 42
    
    # UNet 模型架构参数
    unet_in_channels: int = 22  # 输入通道数（对应 EEG 通道数）
    unet_out_channels: int = 22  # 输出通道数
    unet_layers_per_block: int = 2  # 每个 UNet block 中的 ResNet 层数
    unet_block_out_channels: tuple = (64, 128, 256, 256)  # 每个 UNet block 的输出通道数
    cross_attention_dim: int = 256  # 交叉注意力维度（K, V 的维度）
    encoder_hid_dim: int = 256  # 条件的维度

    
    # 数据相关
    data_root: Path = Path("./standard_2a_data")
    leave_sub: int = 1
    subjects: list = None

    # DelayEmbedder 参数（用于将 EEG 时序信号转为图片）
    delay: int = 15
    embedding: int = 64
    
    # 输出目录
    pic_dir: str = "./model_pic/DDPM_attencond/"
    
    def __post_init__(self):
        if self.subjects is None:
            self.subjects = [1, 3]  #2, 3, 4, 5, 6, 7, 8, 9
        # 设置输出目录
        self.output_dir = str(Path(self.output_dir) / f"sub{self.leave_sub}")
        self.pic_dir = str(Path(self.pic_dir) / f"sub{self.leave_sub}")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.pic_dir, exist_ok=True)

