import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


# LabelEmbedder：将类别标签转换为 embedding
class LabelEmbedder(nn.Module):
    """将类别标签转换为 embedding，用于交叉注意力"""
    def __init__(self, num_classes, emb_dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, emb_dim)
    
    def forward(self, labels):
        """
        参数:
            labels: (B,) 类别标签
        
        返回:
            emb: (B, emb_dim) 类别嵌入
        """
        return self.embed(labels)  # (B, emb_dim)


# 创建使用交叉注意力的条件 UNet 模型
def create_unet2dcond_model(config, num_classes=4):
    """
    创建使用交叉注意力的条件 UNet 模型
    
    参数:
        config: TrainingConfig 配置对象
        num_classes: 标签类别数量（默认4，对应BCI2a的4个类别）
    
    返回:
        unet: UNet2DConditionModel 模型（使用交叉注意力）
        label_embedder: LabelEmbedder 标签编码器
    """
    # 使用交叉注意力的 block 类型
    num_blocks = len(config.unet_block_out_channels)
    
    unet_kwargs = {
        "sample_size": config.image_size,           # 图像尺寸
        "in_channels": config.unet_in_channels,     # 输入通道数
        "out_channels": config.unet_out_channels,   # 输出通道数
        "block_out_channels": config.unet_block_out_channels,   # 每个 block 的输出通道数
        "layers_per_block": config.unet_layers_per_block,  # 每个 block 的层数
        "cross_attention_dim": config.cross_attention_dim,  # 交叉注意力维度
    }
    
    unet = UNet2DConditionModel(**unet_kwargs)
    
    # 创建标签编码器
    label_embedder = LabelEmbedder(num_classes=num_classes, emb_dim=config.encoder_hid_dim)
    
    return unet, label_embedder

