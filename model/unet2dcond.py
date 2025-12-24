import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


# 1）自定义条件编码器（比如标签向量）
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, emb_dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, emb_dim)

    def forward(self, labels):
        return self.embed(labels)  # (B, emb_dim)


# 2）创建条件 UNet 模型和标签编码器
def create_unet2dcond_model(config, num_classes=4, cond_emb_dim=64):
    """
    创建条件 UNet 模型和标签编码器
    
    参数:
        config: TrainingConfig 配置对象
        num_classes: 标签类别数量（默认4，对应BCI2a的4个类别）
        cond_emb_dim: 条件嵌入维度（默认64）
    
    返回:
        unet: UNet2DConditionModel 模型
        label_embedder: LabelEmbedder 标签编码器
    """
    # 构建 conditional UNet
    # 注意：UNet2DConditionModel 要求 `block_out_channels` 的长度
    # 必须和 `down_block_types` / `up_block_types` 一致
    block_out_channels = config.unet_block_out_channels
    # 这里我们使用 3 个带交叉注意力的 down / up block
    down_block_types = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    )
    up_block_types = (
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    )

    unet = UNet2DConditionModel(
        sample_size=config.image_size,           # 图像尺寸
        in_channels=config.unet_in_channels,     # 输入通道数
        out_channels=config.unet_out_channels,   # 输出通道数
        block_out_channels=block_out_channels,   # 每个 block 的输出通道数
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        encoder_hid_dim=cond_emb_dim,            # 交叉注意力的条件维度
        layers_per_block=config.unet_layers_per_block,  # 每个 block 的层数
    )
    
    # 创建标签编码器
    label_embedder = LabelEmbedder(num_classes=num_classes, emb_dim=cond_emb_dim)
    
    return unet, label_embedder

