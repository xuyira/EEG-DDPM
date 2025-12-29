import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


# 创建使用类别嵌入的条件 UNet 模型（只使用类别+时间嵌入，不使用交叉注意力）
def create_unet2dcond_model(config, num_classes=4):
    """
    创建使用类别嵌入的条件 UNet 模型（只使用类别+时间嵌入）
    
    参数:
        config: TrainingConfig 配置对象
        num_classes: 标签类别数量（默认4，对应BCI2a的4个类别）
    
    返回:
        unet: UNet2DConditionModel 模型（只使用 class_labels，不使用交叉注意力）
    """
    # 使用不使用交叉注意力的 block 类型
    num_blocks = len(config.unet_block_out_channels)
    down_block_types = ["DownBlock2D"] * num_blocks
    up_block_types = ["UpBlock2D"] * num_blocks
    
    unet_kwargs = {
        "sample_size": config.image_size,           # 图像尺寸
        "in_channels": config.unet_in_channels,     # 输入通道数
        "out_channels": config.unet_out_channels,   # 输出通道数
        "block_out_channels": config.unet_block_out_channels,   # 每个 block 的输出通道数
        "layers_per_block": config.unet_layers_per_block,  # 每个 block 的层数
        "down_block_types": down_block_types,  # 下采样块类型（不使用交叉注意力）
        "up_block_types": up_block_types,  # 上采样块类型（不使用交叉注意力）
        "mid_block_type": None,  # 跳过中间块（不使用交叉注意力）
        "class_embed_type": config.class_embed_type,  # 类别嵌入类型
        "num_class_embeds": config.num_class_embeds,  # 类别数量
        "class_embeddings_concat": config.class_embeddings_concat,  # 是否拼接
    }
    
    # 如果使用 projection 类型，需要添加 projection_class_embeddings_input_dim
    if config.class_embed_type == "projection":
        if config.projection_class_embeddings_input_dim is None:
            raise ValueError("当 class_embed_type='projection' 时，必须设置 projection_class_embeddings_input_dim")
        unet_kwargs["projection_class_embeddings_input_dim"] = config.projection_class_embeddings_input_dim
    
    unet = UNet2DConditionModel(**unet_kwargs)
    
    return unet

