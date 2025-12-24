import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ClassConditionedUnet(nn.Module):
    """
    条件 UNet 模型，使用类别标签作为条件
    用于 EEG 数据的条件生成（4个类别：0-3）
    """
    def __init__(self, config, num_classes=4, class_emb_size=4):
        super().__init__()
        
        self.num_classes = num_classes
        self.class_emb_size = class_emb_size
        
        # 类别嵌入层：将类别标签映射到嵌入向量
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # UNet 模型，输入通道数增加 class_emb_size 以接受条件信息
        self.model = UNet2DModel(
            sample_size=config.image_size,  # 目标图像分辨率
            in_channels=config.unet_in_channels + class_emb_size,  # 额外输入通道用于类别条件
            out_channels=config.unet_out_channels,  # 输出通道数
            layers_per_block=config.unet_layers_per_block,  # 每个 UNet block 中的 ResNet 层数
            block_out_channels=config.unet_block_out_channels,  # 每个 UNet block 的输出通道数
            down_block_types=(
                "DownBlock2D",  # 常规 ResNet 下采样块
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # 带空间自注意力的 ResNet 下采样块
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # 常规 ResNet 上采样块
                "AttnUpBlock2D",  # 带空间自注意力的 ResNet 上采样块
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, timesteps, class_labels, return_dict=True):
        """
        前向传播
        
        参数:
            x: 输入图像 (B, C, H, W)
            timesteps: 时间步 (B,)
            class_labels: 类别标签 (B,)
            return_dict: 是否返回字典格式
        
        返回:
            噪声预测 (B, C, H, W)
        """
        bs, ch, h, w = x.shape

        # 类别条件：将类别标签映射到嵌入维度
        class_cond = self.class_emb(class_labels)  # (B, class_emb_size)
        # 扩展为空间维度 (B, class_emb_size, H, W)
        class_cond = class_cond.view(bs, self.class_emb_size, 1, 1).expand(bs, self.class_emb_size, h, w)

        # 将输入图像和类别条件在通道维度拼接
        net_input = torch.cat((x, class_cond), dim=1)  # (B, C + class_emb_size, H, W)

        # 输入到 UNet 并返回预测
        output = self.model(net_input, timesteps, return_dict=return_dict)
        
        if return_dict:
            return output
        else:
            return output.sample
