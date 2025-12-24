"""
主入口文件
用于训练 DDPM 模型生成 EEG 数据的 CWT 表示
"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import torch.nn.functional as F
import gc
from diffusers.optimization import get_cosine_schedule_with_warmup

from config import TrainingConfig
from model import ClassConditionedUnet
from utils.train import train_loop_conditional
from utils.data_preprocessing import bci2a_preprcessing_method
from torch.utils.data import TensorDataset

def main():
    """主函数"""
    # 创建配置（目录会在 __post_init__ 中自动创建）
    config = TrainingConfig()
    
    print("图片输出文件夹：", config.pic_dir)
    print("模型输出文件夹：", config.output_dir)
    
    # 选择运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 数据预处理（使用 DelayEmbedder 将 EEG 时序信号转成图片）
    print("\n开始数据预处理（DelayEmbedder）...")
    X_train_img_np, X_train_label, X_train_sub = bci2a_preprcessing_method(
        root=config.data_root,
        leave_sub=config.leave_sub,
        subjects=config.subjects,
        delay=config.delay,
        embedding=config.embedding,
        device=device,
    )

    # 转换为 tensor，(B, C, H, W)
    print("转换为 tensor...")
    X_train_img = torch.from_numpy(X_train_img_np).float()
    # 删除 numpy 数组以释放内存
    del X_train_img_np
    gc.collect()  # 强制垃圾回收
    print("数据维度：", X_train_img.shape)
    
    # 可视化一个样本
    show_trial = 0
    show_channel = 0
    plt.figure()
    plt.imshow(X_train_img[show_trial, show_channel, :, :].numpy(), aspect='auto')
    plt.colorbar()
    plt.title("Normalized EEG Signal after transform trial{} channel{}".format(show_trial, show_channel))
    plt.savefig(f"{config.pic_dir}/sample_transpic_trial{show_trial}_channel{show_channel}.png", dpi=300)
    plt.close()
    
    # 创建条件模型
    print("\n创建条件模型...")
    model = ClassConditionedUnet(config, num_classes=4, class_emb_size=4)
    model = model.to(device)
    
    # 测试模型输入输出
    sample_image = X_train_img[0].unsqueeze(0).to(device)
    sample_label = torch.tensor([X_train_label[0]], device=device)
    print("输入图像维度：", sample_image.shape)
    print("输入标签：", sample_label.item())
    with torch.no_grad():
        output = model(sample_image, torch.tensor([0], device=device), sample_label, return_dict=False)
    print("输出维度：", output.shape)
    
    # 创建数据加载器（包含图像和标签）
    # 将标签转换为 tensor
    X_train_label_tensor = torch.from_numpy(X_train_label).long()
    
    # 创建 TensorDataset，包含图像和标签
    train_dataset = TensorDataset(X_train_img, X_train_label_tensor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size, 
        shuffle=True
    )
    
    # 创建噪声调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    
    # 开始训练（条件生成）
    print("\n开始条件训练...")
    train_loop_conditional(
        config=config,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
        pic_dir=config.pic_dir
    )
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()

