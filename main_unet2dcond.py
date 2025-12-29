"""
主入口文件
用于训练条件 DDPM 模型生成 EEG 数据的 CWT 表示
使用标签作为条件进行条件生成
"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
import torch.nn.functional as F
import gc
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np

from config import TrainingConfig
from model.unet2dcond import create_unet2dcond_model
from utils.data_preprocessing import bci2a_preprcessing_method, DelayEmbedder


class EEGDataset(Dataset):
    """自定义数据集类，同时返回图像和标签"""
    def __init__(self, images, labels):
        """
        参数:
            images: torch.Tensor, 形状为 (N, C, H, W)
            labels: numpy.ndarray 或 torch.Tensor, 形状为 (N,)
        """
        self.images = images
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

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
    print("标签维度：", X_train_label.shape)
    print("标签范围：", f"{X_train_label.min()} - {X_train_label.max()}")
    print("标签唯一值：", np.unique(X_train_label))
    
    # 可视化一个样本
    show_trial = 0
    show_channel = 0
    plt.figure()
    plt.imshow(X_train_img[show_trial, show_channel, :, :].numpy(), aspect='auto')
    plt.colorbar()
    plt.title("Normalized EEG Signal after transform trial{} channel{} label{}".format(
        show_trial, show_channel, X_train_label[show_trial]))
    plt.savefig(f"{config.pic_dir}/sample_transpic_trial{show_trial}_channel{show_channel}.png", dpi=300)
    plt.close()
    
    # 创建条件模型
    print("\n创建条件模型...")
    num_classes = len(np.unique(X_train_label))  # 自动检测类别数量
    # 更新 config 中的类别数量
    config.num_class_embeds = num_classes
    print(f"类别数量: {num_classes}, 类别嵌入类型: {config.class_embed_type}, 拼接方式: {config.class_embeddings_concat}")
    print(f"使用交叉注意力: {config.use_cross_attention}")
    
    unet = create_unet2dcond_model(
        config, 
        num_classes=num_classes
    )
    
    # 测试模型输入输出（在 CPU 上测试，避免设备冲突）
    sample_image = X_train_img[0].unsqueeze(0)  # 保持在 CPU
    sample_label = torch.tensor([X_train_label[0]], dtype=torch.long)
    sample_timestep = torch.tensor([0], dtype=torch.long)
    
    # 测试 UNet 输出
    with torch.no_grad():
        if config.use_cross_attention:
            # 如果使用交叉注意力，需要提供 encoder_hidden_states
            # 这里先用 None 测试，实际使用时需要提供真实的 encoder_hidden_states
            output = unet(sample_image, sample_timestep, class_labels=sample_label, encoder_hidden_states=None).sample
        else:
            # 只使用类别嵌入，不需要 encoder_hidden_states
            output = unet(sample_image, sample_timestep, class_labels=sample_label, encoder_hidden_states=None).sample
        print("输入图像维度：", sample_image.shape)
        print("输入标签：", sample_label)
        print("输出维度：", output.shape)
        
    # 创建数据集和数据加载器
    train_dataset = EEGDataset(X_train_img, X_train_label)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True
    )
    
    # 创建噪声调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 创建优化器和学习率调度器（只需要优化 UNet，类别嵌入在 UNet 内部）
    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=config.learning_rate
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    
    # 开始条件训练
    print("\n开始条件训练...")
    train_loop_conditional(
        config=config,
        unet=unet,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
        pic_dir=config.pic_dir,
        device=device,
        num_classes=num_classes
    )
    
    print("\n训练完成！")


def train_loop_conditional(
    config, unet, noise_scheduler, optimizer, 
    train_dataloader, lr_scheduler, pic_dir, device, num_classes
):
    """条件训练循环（使用类别嵌入方式）"""
    from accelerate import Accelerator
    from tqdm.auto import tqdm
    import os
    from pathlib import Path
    
    # 初始化 accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_conditional")
    
    # 准备模型和数据加载器（不需要包装类，直接使用 UNet）
    model = unet
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0
    
    # 训练循环
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, (clean_images, labels) in enumerate(train_dataloader):
            clean_images = clean_images.to(accelerator.device)
            labels = labels.to(accelerator.device)
            
            # 采样噪声
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            
            # 采样随机时间步
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), 
                device=clean_images.device, dtype=torch.int64
            )
            
            # 添加噪声到干净图像
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # 预测噪声残差（使用类别标签，encoder_hidden_states 设为 None）
                noise_pred = model(noisy_images, timesteps, class_labels=labels, encoder_hidden_states=None).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0], 
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        # 每个 epoch 后可选地生成一些样本图像
        if accelerator.is_main_process and ((epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1):
            # 解包模型
            unwrapped_unet = accelerator.unwrap_model(model)
            
            # 生成样本（为每个类别生成一个样本）
            generate_conditional_samples(
                unwrapped_unet, noise_scheduler, 
                config, epoch, pic_dir, num_classes, device
            )
        
        # 保存模型
        if accelerator.is_main_process and ((epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1):
            unwrapped_unet = accelerator.unwrap_model(model)
            
            # 保存模型
            save_dir = Path(config.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存 UNet（类别嵌入已包含在 UNet 内部）
            unwrapped_unet.save_pretrained(save_dir / "unet")
            print(f"模型已保存到 {save_dir}")


def generate_conditional_samples(unet, noise_scheduler, config, epoch, pic_dir, num_classes, device):
    """生成条件样本，并输出两种图：
    1）图片域（按类别排列的生成图像）
    2）时间域 EEG（通过 DelayEmbedder.img_to_ts 还原）
    """
    unet.eval()
    
    # 为每个类别生成样本
    images_per_class = config.eval_batch_size // num_classes
    if images_per_class == 0:
        images_per_class = 1
    
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for class_id in range(num_classes):
            # 创建标签
            labels = torch.full((images_per_class,), class_id, dtype=torch.long, device=device)
            
            # 生成随机噪声
            shape = (images_per_class, config.unet_in_channels, *config.image_size)
            noisy_images = torch.randn(shape, device=device)
            
            # 反向扩散过程（使用 class_labels，encoder_hidden_states 设为 None）
            for t in noise_scheduler.timesteps:
                timesteps = torch.full((images_per_class,), t, dtype=torch.long, device=device)
                noise_pred = unet(noisy_images, timesteps, class_labels=labels, encoder_hidden_states=None).sample
                noisy_images = noise_scheduler.step(noise_pred, t, noisy_images).prev_sample
            
            all_images.append(noisy_images.cpu().numpy())
            all_labels.extend([class_id] * images_per_class)
    
    # 合并所有图像
    images = np.concatenate(all_images, axis=0)  # (B, C, H, W)
    print(f"生成图像维度：{images.shape}")
    
    # ================== 图 1：图片域可视化（与原 evaluate 类似） ==================
    sample_channel = 0
    fig, ax = plt.subplots(num_classes, images_per_class, figsize=(images_per_class * 2, num_classes * 2))
    if num_classes == 1:
        ax = ax.reshape(1, -1)
    if images_per_class == 1:
        ax = ax.reshape(-1, 1)
    
    for i in range(num_classes):
        for j in range(images_per_class):
            idx = i * images_per_class + j
            if idx < images.shape[0]:
                ax[i, j].imshow(images[idx, sample_channel, :, :], aspect='auto')
                ax[i, j].axis("off")
                ax[i, j].set_title(f"Class {i}")
    
    plt.tight_layout()
    plt.savefig(f"{pic_dir}/conditional_samples_epoch{epoch:04d}.png", dpi=400)
    plt.close()

    # ================== 图 2：基于 img_to_ts 还原 EEG 信号并画图 ==================
    try:
        # images: (B, C, H, W)，与 DelayEmbedder.ts_to_img 的输出一致
        img_tensor = torch.from_numpy(images).float()  # (B, C, H, W)

        # 使用与预处理相同的 DelayEmbedder 参数
        seq_len = 1000  # BCI2a 每 trial 的时间点数
        delay = config.delay
        embedding = config.embedding

        # 按照 DelayEmbedder.ts_to_img 的逻辑计算原始列数 original_rows（未 pad 前的宽度）
        i = 0
        while (i * delay + embedding) <= seq_len:
            i += 1
        if i * delay != seq_len and i * delay + embedding > seq_len:
            i += 1

        original_cols = embedding
        original_rows = i

        embedder = DelayEmbedder(device="cpu", seq_len=seq_len, delay=delay, embedding=embedding)
        # 手动设置 img_shape，供 unpad 使用
        embedder.img_shape = (img_tensor.shape[0], img_tensor.shape[1], original_cols, original_rows)

        # 利用 img_to_ts 将图像还原为时间序列: (B, T, C)
        ts_tensor = embedder.img_to_ts(img_tensor)
        ts_np = ts_tensor.detach().cpu().numpy()  # (B, T, C)

        # 画时间域 EEG，大图按 epoch 存一张（布局与图片域一致：num_classes x images_per_class）
        fig_ts, ax_ts = plt.subplots(num_classes, images_per_class, figsize=(images_per_class * 2.5, num_classes * 2.5))
        if num_classes == 1:
            ax_ts = ax_ts.reshape(1, -1)
        if images_per_class == 1:
            ax_ts = ax_ts.reshape(-1, 1)

        for i_class in range(num_classes):
            for j_img in range(images_per_class):
                idx = i_class * images_per_class + j_img
                if idx < ts_np.shape[0]:
                    sample = ts_np[idx]  # (T, C)
                    t = np.arange(sample.shape[0])
                    offset = 0.0
                    for ch in range(sample.shape[1]):
                        ax_ts[i_class, j_img].plot(t, sample[:, ch] + offset, linewidth=0.4)
                        offset += 5.0
                    ax_ts[i_class, j_img].axis("off")
                    ax_ts[i_class, j_img].set_title(f"EEG C{i_class}")

        plt.tight_layout()
        plt.savefig(f"{pic_dir}/conditional_samples_epoch{epoch:04d}_ts.png", dpi=400)
        plt.close(fig_ts)
    except Exception as e:
        print(f"基于 img_to_ts 的 EEG 信号可视化失败: {e}")
    
    unet.train()


if __name__ == "__main__":
    main()

