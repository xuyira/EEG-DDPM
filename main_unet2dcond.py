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
import pandas as pd

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
    
    # 创建条件模型（使用交叉注意力）
    print("\n创建条件模型（交叉注意力方式）...")
    num_classes = len(np.unique(X_train_label))  # 自动检测类别数量
    print(f"类别数量: {num_classes}, 交叉注意力维度: {config.cross_attention_dim}, encoder_hid_dim: {config.encoder_hid_dim}")
    
    unet, label_embedder = create_unet2dcond_model(
        config, 
        num_classes=num_classes
    )
    
    # 测试模型输入输出（在 CPU 上测试，避免设备冲突）
    sample_image = X_train_img[0].unsqueeze(0)  # 保持在 CPU
    sample_label = torch.tensor([X_train_label[0]], dtype=torch.long)
    sample_timestep = torch.tensor([0], dtype=torch.long)
    
    # 测试 UNet 输出（使用交叉注意力）
    with torch.no_grad():
        # 将标签转换为 embedding
        cond_emb = label_embedder(sample_label)  # (1, encoder_hid_dim)
        # 扩展维度以匹配 UNet 的期望输入 (B, seq_len, encoder_hid_dim)
        cond_emb = cond_emb.unsqueeze(1).repeat(1, 4, 1)  # (1, 4, encoder_hid_dim)
        
        output = unet(sample_image, sample_timestep, encoder_hidden_states=cond_emb).sample
        print("输入图像维度：", sample_image.shape)
        print("输入标签：", sample_label)
        print("条件嵌入维度：", cond_emb.shape)
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
    
    # 创建优化器和学习率调度器（需要同时优化 UNet 和 label_embedder）
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(label_embedder.parameters()), 
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
        label_embedder=label_embedder,
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
    config, unet, label_embedder, noise_scheduler, optimizer, 
    train_dataloader, lr_scheduler, pic_dir, device, num_classes
):
    """条件训练循环（使用交叉注意力方式）"""
    from accelerate import Accelerator
    from tqdm.auto import tqdm
    import os
    from pathlib import Path
    
    # 初始化 accelerator（支持 wandb 或 tensorboard）
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=config.log_with,  # 从 config 读取，可以是 "wandb" 或 "tensorboard"
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_conditional")
    
    # 将 unet 和 label_embedder 组合成一个模型
    class ConditionalModel(torch.nn.Module):
        def __init__(self, unet, label_embedder):
            super().__init__()
            self.unet = unet
            self.label_embedder = label_embedder
        
        def forward(self, noisy_images, timesteps, labels):
            # 将标签转换为 embedding
            cond_emb = self.label_embedder(labels)  # (B, encoder_hid_dim)
            # 扩展维度以匹配 UNet 的期望输入 (B, seq_len, encoder_hid_dim)
            cond_emb = cond_emb.unsqueeze(1).repeat(1, 4, 1)  # (B, 4, encoder_hid_dim)
            # 使用交叉注意力
            return self.unet(noisy_images, timesteps, encoder_hidden_states=cond_emb).sample
    
    model = ConditionalModel(unet, label_embedder)
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0
    
    # 用于记录训练历史（用于绘制曲线）
    training_history = {
        "step": [],
        "loss": [],
        "lr": [],
        "epoch": []
    }
    
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
                # 预测噪声残差（使用交叉注意力，将 labels 转换为 embedding）
                noise_pred = model(noisy_images, timesteps, labels)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                
                # 调试：检查 LabelEmbedder 是否在训练（只在训练初期检查一次）
                if step == 0 and epoch == 0:
                    from check_label_embedding import check_label_embedder_training, check_embedding_differences
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_label_embedder = unwrapped_model.label_embedder
                    
                    # 检查是否在优化器中
                    check_label_embedder_training(unwrapped_model, optimizer)
                    
                    # 检查 embedding 差异（初始状态）
                    print("\n[初始状态] 检查 LabelEmbedder 的初始 embedding:")
                    check_embedding_differences(unwrapped_label_embedder, num_classes, device=accelerator.device)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # 调试：检查训练后的 embedding 差异（每50个epoch检查一次）
                if step == len(train_dataloader) - 1 and (epoch + 1) % 50 == 0:
                    from check_label_embedding import check_embedding_differences
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_label_embedder = unwrapped_model.label_embedder
                    print(f"\n[训练后 - Epoch {epoch}] 检查 LabelEmbedder 的 embedding:")
                    check_embedding_differences(unwrapped_label_embedder, num_classes, device=accelerator.device)
            
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0], 
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            # 记录训练历史（用于绘制曲线和保存到文件）
            if accelerator.is_main_process:
                training_history["step"].append(global_step)
                training_history["loss"].append(loss.detach().item())
                training_history["lr"].append(lr_scheduler.get_last_lr()[0])
                training_history["epoch"].append(epoch)
            
            global_step += 1
        
        # 每个 epoch 后保存训练历史并绘制曲线
        if accelerator.is_main_process and ((epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1):
            # 保存训练历史到 CSV 文件
            save_training_history(training_history, config.output_dir, epoch)
            # 绘制并保存 loss 曲线
            plot_training_curves(training_history, config.output_dir, epoch)
        
        # 每个 epoch 后可选地生成一些样本图像
        if accelerator.is_main_process and ((epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1):
            # 解包模型
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_unet = unwrapped_model.unet
            unwrapped_label_embedder = unwrapped_model.label_embedder
            
            # 生成样本（为每个类别生成一个样本）
            generate_conditional_samples(
                unwrapped_unet, unwrapped_label_embedder, noise_scheduler, 
                config, epoch, pic_dir, num_classes, device
            )
        
        # 保存模型（按 epoch 保存，不覆盖）
        if accelerator.is_main_process and ((epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1):
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_unet = unwrapped_model.unet
            unwrapped_label_embedder = unwrapped_model.label_embedder
            
            # 保存模型（按 epoch 保存）
            base_save_dir = Path(config.output_dir)
            base_save_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建 epoch 特定的保存目录
            epoch_save_dir = base_save_dir / f"epoch_{epoch:04d}"
            epoch_save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存 UNet
            unet_save_dir = epoch_save_dir / "unet"
            unwrapped_unet.save_pretrained(unet_save_dir)
            # 保存 label_embedder
            torch.save(unwrapped_label_embedder.state_dict(), epoch_save_dir / "label_embedder.pt")
            print(f"模型已保存到 {epoch_save_dir}")
            
            # 同时保存到 latest 目录（方便快速访问最新模型）
            latest_save_dir = base_save_dir / "latest"
            latest_save_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_unet.save_pretrained(latest_save_dir / "unet")
            torch.save(unwrapped_label_embedder.state_dict(), latest_save_dir / "label_embedder.pt")


def generate_conditional_samples(unet, label_embedder, noise_scheduler, config, epoch, pic_dir, num_classes, device):
    """生成条件样本，并输出两种图：
    1）图片域（按类别排列的生成图像）
    2）时间域 EEG（通过 DelayEmbedder.img_to_ts 还原）
    """
    unet.eval()
    label_embedder.eval()
    
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
            
            # 将标签转换为 embedding
            cond_emb = label_embedder(labels)  # (B, encoder_hid_dim)
            # 扩展维度以匹配 UNet 的期望输入 (B, seq_len, encoder_hid_dim)
            cond_emb = cond_emb.unsqueeze(1).repeat(1, 4, 1)  # (B, 4, encoder_hid_dim)
            
            # 生成随机噪声
            shape = (images_per_class, config.unet_in_channels, *config.image_size)
            noisy_images = torch.randn(shape, device=device)
            
            # 反向扩散过程（使用交叉注意力）
            for t in noise_scheduler.timesteps:
                timesteps = torch.full((images_per_class,), t, dtype=torch.long, device=device)
                noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=cond_emb).sample
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
    label_embedder.train()


def save_training_history(training_history, output_dir, current_epoch):
    """
    保存训练历史到 CSV 文件
    
    参数:
        training_history: 包含 step, loss, lr, epoch 的字典
        output_dir: 输出目录
        current_epoch: 当前 epoch
    """
    try:
        # 创建 DataFrame
        df = pd.DataFrame(training_history)
        
        # 保存到 CSV
        csv_path = Path(output_dir) / "training_history.csv"
        df.to_csv(csv_path, index=False)
        
        # 每 10 个 epoch 打印一次保存信息
        if current_epoch % 10 == 0:
            print(f"训练历史已保存到: {csv_path} (共 {len(df)} 条记录)")
    except Exception as e:
        print(f"保存训练历史失败: {e}")


def plot_training_curves(training_history, output_dir, current_epoch):
    """
    绘制并保存训练曲线（loss 和 learning rate）
    
    参数:
        training_history: 包含 step, loss, lr, epoch 的字典
        output_dir: 输出目录
        current_epoch: 当前 epoch
    """
    try:
        if len(training_history["step"]) == 0:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        steps = training_history["step"]
        losses = training_history["loss"]
        lrs = training_history["lr"]
        epochs = training_history["epoch"]
        
        # 图1: Loss 曲线
        ax1 = axes[0]
        ax1.plot(steps, losses, 'b-', alpha=0.6, linewidth=0.5, label='Loss')
        
        # 计算移动平均（每 100 个点平均）
        if len(losses) > 100:
            window_size = min(100, len(losses) // 10)
            moving_avg = pd.Series(losses).rolling(window=window_size, center=True).mean()
            ax1.plot(steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Loss (Epoch {current_epoch + 1})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: Learning Rate 曲线
        ax2 = axes[1]
        ax2.plot(steps, lrs, 'g-', alpha=0.6, linewidth=0.5, label='Learning Rate')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # 使用对数刻度
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = Path(output_dir) / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 每 10 个 epoch 打印一次保存信息
        if current_epoch % 10 == 0:
            print(f"训练曲线已保存到: {plot_path}")
            
    except Exception as e:
        print(f"绘制训练曲线失败: {e}")


if __name__ == "__main__":
    main()

