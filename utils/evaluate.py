import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from diffusers import DDPMPipeline
from utils.data_preprocessing import DelayEmbedder

def evaluate(config, epoch, pipeline, pic_dir, sample_channel=0):
    # 从随机噪声中生成一些图像（这是反向扩散过程）。
    # 默认的管道输出类型是 `List[torch.Tensor]`
    # 取sample_channel来展示

    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed),  # 使用单独的 torch 生成器来避免回绕主训练循环的随机状态
        output_type="np.array",
    ).images

    print("生成图像维度：", images.shape)
    # (batch, height, width, channel)

    # 将生成的eval_batch_size个图像拼接成一张大图（图片域）
    fig, ax = plt.subplots(2, 8, figsize=(20, 4))
    for i in range(2):
        for j in range(8):
            if i * 8 + j < images.shape[0]:
                ax[i, j].imshow(images[i * 8 + j, :, :, sample_channel], aspect='auto')
                ax[i, j].axis("off")
                ax[i, j].set_title(f"Image {i * 8 + j}")

    plt.savefig(f"{pic_dir}/{epoch:04d}.png", dpi=400)
    plt.close()

    # ================== 基于 img_to_ts 还原 EEG 信号并画图 ==================
    try:
        import numpy as np

        # images: (B, H, W, C) -> (B, C, H, W) 并转为 tensor
        images_bchw = np.transpose(images, (0, 3, 1, 2))
        img_tensor = torch.from_numpy(images_bchw).float()

        # 使用与预处理相同的 DelayEmbedder 参数
        # BCI2a 每 trial 的时间点数为 1000，这里直接使用 1000
        seq_len = 1000
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

        # 画第二种图：时间域 EEG，大图按 epoch 存一张
        fig_ts, ax_ts = plt.subplots(2, 8, figsize=(20, 6))
        for i in range(2):
            for j in range(8):
                idx = i * 8 + j
                if idx < ts_np.shape[0]:
                    sample = ts_np[idx]  # (T, C) = (1000, 22)
                    t = np.arange(sample.shape[0])
                    offset = 0.0
                    # 仿照 visualize_eeg 的堆叠效果，简单做通道平移
                    for ch in range(sample.shape[1]):
                        ax_ts[i, j].plot(t, sample[:, ch] + offset, linewidth=0.4)
                        offset += 5.0
                    ax_ts[i, j].axis("off")
                    ax_ts[i, j].set_title(f"EEG {idx}")

        plt.tight_layout()
        plt.savefig(f"{pic_dir}/{epoch:04d}_ts.png", dpi=400)
        plt.close(fig_ts)
    except Exception as e:
        print(f"基于 img_to_ts 的 EEG 信号可视化失败: {e}")


def evaluate_conditional(config, epoch, model, noise_scheduler, pic_dir, sample_channel=0):
    """
    条件生成评估函数，支持类别标签作为条件
    """
    import numpy as np
    from tqdm.auto import tqdm
    
    device = next(model.parameters()).device
    num_classes = 4  # BCI2a 有 4 个类别 (0-3)
    
    # 为每个类别生成一些样本
    batch_size = config.eval_batch_size
    # 生成每个类别的样本数
    samples_per_class = batch_size // num_classes
    if samples_per_class == 0:
        samples_per_class = 1
    
    # 准备类别标签：每个类别生成 samples_per_class 个样本
    class_labels = []
    for cls in range(num_classes):
        class_labels.extend([cls] * samples_per_class)
    # 如果 batch_size 不能被 num_classes 整除，补充剩余的
    remaining = batch_size - len(class_labels)
    if remaining > 0:
        class_labels.extend([0] * remaining)
    
    class_labels = torch.tensor(class_labels[:batch_size], device=device)
    
    # 初始化随机噪声
    shape = (batch_size, config.unet_in_channels, *config.image_size)
    sample = torch.randn(shape, device=device)
    
    # 采样循环
    model.eval()
    with torch.no_grad():
        # 确保 timesteps 在正确的设备上
        timesteps = noise_scheduler.timesteps.to(device)
        for t in tqdm(timesteps, desc="采样中"):
            # 预测噪声残差
            noise_pred = model(sample, t, class_labels, return_dict=False)
            
            # 更新样本
            sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
    
    # 转换为 numpy: (B, C, H, W) -> (B, H, W, C)
    images = sample.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    print("生成图像维度：", images.shape)
    # (batch, height, width, channel)
    
    # 将生成的图像拼接成一张大图（图片域）
    fig, ax = plt.subplots(2, 8, figsize=(20, 4))
    for i in range(2):
        for j in range(8):
            idx = i * 8 + j
            if idx < images.shape[0]:
                ax[i, j].imshow(images[idx, :, :, sample_channel], aspect='auto')
                ax[i, j].axis("off")
                ax[i, j].set_title(f"Label {class_labels[idx].item()}")
    
    plt.savefig(f"{pic_dir}/{epoch:04d}.png", dpi=400)
    plt.close()
    
    # ================== 基于 img_to_ts 还原 EEG 信号并画图 ==================
    try:
        # images: (B, H, W, C) -> (B, C, H, W) 并转为 tensor
        images_bchw = np.transpose(images, (0, 3, 1, 2))
        img_tensor = torch.from_numpy(images_bchw).float()
        
        # 使用与预处理相同的 DelayEmbedder 参数
        seq_len = 1000
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
        
        # 画第二种图：时间域 EEG，大图按 epoch 存一张
        fig_ts, ax_ts = plt.subplots(2, 8, figsize=(20, 6))
        for i in range(2):
            for j in range(8):
                idx = i * 8 + j
                if idx < ts_np.shape[0]:
                    sample_ts = ts_np[idx]  # (T, C) = (1000, 22)
                    t = np.arange(sample_ts.shape[0])
                    offset = 0.0
                    # 仿照 visualize_eeg 的堆叠效果，简单做通道平移
                    for ch in range(sample_ts.shape[1]):
                        ax_ts[i, j].plot(t, sample_ts[:, ch] + offset, linewidth=0.4)
                        offset += 5.0
                    ax_ts[i, j].axis("off")
                    ax_ts[i, j].set_title(f"Label {class_labels[idx].item()}")
        
        plt.tight_layout()
        plt.savefig(f"{pic_dir}/{epoch:04d}_ts.png", dpi=400)
        plt.close(fig_ts)
    except Exception as e:
        print(f"基于 img_to_ts 的 EEG 信号可视化失败: {e}")