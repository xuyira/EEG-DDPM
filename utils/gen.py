import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from diffusers import DDPMPipeline
from utils.data_preprocessing import DelayEmbedder
from tqdm import tqdm

def generate_eeg_for_label(config, model, noise_scheduler, target_label, n_samples, save_path):
    device = next(model.parameters()).device

    # 1. 构造标签
    labels = torch.full((n_samples,), int(target_label), dtype=torch.long, device=device)

    # 2. 初始化噪声
    shape = (n_samples, config.unet_in_channels, *config.image_size)
    sample = torch.randn(shape, device=device)

    # 3. 走反向扩散
    model.eval()
    with torch.no_grad():
        timesteps = noise_scheduler.timesteps.to(device)
        # 显示扩散过程的进度条
        for t in tqdm(timesteps, desc=f"生成 {n_samples} 个样本 (扩散去噪)", unit="step"):
            noise_pred = model(sample, t, labels, return_dict=False)
            sample = noise_scheduler.step(noise_pred, t, sample).prev_sample

    # 4. 图像 -> 时间序列
    print("正在将图像转换为时间序列...")
    images = sample.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B,H,W,C) 如果你想复用 evaluate_conditional 下面那段，也可以先转成 (B,C,H,W)
    images_bchw = np.transpose(images, (0, 3, 1, 2))
    img_tensor = torch.from_numpy(images_bchw).float()

    from utils.data_preprocessing import DelayEmbedder
    seq_len = 1000
    delay = config.delay
    embedding = config.embedding

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

    # 5. 保存到本地
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, ts_np)
    print(f"已为类别 {target_label} 生成 {n_samples} 条 EEG 数据，形状: {ts_np.shape}")
    print(f"已保存到: {save_path}")