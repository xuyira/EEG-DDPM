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