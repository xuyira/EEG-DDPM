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

    
"""
EEG 信号可视化工具
用于可视化形状为 (N, 1000, 22) 或 (N, 22, 1000) 的 NPY 文件

用法：
    python visualize_eeg.py --input path/to/data.npy --sample-idx 0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(npy_path: Path, return_label: bool = False):
    """
    加载 NPY 文件，自动处理不同的数据格式
    
    参数:
        npy_path: NPY 文件路径
        return_label: 是否返回标签（如果存在）
    
    返回:
        如果 return_label=True 且存在标签，返回 (data, label)
        否则只返回 data
    """
    data = np.load(npy_path, allow_pickle=True)
    label = None
    
    # 如果是字典格式（包含 'data' 和 'label'）
    if isinstance(data, np.ndarray) and data.dtype == object:
        data_dict = data.item()
        if isinstance(data_dict, dict) and 'data' in data_dict:
            data = data_dict['data']
            label = data_dict.get('label', None)
            print(f"检测到字典格式，data 形状: {data.shape}")
            if label is not None:
                label = np.array(label)
                print(f"标签形状: {label.shape}, 唯一值: {np.unique(label)}")
        else:
            data = data_dict
    elif isinstance(data, dict):
        if 'data' in data:
            label = data.get('label', None)
            data = data['data']
            print(f"检测到字典格式，data 形状: {data.shape}")
            if label is not None:
                label = np.array(label)
                print(f"标签形状: {label.shape}, 唯一值: {np.unique(label)}")
    
    if return_label:
        return data, label
    return data


def visualize_eeg(data: np.ndarray, sample_idx: int = 0, output_path: Path = None, 
                  channel_names: list = None, title: str = None, stacked: bool = True):
    """
    可视化 EEG 信号
    
    参数:
        data: 形状为 (N, 1000, 22) 或 (N, 22, 1000) 的数组
        sample_idx: 要可视化的样本索引（从0开始）
        output_path: 保存图片的路径（可选）
        channel_names: 通道名称列表（可选）
        title: 图表标题（可选）
        stacked: 是否使用堆叠显示（True=垂直堆叠不重叠，False=重叠显示）
    """
    # 检查数据形状并标准化为 (N, 1000, 22)
    if data.ndim == 3:
        n_samples, dim1, dim2 = data.shape
        
        # 判断是 (N, 1000, 22) 还是 (N, 22, 1000)
        if dim1 == 1000 and dim2 == 22:
            # 已经是 (N, 1000, 22)
            pass
        elif dim1 == 22 and dim2 == 1000:
            # 需要转置为 (N, 1000, 22)
            data = np.transpose(data, (0, 2, 1))
        else:
            # 尝试自动判断：如果 dim1 > dim2，可能是 (N, 1000, 22)
            if dim1 > dim2:
                print(f"警告: 数据形状为 {data.shape}，假设为 (N, 时间点, 通道)")
            else:
                print(f"警告: 数据形状为 {data.shape}，假设为 (N, 通道, 时间点)，将转置")
                data = np.transpose(data, (0, 2, 1))
    else:
        raise ValueError(f"数据维度应为3，当前为 {data.ndim}，形状: {data.shape}")
    
    # 检查样本索引
    if sample_idx < 0 or sample_idx >= data.shape[0]:
        raise ValueError(f"样本索引 {sample_idx} 超出范围 [0, {data.shape[0]-1}]")
    
    # 提取单个样本: (1000, 22)
    sample = data[sample_idx]
    n_timepoints, n_channels = sample.shape
    
    print(f"数据形状: {data.shape}")
    print(f"选择样本 {sample_idx}: 形状 {sample.shape}")
    print(f"时间点数: {n_timepoints}, 通道数: {n_channels}")
    
    # 创建图形
    plt.figure(figsize=(14, 8))
    
    # 设置通道名称
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        print(f"警告: 通道名称数量 ({len(channel_names)}) 与通道数 ({n_channels}) 不匹配，使用默认名称")
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    # 时间轴
    time_axis = np.arange(n_timepoints)
    
    # 绘制所有通道
    colors = plt.cm.tab20(np.linspace(0, 1, n_channels))
    
    if stacked:
        # 堆叠显示：计算每个通道的偏移量，使它们垂直排列不重叠
        # 计算每个通道的信号范围
        channel_ranges = []
        for ch_idx in range(n_channels):
            ch_data = sample[:, ch_idx]
            ch_range = np.max(ch_data) - np.min(ch_data)
            channel_ranges.append(ch_range if ch_range > 0 else 1.0)
        
        # 计算总范围，用于确定偏移量
        max_range = max(channel_ranges)
        offset_step = max_range * 1.5  # 通道之间的间距
        
        # 从下往上绘制通道（通道1在底部，通道22在顶部）
        y_ticks = []
        y_tick_labels = []
        
        for ch_idx in range(n_channels):
            ch_data = sample[:, ch_idx]
            # 计算偏移量：从底部开始，每个通道向上偏移
            offset = ch_idx * offset_step
            # 将信号居中（减去均值，使信号在0附近）
            ch_data_centered = ch_data - np.mean(ch_data)
            ch_data_offset = ch_data_centered + offset
            
            plt.plot(time_axis, ch_data_offset, 
                    label=channel_names[ch_idx], 
                    color=colors[ch_idx],
                    linewidth=1.0,
                    alpha=0.8)
            
            # 记录y轴刻度位置和标签
            y_ticks.append(offset)
            y_tick_labels.append(channel_names[ch_idx])
        
        # 设置y轴刻度和标签
        plt.yticks(y_ticks, y_tick_labels, fontsize=8)
        plt.ylabel('通道 (Channels)', fontsize=12)
        
        # 添加分隔线（可选）
        for i in range(1, n_channels):
            plt.axhline(y=(i - 0.5) * offset_step, color='gray', 
                       linestyle='--', linewidth=0.5, alpha=0.3)
    else:
        # 重叠显示：原有方式
        for ch_idx in range(n_channels):
            plt.plot(time_axis, sample[:, ch_idx], 
                    label=channel_names[ch_idx], 
                    color=colors[ch_idx],
                    linewidth=1.0,
                    alpha=0.8)
        
        plt.ylabel('信号值 (Signal Value)', fontsize=12)
        
        # 添加图例（如果通道数不太多）
        if n_channels <= 22:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                      ncol=1, fontsize=8)
        else:
            print(f"通道数 ({n_channels}) 较多，不显示图例")
    
    # 设置标签和标题
    plt.xlabel('Time Points', fontsize=12)
    
    if title is None:
        title = f'EEG-sample {sample_idx}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()

