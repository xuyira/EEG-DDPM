"""
可视化一个 EEG 样本 (1, 1000, 22) 的三种转图片方法：
1. DelayEmbedder - 延迟嵌入
2. CWT - 连续小波变换
3. STFTEmbedder - 短时傅里叶变换

每个通道显示三种转换结果，共 22 张图，每张图包含 3 个子图。
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchaudio.transforms as T
from tqdm import tqdm

# 导入 onlypre.py 中的类和函数
import sys
sys.path.insert(0, str(Path(__file__).parent))
from onlypre import (
    load_mat_T,
    normalize_eeg,
    DelayEmbedder,
    eeg_to_cwt,
    STFTEmbedder,
    MinMaxScaler,
    MinMaxArgs
)


def load_sample(data_root, sub, sample_idx=0):
    """
    加载一个样本
    
    返回:
        sample: (1, 1000, 22) - 单个样本的 EEG 数据
        label: 标签值 (0-3)
    """
    data, labels, _ = load_mat_T(Path(data_root), sub)
    # data 形状: (B, 22, 1000)
    # 转换为 (B, 1000, 22)
    data = np.transpose(data, (0, 2, 1))
    
    if sample_idx >= len(data):
        raise ValueError(f"样本索引 {sample_idx} 超出范围，共有 {len(data)} 个样本")
    
    sample = data[sample_idx:sample_idx+1]  # (1, 1000, 22)
    label = labels[sample_idx]
    
    print(f"加载样本 {sample_idx}: 形状={sample.shape}, 标签={label}")
    return sample, label


def apply_delay_embedding(sample, delay=15, embedding=64, device="cpu"):
    """
    应用 DelayEmbedder 转换
    
    参数:
        sample: (1, 1000, 22)
    
    返回:
        img: (1, 22, H, W) - 每个通道一张图
    """
    # 归一化
    sample_norm = normalize_eeg(sample)
    
    # 转为 tensor
    sample_tensor = torch.from_numpy(sample_norm).float().to(device)
    
    # DelayEmbedder
    embedder = DelayEmbedder(device=device, seq_len=1000, delay=delay, embedding=embedding)
    with torch.no_grad():
        img = embedder.ts_to_img(sample_tensor, pad=True, mask=0)  # (1, 22, H, W)
    
    return img.detach().cpu().numpy()


def apply_cwt(sample, fs=1000, num_freqs=50, fmin=1, fmax=50, wavelet='morl'):
    """
    应用 CWT 转换
    
    参数:
        sample: (1, 1000, 22)
    
    返回:
        cwt_img: (1, 22, num_freqs, T) - 每个通道一张时频图
    """
    cwt_data = eeg_to_cwt(
        sample,
        fs=fs,
        num_freqs=num_freqs,
        fmin=fmin,
        fmax=fmax,
        wavelet=wavelet,
        pbar=None
    )  # (1, 22, num_freqs, T)
    
    return cwt_data


def apply_stft(sample, n_fft=512, hop_length=128, device="cpu", use_magnitude=True):
    """
    应用 STFTEmbedder 转换
    
    参数:
        sample: (1, 1000, 22)
        n_fft: FFT 窗口大小，建议 512 或 1024（对于 1000 Hz 采样率，512 提供约 2 Hz 频率分辨率）
        hop_length: 跳跃长度，建议 n_fft/4 或 n_fft/8（平衡时间分辨率和频率分辨率）
        device: 计算设备
        use_magnitude: 是否使用功率谱（magnitude）而不是实部，默认 True（推荐用于脑电信号）
    
    返回:
        stft_img: (1, 22, H, W) - 功率谱或实部
    """
    # 归一化
    sample_norm = normalize_eeg(sample)
    
    # 转为 tensor
    sample_tensor = torch.from_numpy(sample_norm).float().to(device)
    
    # STFTEmbedder
    embedder = STFTEmbedder(device=device, seq_len=1000, n_fft=n_fft, hop_length=hop_length)
    
    # 需要先缓存 min/max 参数（使用训练数据，这里用样本本身）
    embedder.cache_min_max_params(sample_tensor)
    
    with torch.no_grad():
        # 获取 STFT 的实部和虚部
        real, imag = embedder.stft_transform(sample_tensor)
        
        if use_magnitude:
            # 计算功率谱（magnitude）：|real + i*imag| = sqrt(real^2 + imag^2)
            # 这对于脑电信号更直观，能更好地显示不同频段的能量
            magnitude = torch.sqrt(real ** 2 + imag ** 2)  # (1, 22, n_freq_bins, n_time_frames)
            stft_result = magnitude.detach().cpu().numpy()  # (1, 22, H, W)
        else:
            # 使用归一化后的实部（原始方法）
            stft_img = embedder.ts_to_img(sample_tensor)  # (1, 44, H, W)
            stft_result = stft_img[:, :22, :, :].detach().cpu().numpy()  # (1, 22, H, W)
    
    return stft_result


def visualize_three_methods(
    sample,
    delay_img,
    cwt_img,
    stft_img,
    save_path,
    channel_names=None
):
    """
    可视化三种转换方法的结果
    
    参数:
        sample: (1, 1000, 22) - 原始样本
        delay_img: (1, 22, H, W) - DelayEmbedder 结果
        cwt_img: (1, 22, num_freqs, T) - CWT 结果
        stft_img: (1, 22, H, W) - STFT 结果（实部）
        save_path: 保存路径
        channel_names: 通道名称列表（可选）
    """
    num_channels = 22
    
    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(num_channels)]
    
    # 创建大图：22行，每行3列（三种方法）
    # 使用更大的图形尺寸以适应22个通道
    fig, axes = plt.subplots(num_channels, 3, figsize=(18, 4 * num_channels))
    
    if num_channels == 1:
        axes = axes.reshape(1, -1)
    
    for ch in range(num_channels):
        # DelayEmbedder 结果
        ax1 = axes[ch, 0]
        delay_ch = delay_img[0, ch, :, :]  # (H, W)
        im1 = ax1.imshow(delay_ch, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title(f'{channel_names[ch]} - DelayEmbedder', fontsize=9, pad=5)
        ax1.set_xlabel('Time Window', fontsize=8)
        ax1.set_ylabel('Embedding', fontsize=8)
        ax1.tick_params(labelsize=7)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
        
        # CWT 结果
        ax2 = axes[ch, 1]
        cwt_ch = cwt_img[0, ch, :, :]  # (num_freqs, T)
        im2 = ax2.imshow(cwt_ch, aspect='auto', cmap='jet', origin='lower')
        ax2.set_title(f'{channel_names[ch]} - CWT', fontsize=9, pad=5)
        ax2.set_xlabel('Time', fontsize=8)
        ax2.set_ylabel('Frequency', fontsize=8)
        ax2.tick_params(labelsize=7)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
        
        # STFT 结果
        ax3 = axes[ch, 2]
        stft_ch = stft_img[0, ch, :, :]  # (H, W)
        im3 = ax3.imshow(stft_ch, aspect='auto', cmap='hot', origin='lower')
        # 根据是否使用功率谱调整标题
        stft_title = 'STFT (Magnitude)' if stft_img.shape[1] == 22 else 'STFT (Real)'
        ax3.set_title(f'{channel_names[ch]} - {stft_title}', fontsize=9, pad=5)
        ax3.set_xlabel('Time Frame', fontsize=8)
        ax3.set_ylabel('Frequency Bin', fontsize=8)
        ax3.tick_params(labelsize=7)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)
    
    # 添加总标题
    fig.suptitle('EEG 样本三种转换方法对比 (22个通道 × 3种方法)', 
                 fontsize=14, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # 为总标题留出空间
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化一个 EEG 样本的三种转图片方法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 可视化 sub1 的第 0 个样本
  python visualize_three_methods.py --data_root EEG-Conformer/data/standard_2a_data/ --sub 1 --sample_idx 0
  
  # 可视化 sub3 的第 10 个样本，使用推荐的脑电信号参数
  python visualize_three_methods.py --data_root EEG-Conformer/data/standard_2a_data/ \\
                                    --sub 3 --sample_idx 10 \\
                                    --delay 15 --embedding 64 \\
                                    --n_fft 512 --hop_length 128 \\
                                    --stft_use_magnitude
  
  # 使用更高频率分辨率（n_fft=1024）
  python visualize_three_methods.py --data_root EEG-Conformer/data/standard_2a_data/ \\
                                    --sub 1 --sample_idx 0 \\
                                    --n_fft 1024 --hop_length 256
        """
    )
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录，包含 A0*T.mat 文件')
    parser.add_argument('--sub', type=int, default=1,
                       help='受试者编号 (1-9)，默认: 1')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='样本索引，默认: 0')
    parser.add_argument('--delay', type=int, default=15,
                       help='DelayEmbedder 的 delay 参数，默认: 15')
    parser.add_argument('--embedding', type=int, default=64,
                       help='DelayEmbedder 的 embedding 参数，默认: 64')
    parser.add_argument('--n_fft', type=int, default=512,
                       help='STFTEmbedder 的 n_fft 参数（FFT窗口大小），默认: 512。'
                            '对于1000Hz采样率：512提供约2Hz频率分辨率，1024提供约1Hz频率分辨率。'
                            '建议值：512（平衡）或 1024（更高频率分辨率）')
    parser.add_argument('--hop_length', type=int, default=128,
                       help='STFTEmbedder 的 hop_length 参数（跳跃长度），默认: 128。'
                            '建议值为 n_fft/4 或 n_fft/8，用于平衡时间分辨率和频率分辨率。'
                            '对于 n_fft=512，建议值：128（默认）或 64（更高时间分辨率）')
    parser.add_argument('--stft_use_magnitude', action='store_true', default=True,
                       help='STFT 可视化使用功率谱（magnitude）而不是实部，默认: True。'
                            '功率谱更适合脑电信号，能更好地显示不同频段的能量分布。'
                            '使用 --no_stft_use_magnitude 来禁用（使用实部）')
    parser.add_argument('--no_stft_use_magnitude', dest='stft_use_magnitude', action='store_false',
                       help='禁用功率谱，使用实部进行 STFT 可视化')
    parser.add_argument('--num_freqs', type=int, default=50,
                       help='CWT 的频率数量，默认: 50')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备，默认: cpu')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                       help='输出目录，默认: ./visualization_results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EEG 样本三种转换方法可视化")
    print("=" * 60)
    
    # 1. 加载样本
    print(f"\n[1/4] 加载样本...")
    sample, label = load_sample(args.data_root, args.sub, args.sample_idx)
    
    # 2. 应用 DelayEmbedder
    print(f"\n[2/4] 应用 DelayEmbedder (delay={args.delay}, embedding={args.embedding})...")
    delay_img = apply_delay_embedding(
        sample,
        delay=args.delay,
        embedding=args.embedding,
        device=args.device
    )
    print(f"  DelayEmbedder 输出形状: {delay_img.shape}")
    
    # 3. 应用 CWT
    print(f"\n[3/4] 应用 CWT (num_freqs={args.num_freqs})...")
    cwt_img = apply_cwt(sample, num_freqs=args.num_freqs)
    print(f"  CWT 输出形状: {cwt_img.shape}")
    
    # 4. 应用 STFT
    print(f"\n[4/4] 应用 STFTEmbedder (n_fft={args.n_fft}, hop_length={args.hop_length}, use_magnitude={args.stft_use_magnitude})...")
    print(f"  参数说明:")
    print(f"    - n_fft={args.n_fft}: FFT窗口大小，频率分辨率 ≈ {1000/args.n_fft:.2f} Hz")
    print(f"    - hop_length={args.hop_length}: 跳跃长度，时间分辨率 ≈ {args.hop_length/1000*1000:.1f} ms")
    print(f"    - use_magnitude={args.stft_use_magnitude}: 使用功率谱（推荐）" if args.stft_use_magnitude else f"    - use_magnitude={args.stft_use_magnitude}: 使用实部")
    stft_img = apply_stft(
        sample,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        device=args.device,
        use_magnitude=args.stft_use_magnitude
    )
    print(f"  STFT 输出形状: {stft_img.shape}")
    
    # 5. 可视化
    print(f"\n[5/5] 生成可视化...")
    save_path = output_dir / f"three_methods_sub{args.sub}_sample{args.sample_idx}_label{label}.png"
    visualize_three_methods(
        sample,
        delay_img,
        cwt_img,
        stft_img,
        save_path
    )
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

