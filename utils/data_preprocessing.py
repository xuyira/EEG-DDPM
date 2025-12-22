from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.io
import pywt
import gc
from tqdm import tqdm
from abc import ABC, abstractmethod
import torch

def load_mat_T(root: Path, sub: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 A0{sub}T.mat，返回 (data, label, sub)。

    返回：
        data: (B, 22, 1000)
        label: (B,)  值为 0..3
        sub: (B,)  值为 sub-1，范围是 0-8，每个样本都有
    """
    mat_path = root / f"A0{sub}T.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"文件不存在: {mat_path}")

    mat = scipy.io.loadmat(mat_path)
    data = np.asarray(mat["data"])
    label = np.asarray(mat["label"]).squeeze()

    # 原始 A0*T.mat 实际常见形状: (1000, 22, B) 或 (B, 1000, 22)
    # 这里统一转换为 (B, 22, 1000)
    if data.ndim != 3:
        raise ValueError(f"{mat_path} 中 data 维度应为 3，实际为 {data.ndim}，形状: {data.shape}")

    shape = data.shape
    if 1000 in shape and 22 in shape:
        t_axis = shape.index(1000)
        c_axis = shape.index(22)
        b_axis = [i for i in range(3) if i not in (t_axis, c_axis)][0]
        data = np.transpose(data, (b_axis, c_axis, t_axis))  # (B, 22, 1000)
    else:
        raise ValueError(
            f"{mat_path} 中 data 形状无法识别为包含 (T=1000, C=22) 的三维数组: {shape}"
        )

    label = label.astype(np.int64)
    unique_vals = np.unique(label)
    if np.any((unique_vals < 1) | (unique_vals > 4)):
        raise ValueError(
            f"{mat_path} 中标签值不在 1..4 范围内: {unique_vals}"
        )
    label = label - 1

    # 创建 sub 数组，值为 sub-1，形状为 (B,)
    B = data.shape[0]
    sub_value = sub - 1
    if sub_value < 0 or sub_value > 8:
        raise ValueError(f"sub 值 {sub} 减 1 后为 {sub_value}，不在 0-8 范围内")
    sub_array = np.full(B, sub_value, dtype=np.int64)

    return data, label, sub_array

def build_loso_train_npy(
    root: Path,
    leave_sub: int,
    subjects: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建 LOSO 训练 NPY：合并所有 sub != leave_sub 的 A0%dT.mat。

    输出：
        data: (B_total, 1000, 22)
        label: (B_total,)
        sub: (B_total,)
    
    返回：
        (data, label, sub): 合并后的数据、标签、受试者编号
    """
    all_data: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_subs: List[np.ndarray] = []

    print(f"\n{'='*60}")
    print(f"生成 LOSO{leave_sub} 训练 NPY（排除 subject {leave_sub}）")
    print(f"{'='*60}")

    for sub in subjects:
        if sub == leave_sub:
            continue
        print(f"  加载 A0{sub}T.mat ...")
        data_bct, labels, sub_array = load_mat_T(root, sub)  # (B, 22, 1000), (B,), (B,)
        B = data_bct.shape[0]
        print(
            f"    形状: data={data_bct.shape}, label={labels.shape}, sub={sub_array.shape}, "
            f"B={B}, 标签唯一值={np.unique(labels)}, sub值={np.unique(sub_array)}"
        )

        # 转为 (B, 1000, 22)
        data_btc = np.transpose(data_bct, (0, 2, 1))
        all_data.append(data_btc)
        all_labels.append(labels)
        all_subs.append(sub_array)

    if not all_data:
        raise RuntimeError(f"未找到可用于 LOSO{leave_sub} 的训练数据。")

    merged_data = np.concatenate(all_data, axis=0)  # (B_total, 1000, 22)
    merged_labels = np.concatenate(all_labels, axis=0)  # (B_total,)
    merged_subs = np.concatenate(all_subs, axis=0)  # (B_total,)
    

    print(
        f"[✓] 实现LOSO{leave_sub}训练数据合并，形状: data={merged_data.shape}, "
        f"label={merged_labels.shape}, sub={merged_subs.shape} "
        f"(B_total={merged_data.shape[0]}, T={merged_data.shape[1]}, C={merged_data.shape[2]})"
    )
    return merged_data, merged_labels, merged_subs

def normalize_eeg(eeg):
    """
    eeg: (b, T, C)
    z-score normalization (per trial, per channel)
    """
    mean = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True) + 1e-8
    return (eeg - mean) / std

def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass


class DelayEmbedder(TsImgEmbedder):
    """
    Delay embedding transformation
    """

    def __init__(self, device, seq_len, delay, embedding):
        super().__init__(device, seq_len)
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None

    def pad_to_square(self, x, mask=0):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (
            0, max_side - rows, 0, max_side - cols)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def ts_to_img(self, signal, pad=True, mask=0):

        batch, length, features = signal.shape
        #  if our sequences are of different lengths, this can happen with physionet and climate datasets
        if self.seq_len != length:
            self.seq_len = length

        x_image = torch.zeros((batch, features, self.embedding, self.embedding))
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        ### SPECIAL CASE
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            # end = start + (self.embedding - 1) - missing_vals
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1

        # cache the shape of the image before padding
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)

        return x_image

    def img_to_ts(self, img):
        img_non_square = self.unpad(img, self.img_shape)

        batch, channels, rows, cols = img_non_square.shape

        reconstructed_x_time_series = torch.zeros((batch, channels, self.seq_len))

        for i in range(cols - 1):
            start = i * self.delay
            end = start + self.embedding
            reconstructed_x_time_series[:, :, start:end] = img_non_square[:, :, :, i]

        ### SPECIAL CASE
        start = (cols - 1) * self.delay
        end = reconstructed_x_time_series[:, :, start:].shape[-1]
        reconstructed_x_time_series[:, :, start:] = img_non_square[:, :, :end, cols - 1]
        reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)

        return reconstructed_x_time_series.cuda()

def bci2a_preprcessing_method(
    root: Path,
    leave_sub: int,
    subjects: Iterable[int],
    delay: int,
    embedding: int,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对留一受试者（LOSO）的训练数据进行预处理：
    先生成合并后的 NPY，再通过 DelayEmbedder.ts_to_img 将时序信号转成图片。

    参数：
        root: mat 文件所在目录
        leave_sub: 留出的受试者编号（1-9）
        subjects: 所有受试者编号列表
        delay: 延迟嵌入的步长（DelayEmbedder 的 delay）
        embedding: 延迟嵌入的维度（DelayEmbedder 的 embedding）
        device: 运行 DelayEmbedder 的设备字符串，例如 "cpu" 或 "cuda"

    返回：
        (img_data, label, sub): 
            img_data: (B_total, C, H, W) 由 DelayEmbedder 生成并按需要补成方形的图像
            label: (B_total,) 标签数组，值为 0..3
            sub: (B_total,) 受试者编号数组，值为 0-8
    """
    # 1. 调用 build_loso_train_npy 生成 NPY 文件并获取合并后的数据
    eeg_data, merged_labels, merged_subs = build_loso_train_npy(
        root, leave_sub, subjects
    )

    # eeg_data 形状为 (B_total, T, C)
    B_total, T, C = eeg_data.shape
    print(
        f"\n准备进行 DelayEmbedder 变换，数据形状: {eeg_data.shape} "
        f"(B_total={B_total}, T={T}, C={C})"
    )
    print(f"  DelayEmbedder 参数: delay={delay}, embedding={embedding}, device={device}")

    # 2. 归一化 (与 CWT 前相同的 z-score 归一化策略)
    eeg_data_norm = normalize_eeg(eeg_data)

    # 3. 转为 tensor 并移动到指定设备
    eeg_tensor = torch.from_numpy(eeg_data_norm).float().to(device)

    # 4. 使用 DelayEmbedder 将时序信号转为图片
    embedder = DelayEmbedder(device=device, seq_len=T, delay=delay, embedding=embedding)

    with torch.no_grad():
        img_tensor = embedder.ts_to_img(eeg_tensor, pad=True, mask=0)  # (B, C, H, W)

    # 5. 转回 numpy，并清理中间变量以节省内存
    img_data = img_tensor.detach().cpu().numpy().astype(np.float32)

    del eeg_data, eeg_data_norm, eeg_tensor, img_tensor
    gc.collect()

    print(
        f"[✓] DelayEmbedder 变换完成，输出形状: img_data={img_data.shape}, "
        f"label={merged_labels.shape}, sub={merged_subs.shape}"
    )
    print(
        f"    (B_total={img_data.shape[0]}, C={img_data.shape[1]}, "
        f"H={img_data.shape[2]}, W={img_data.shape[3]})"
    )

    return img_data, merged_labels, merged_subs