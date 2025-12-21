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


def eeg_to_cwt(
    eeg,
    fs=1000,
    num_freqs=50,
    fmin=1,
    fmax=50,
    wavelet='morl',
    pbar=None  # 可选的进度条对象（用于显示总进度）
):
    """
    eeg: (b, T, C)
    return: (b, C, num_freqs, T)
    """

    b, T, C = eeg.shape

    # 归一化
    eeg = normalize_eeg(eeg)

    # 生成频率 → scale 映射
    freqs = np.linspace(fmin, fmax, num_freqs)
    scales = pywt.central_frequency(wavelet) * fs / freqs

    cwt_data = np.zeros((b, C, num_freqs, T), dtype=np.float32)

    # 如果没有传入进度条，创建局部进度条
    if pbar is None:
        total_tasks = b * C
        pbar = tqdm(total=total_tasks, desc="CWT转换", unit="通道")
        local_pbar = True
    else:
        local_pbar = False

    try:
        for i in range(b):
            for ch in range(C):
                coef, _ = pywt.cwt(
                    eeg[i, :, ch],
                    scales=scales,
                    wavelet=wavelet
                )
                # coef: (num_freqs, T)
                cwt_data[i, ch] = coef.real  # 通常取实部
                if pbar is not None:
                    pbar.update(1)  # 更新进度条
    finally:
        if local_pbar:
            pbar.close()

    return cwt_data

def bci2a_preprcessing_method(
    root: Path,
    leave_sub: int,
    subjects: Iterable[int],
    fs: int = 1000,
    num_freqs: int = 50,
    fmin: float = 1,
    fmax: float = 50,
    wavelet: str = 'morl',
    cwt_batch_size: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对留一受试者（LOSO）的训练数据进行预处理：生成 NPY 文件并进行 CWT 变换。
    
    参数：
        root: mat 文件所在目录
        leave_sub: 留出的受试者编号（1-9）
        subjects: 所有受试者编号列表
        fs: 采样频率，默认 1000
        num_freqs: CWT 频率数量，默认 50
        fmin: 最小频率，默认 1
        fmax: 最大频率，默认 50
        wavelet: 小波类型，默认 'morl'
        cwt_batch_size: CWT 转换批次大小（用于减少内存占用，None 表示不分批）
    
    返回：
        (cwt_data, label, sub): 
            cwt_data: (B_total, C, num_freqs, T) 归一化后的 CWT 数据
            label: (B_total,) 标签数组，值为 0..3
            sub: (B_total,) 受试者编号数组，值为 0-8
    """
    # 1. 调用 build_loso_train_npy 生成 NPY 文件并获取合并后的数据
    eeg_data, merged_labels, merged_subs = build_loso_train_npy(
        root, leave_sub, subjects
    )
    
    # eeg_data 形状为 (B_total, 1000, 22)
    B_total, T, C = eeg_data.shape
    print(f"\n准备进行 CWT 变换，数据形状: {eeg_data.shape} (B_total={B_total}, T={T}, C={C})")
    
    # 2. 调用 eeg_to_cwt 进行 CWT 变换
    # eeg_to_cwt 输入: (b, T, C)，输出: (b, C, num_freqs, T)
    print(f"\n进行 CWT 变换...")
    print(f"  参数: fs={fs}, num_freqs={num_freqs}, fmin={fmin}, fmax={fmax}, wavelet={wavelet}")
    
    # 如果设置了 cwt_batch_size，则分批处理
    if cwt_batch_size is not None and cwt_batch_size < B_total:
        print(f"  使用分批处理，批次大小: {cwt_batch_size} 样本/批次")
        # 预分配输出数组
        cwt_data = np.zeros((B_total, C, num_freqs, T), dtype=np.float32)
        
        # 创建总进度条（显示所有通道的总进度）
        total_channels = B_total * C
        pbar = tqdm(total=total_channels, desc="CWT转换（总进度）", unit="通道")
        
        # 分批处理
        num_batches = (B_total + cwt_batch_size - 1) // cwt_batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * cwt_batch_size
            end_idx = min(start_idx + cwt_batch_size, B_total)
            batch_size_actual = end_idx - start_idx  # 实际批次大小（最后一批可能小于 cwt_batch_size）
            batch_data = eeg_data[start_idx:end_idx]
            
            print(f"\n  批次 {batch_idx + 1}/{num_batches}: 处理样本 {start_idx} 到 {end_idx - 1} (共 {batch_size_actual} 个样本, {batch_size_actual * C} 个通道)")
            batch_cwt = eeg_to_cwt(
                batch_data,
                fs=fs,
                num_freqs=num_freqs,
                fmin=fmin,
                fmax=fmax,
                wavelet=wavelet,
                pbar=pbar  # 传入总进度条
            )
            cwt_data[start_idx:end_idx] = batch_cwt
            
            # 释放批次数据内存
            del batch_data, batch_cwt
            gc.collect()
        
        pbar.close()
    else:
        # 一次性处理所有数据
        cwt_data = eeg_to_cwt(
            eeg_data,
            fs=fs,
            num_freqs=num_freqs,
            fmin=fmin,
            fmax=fmax,
            wavelet=wavelet
        )
    
    # 删除原始 eeg_data 以释放内存
    del eeg_data
    gc.collect()
    
    print(f"[✓] CWT 变换完成，输出形状: cwt_data={cwt_data.shape}, label={merged_labels.shape}, sub={merged_subs.shape}")
    print(f"    (B_total={cwt_data.shape[0]}, C={cwt_data.shape[1]}, num_freqs={cwt_data.shape[2]}, T={cwt_data.shape[3]})")
    
    return cwt_data, merged_labels, merged_subs