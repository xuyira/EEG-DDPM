"""
评估脚本：使用训练好的模型对指定的 npy 文件进行预测

单受试者模式：
- 如果模型是 sub1 训练的，归一化使用 sub1 的训练数据（T 数据）
- 如果模型是 sub2 训练的，归一化使用 sub2 的训练数据（T 数据）
- 以此类推
"""
import os
import numpy as np
import torch
import scipy.io
from conformer_onesub import Conformer, ExP


def evaluate_npy_file(npy_path, model_path, target_label=2, nsub=1, data_root=None):
    """
    使用训练好的单受试者模型评估指定的 npy 文件
    
    参数:
        npy_path: npy 文件路径，格式为 (N, 1000, 22)
        model_path: 训练好的模型权重路径（单受试者模型，sub nsub 训练的）
        target_label: 真实标签（用于计算准确率）
        nsub: 受试者编号（模型是哪个受试者训练的）
              例如：nsub=1 表示模型是 sub1 训练的，归一化使用 sub1 的训练数据（T 数据）
        data_root: 数据根目录，如果为 None，使用 ExP 的默认路径
    
    返回:
        accuracy: 准确率
        predictions: 预测结果
        probabilities: 预测概率
        true_labels: 真实标签
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载训练数据用于归一化（单受试者方式：使用 sub nsub 自己的训练数据 T）
    print(f"\n加载单受试者训练数据用于归一化（sub{nsub} 的 T 数据）...")
    
    # 如果提供了 data_root，临时修改 ExP 的 root
    if data_root is None:
        exp = ExP(nsub)
    else:
        # 创建临时 ExP 实例并修改 root
        exp = ExP(nsub)
        exp.root = data_root
    
    train_data, _, _, _ = exp.get_source_data()
    
    # 获取训练数据的均值和标准差（用于归一化测试数据）
    target_mean = np.mean(train_data)
    target_std = np.std(train_data)
    print(f"训练数据统计:")
    print(f"  均值: {target_mean:.6f}")
    print(f"  标准差: {target_std:.6f}")
    print(f"  数据来源: sub{nsub} 的训练数据（T 数据）")
    
    # 2. 加载测试数据（npy 文件）
    print(f"\n加载测试数据: {npy_path}")
    test_data = np.load(npy_path)
    print(f"原始数据形状: {test_data.shape}")
    
    # 检查数据格式并转换
    if test_data.ndim == 3:
        # (N, 1000, 22) -> (N, 1, 22, 1000)
        test_data = np.transpose(test_data, (0, 2, 1))
        test_data = np.expand_dims(test_data, axis=1)
    elif test_data.ndim == 4:
        # 已经是 (N, 1, 22, 1000) 格式
        if test_data.shape[1] != 1:
            raise ValueError(f"期望第2维为1，得到 {test_data.shape[1]}")
    else:
        raise ValueError(f"不支持的数据格式，期望 3D 或 4D，得到 {test_data.ndim}D")
    
    print(f"转换后数据形状: {test_data.shape}")
    
    # 归一化（使用训练数据的统计量）
    print(f"\n使用训练数据统计量进行归一化...")
    test_data = (test_data - target_mean) / target_std
    
    # 转换为 tensor
    test_data = torch.from_numpy(test_data).float().to(device)
    
    # 创建真实标签（所有样本都是 target_label）
    true_labels = torch.full((test_data.shape[0],), target_label, dtype=torch.long).to(device)
    print(f"真实标签: {target_label}, 样本数: {test_data.shape[0]}")
    
    # 3. 加载模型
    print(f"\n加载模型: {model_path}")
    model = Conformer(emb_size=40, depth=6, n_classes=4).to(device)
    
    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    # 处理 DataParallel 保存的权重（如果有 'module.' 前缀）
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    print("模型加载成功！")
    
    model.eval()
    
    # 4. 进行预测
    print("\n开始预测...")
    with torch.no_grad():
        _, outputs = model(test_data)
        # 计算 softmax 概率
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.max(outputs, 1)[1]
    
    # 5. 计算准确率
    correct = (predictions == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total
    
    print(f"\n" + "=" * 60)
    print(f"预测结果")
    print(f"=" * 60)
    print(f"总样本数: {total}")
    print(f"正确预测: {correct}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 打印每个类别的预测分布
    unique_preds, counts = torch.unique(predictions, return_counts=True)
    print(f"\n预测类别分布:")
    for pred_class, count in zip(unique_preds.cpu().numpy(), counts.cpu().numpy()):
        print(f"  类别 {pred_class}: {count} 个样本 ({count/total*100:.2f}%)")
    
    return accuracy, predictions.cpu().numpy(), probabilities.cpu().numpy(), true_labels.cpu().numpy()


def train_and_evaluate(npy_path, target_label=2, nsub=1, data_root=None):
    """
    训练指定受试者的单受试者模型，然后评估指定的 npy 文件
    
    参数:
        npy_path: npy 文件路径
        target_label: 真实标签
        nsub: 受试者编号（训练哪个受试者的模型）
        data_root: 数据根目录
    """
    print("=" * 60)
    print(f"训练单受试者模型（sub{nsub}：T 数据训练，E 数据测试）")
    print("=" * 60)
    
    # 设置随机种子
    seed_n = 42  # 固定种子以便复现
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    
    # 训练指定受试者的模型
    exp = ExP(nsub)
    if data_root is not None:
        exp.root = data_root
    bestAcc, averAcc, Y_true, Y_pred = exp.train()
    
    print(f"\n训练完成！最佳准确率: {bestAcc:.4f}")
    
    # 评估 npy 文件
    print("\n" + "=" * 60)
    print("评估指定的 npy 文件")
    print("=" * 60)
    
    model_path = f'best_model_subject{nsub}.pth'
    
    accuracy, predictions, probabilities, true_labels = evaluate_npy_file(
        npy_path=npy_path,
        model_path=model_path,
        target_label=target_label,
        nsub=nsub,
        data_root=data_root
    )
    
    return accuracy, predictions, probabilities, true_labels


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='评估 npy 文件（单受试者模型）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用已训练的 sub1 模型评估 eeg_label2.npy
  python evaluate_npy_onesub.py --npy_path ../model_gen/DDPM_oricond/eeg_label2.npy \\
                                 --model_path best_model_subject1.pth \\
                                 --target_label 2 \\
                                 --nsub 1
  
  # 先训练 sub1 模型，再评估
  python evaluate_npy_onesub.py --npy_path ../model_gen/DDPM_oricond/eeg_label2.npy \\
                                 --target_label 2 \\
                                 --nsub 1 \\
                                 --train
  
  # 指定数据根目录
  python evaluate_npy_onesub.py --npy_path ../model_gen/DDPM_oricond/eeg_label2.npy \\
                                 --model_path best_model_subject1.pth \\
                                 --target_label 2 \\
                                 --nsub 1 \\
                                 --data_root /Data/strict_TE/
        """
    )
    parser.add_argument('--npy_path', type=str, default='../model_gen/DDPM_oricond/eeg_label2.npy',
                        help='npy 文件路径，格式为 (N, 1000, 22)')
    parser.add_argument('--model_path', type=str, default='best_model_subject1.pth',
                        help='模型权重路径（单受试者模型）')
    parser.add_argument('--target_label', type=int, default=2,
                        help='真实标签（用于计算准确率）')
    parser.add_argument('--nsub', type=int, default=1,
                        help='受试者编号（模型是哪个受试者训练的，归一化使用该受试者的训练数据 T）')
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据根目录（如果为 None，使用 ExP 的默认路径）')
    parser.add_argument('--train', action='store_true',
                        help='是否先训练模型（如果模型不存在）')
    
    args = parser.parse_args()
    
    # 如果指定了 --train 或模型不存在，先训练
    if args.train or not os.path.exists(args.model_path):
        if args.train:
            print("指定了 --train，将先训练模型...")
        else:
            print(f"模型文件不存在: {args.model_path}，将先训练模型...")
        accuracy, predictions, probabilities, true_labels = train_and_evaluate(
            npy_path=args.npy_path,
            target_label=args.target_label,
            nsub=args.nsub,
            data_root=args.data_root
        )
    else:
        # 直接评估
        print("使用已训练的模型进行评估...")
        accuracy, predictions, probabilities, true_labels = evaluate_npy_file(
            npy_path=args.npy_path,
            model_path=args.model_path,
            target_label=args.target_label,
            nsub=args.nsub,
            data_root=args.data_root
        )
    
    print(f"\n" + "=" * 60)
    print(f"最终准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"=" * 60)
