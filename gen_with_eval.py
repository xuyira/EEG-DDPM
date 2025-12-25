import torch
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from utils.gen import generate_eeg_for_label
from config import TrainingConfig
from model import ClassConditionedUnet
from diffusers import DDPMScheduler

# 添加 EEG-Conformer 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "EEG-Conformer"))
from evaluate_npy import evaluate_npy_file

# 解析命令行参数
parser = argparse.ArgumentParser(
    description='生成指定类别的 EEG 数据并使用 Conformer 模型评估',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例用法:
  # 生成类别 2 的 100 条数据，使用 sub1 的模型评估
  python gen_with_eval.py --target_label 2 --n_samples 100 --nsub 1
  
  # 生成类别 3 的 200 条数据，使用自定义模型路径
  python gen_with_eval.py --target_label 3 --n_samples 200 \\
                          --conformer_model_path EEG-Conformer/best_model_subject1.pth \\
                          --nsub 1
    """
)
parser.add_argument('--target_label', type=int, default=3,
                    help='想要生成的类别标签 (0-3)，默认: 3')
parser.add_argument('--n_samples', type=int, default=100,
                    help='想生成多少条数据，默认: 100')
parser.add_argument('--conformer_model_path', type=str, 
                    default='EEG-Conformer/best_model_subject1.pth',
                    help='Conformer 模型权重路径，默认: EEG-Conformer/best_model_subject1.pth')
parser.add_argument('--nsub', type=int, default=1,
                    help='测试受试者编号（LOSO 方式，用于归一化），默认: 1')

args = parser.parse_args()

# 创建配置
config = TrainingConfig()

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 模型参数（需要与训练时一致）
num_classes = 4
class_emb_size = 4

# 创建模型
model = ClassConditionedUnet(config, num_classes=num_classes, class_emb_size=class_emb_size)
model = model.to(device)

# 加载训练好的模型权重
# 请根据实际情况修改 checkpoint 路径
checkpoint_path = os.path.join(config.output_dir, "checkpoint_epoch_59.pt")  # 修改为你的 checkpoint 文件名

if not os.path.exists(checkpoint_path):
    # 尝试查找最新的 checkpoint
    checkpoint_dir = Path(config.output_dir)
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
        print(f"找到最新的 checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"未找到 checkpoint 文件，请检查路径: {config.output_dir}")

print(f"加载模型权重: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("模型加载完成！")

# 创建噪声调度器（需要与训练时一致）
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# 创建保存目录
save_dir = Path("./model_gen/DDPM_oricond")
save_dir.mkdir(parents=True, exist_ok=True)

# 从命令行参数获取生成参数
target_label = args.target_label
n_samples = args.n_samples
npy_path = str(save_dir / f"eeg_label{target_label}.npy")

# 验证参数
if target_label < 0 or target_label > 3:
    raise ValueError(f"target_label 必须在 0-3 之间，得到: {target_label}")
if n_samples <= 0:
    raise ValueError(f"n_samples 必须大于 0，得到: {n_samples}")

print(f"\n生成参数:")
print(f"  目标类别: {target_label}")
print(f"  生成样本数: {n_samples}")

# 生成指定类别的 EEG 数据
print(f"\n{'='*60}")
print(f"生成类别 {target_label} 的 EEG 数据")
print(f"{'='*60}")
generate_eeg_for_label(
    config,
    model,
    noise_scheduler,
    target_label=target_label,
    n_samples=n_samples,
    save_path=npy_path
)

# 使用 Conformer 模型评估生成的数据
print(f"\n{'='*60}")
print(f"使用 Conformer 模型评估生成的数据")
print(f"{'='*60}")

# 从命令行参数获取评估参数
conformer_model_path = args.conformer_model_path
nsub = args.nsub

print(f"\n评估参数:")
print(f"  Conformer 模型路径: {conformer_model_path}")
print(f"  测试受试者编号: {nsub}")

if not os.path.exists(conformer_model_path):
    print(f"警告: Conformer 模型文件不存在: {conformer_model_path}")
    print("跳过评估步骤")
else:
    # 评估生成的数据
    accuracy, predictions, probabilities, true_labels = evaluate_npy_file(
        npy_path=npy_path,
        model_path=conformer_model_path,
        target_label=target_label,
        nsub=nsub
    )
    
    # 创建 DataFrame
    results_df = pd.DataFrame({
        '真实标签': true_labels,
        '预测标签': predictions,
        '预测为类0的概率': probabilities[:, 0],
        '预测为类1的概率': probabilities[:, 1],
        '预测为类2的概率': probabilities[:, 2],
        '预测为类3的概率': probabilities[:, 3]
    })
    
    # 保存为 Excel 文件
    excel_path = str(save_dir / f"evaluation_results_label{target_label}.xlsx")
    results_df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"\n评估结果已保存到: {excel_path}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
