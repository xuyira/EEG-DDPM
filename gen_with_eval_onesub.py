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
from evaluate_npy_onesub import evaluate_npy_file

# 解析命令行参数
parser = argparse.ArgumentParser(
    description='生成指定类别的 EEG 数据并使用 Conformer 模型评估',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例用法:
  # 生成类别 2 的 100 条数据，使用 sub1 的模型评估（自动查找 checkpoint）
  python gen_with_eval_onesub.py --target_label 2 --n_samples 100 --nsub 1
  
  # 指定 checkpoint 文件
  python gen_with_eval_onesub.py --target_label 2 --n_samples 100 --nsub 1 \\
                                  --checkpoint model_checkpoints/DDPM_oricond_3/sub1/checkpoint_epoch_39.pt
  
  # 生成类别 3 的 200 条数据，使用自定义模型路径和 checkpoint
  python gen_with_eval_onesub.py --target_label 3 --n_samples 200 \\
                                  --conformer_model_path EEG-Conformer/best_model_subject1.pth \\
                                  --nsub 1 \\
                                  --checkpoint model_checkpoints/DDPM_oricond_3/sub1/checkpoint_epoch_59.pt
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
                    help='受试者编号（单受试者模式，使用该受试者的训练数据 T 进行归一化），默认: 1')
parser.add_argument('--data_root', type=str, default='EEG-Conformer/data/standard_2a_data/',
                    help='数据根目录，默认: EEG-Conformer/data/standard_2a_data/')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='指定 checkpoint 文件路径（相对或绝对路径）。如果不指定，将尝试加载 checkpoint_epoch_59.pt，如果不存在则使用最新的 checkpoint')

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
if args.checkpoint:
    # 如果用户指定了 checkpoint 路径
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        # 如果是相对路径，相对于项目根目录
        checkpoint_path = os.path.join(Path(__file__).parent, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"指定的 checkpoint 文件不存在: {checkpoint_path}")
    print(f"使用指定的 checkpoint: {checkpoint_path}")
else:
    # 自动查找 checkpoint
    default_checkpoint = os.path.join(config.output_dir, "checkpoint_epoch_59.pt")
    
    if os.path.exists(default_checkpoint):
        checkpoint_path = default_checkpoint
        print(f"使用默认 checkpoint: {checkpoint_path}")
    else:
        # 尝试查找最新的 checkpoint
        checkpoint_dir = Path(config.output_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
            print(f"未找到 checkpoint_epoch_59.pt，使用最新的 checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"未找到 checkpoint 文件。请检查路径: {config.output_dir}\n"
                f"或使用 --checkpoint 参数指定 checkpoint 路径"
            )

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
print(f"  受试者编号: {nsub} (单受试者模式，使用 sub{nsub} 的训练数据 T 进行归一化)")
if args.data_root:
    print(f"  数据根目录: {args.data_root}")

if not os.path.exists(conformer_model_path):
    print(f"警告: Conformer 模型文件不存在: {conformer_model_path}")
    print("跳过评估步骤")
else:
    # 评估生成的数据
    # 确保 data_root 路径正确（如果为 None，使用默认路径，并确保以 / 结尾）
    if args.data_root:
        data_root = args.data_root.rstrip('/') + '/'
    else:
        data_root = 'EEG-Conformer/data/standard_2a_data/'
    
    accuracy, predictions, probabilities, true_labels = evaluate_npy_file(
        npy_path=npy_path,
        model_path=conformer_model_path,
        target_label=target_label,
        nsub=nsub,
        data_root=data_root
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
