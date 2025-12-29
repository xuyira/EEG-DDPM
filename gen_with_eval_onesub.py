import torch
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import TrainingConfig
from model.unet2dcond import create_unet2dcond_model
from diffusers import DDPMScheduler

def generate_eeg_for_label(config, unet, noise_scheduler, target_label, n_samples, save_path, gen_batch_size=None):
    """
    生成指定类别的 EEG 数据
    
    参数:
        config: TrainingConfig 配置对象
        unet: UNet 模型
        noise_scheduler: 噪声调度器
        target_label: 目标类别标签
        n_samples: 要生成的样本总数
        save_path: 保存路径
        gen_batch_size: 生成时的 batch size（如果为 None，则一次性生成所有样本）
    """
    device = next(unet.parameters()).device
    
    # 如果没有指定 gen_batch_size，则一次性生成所有样本
    if gen_batch_size is None:
        gen_batch_size = n_samples
    
    # 确保 gen_batch_size 不超过 n_samples
    gen_batch_size = min(gen_batch_size, n_samples)
    
    print(f"生成配置: 总样本数={n_samples}, batch_size={gen_batch_size}")

    # 计算需要多少个 batch
    n_batches = (n_samples + gen_batch_size - 1) // gen_batch_size
    
    all_samples = []
    
    unet.eval()
    with torch.no_grad():
        # 分批生成
        for batch_idx in range(n_batches):
            # 计算当前 batch 的大小
            start_idx = batch_idx * gen_batch_size
            end_idx = min(start_idx + gen_batch_size, n_samples)
            current_batch_size = end_idx - start_idx
            
            print(f"\n生成 batch {batch_idx + 1}/{n_batches} (样本 {start_idx + 1}-{end_idx})...")
            
            # 1. 构造标签
            labels = torch.full((current_batch_size,), int(target_label), dtype=torch.long, device=device)

            # 2. 初始化噪声
            shape = (current_batch_size, config.unet_in_channels, *config.image_size)
            sample = torch.randn(shape, device=device)

            # 3. 走反向扩散（使用类别嵌入方式）
            timesteps = noise_scheduler.timesteps.to(device)
            # 显示扩散过程的进度条
            for t in tqdm(timesteps, desc=f"  扩散去噪 (batch {batch_idx + 1}/{n_batches})", unit="step", leave=False):
                timestep_tensor = torch.full((current_batch_size,), t, dtype=torch.long, device=device)
                noise_pred = unet(sample, timestep_tensor, class_labels=labels, encoder_hidden_states=None).sample
                sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
            
            # 保存当前 batch 的结果
            all_samples.append(sample.cpu())
    
    # 合并所有 batch 的结果
    print(f"\n合并 {n_batches} 个 batch 的结果...")
    sample = torch.cat(all_samples, dim=0)  # (n_samples, C, H, W)

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
  
  # 指定 checkpoint 文件，使用 batch_size=16 分批生成（节省显存）
  python gen_with_eval_onesub.py --target_label 2 --n_samples 100 --nsub 1 \\
                                  --checkpoint model_checkpoints/DDPM_oricond_3/sub1/checkpoint_epoch_39.pt \\
                                  --gen_batch_size 16
  
  # 生成类别 3 的 200 条数据，使用自定义模型路径和 checkpoint，分批生成
  python gen_with_eval_onesub.py --target_label 3 --n_samples 200 \\
                                  --conformer_model_path EEG-Conformer/best_model_subject1.pth \\
                                  --nsub 1 \\
                                  --checkpoint model_checkpoints/DDPM_oricond_3/sub1/checkpoint_epoch_59.pt \\
                                  --gen_batch_size 32
    """
)
parser.add_argument('--target_label', type=int, default=3,
                    help='想要生成的类别标签 (0-3)，默认: 3')
parser.add_argument('--n_samples', type=int, default=100,
                    help='想生成多少条数据，默认: 100')
parser.add_argument('--gen_batch_size', type=int, default=None,
                    help='生成时的 batch size（用于节省显存）。如果为 None，则一次性生成所有样本。默认: None')
parser.add_argument('--conformer_model_path', type=str, 
                    default='EEG-Conformer/best_model_subject1.pth',
                    help='Conformer 模型权重路径，默认: EEG-Conformer/best_model_subject1.pth')
parser.add_argument('--nsub', type=int, default=1,
                    help='受试者编号（单受试者模式，使用该受试者的训练数据 T 进行归一化），默认: 1')
parser.add_argument('--data_root', type=str, default='EEG-Conformer/data/standard_2a_data/',
                    help='数据根目录，默认: EEG-Conformer/data/standard_2a_data/')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='指定模型路径（相对或绝对路径）。可以是：1) 使用 save_pretrained 保存的模型目录（包含 unet 子目录），2) 旧的 checkpoint.pt 文件路径。如果不指定，将尝试从 config.output_dir/unet 加载')

args = parser.parse_args()

# 创建配置
config = TrainingConfig()

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 模型参数（需要与训练时一致）
num_classes = 4
config.num_class_embeds = num_classes

# 确定模型路径
if args.checkpoint:
    # 如果用户指定了模型路径
    model_path = args.checkpoint
    if not os.path.isabs(model_path):
        # 如果是相对路径，相对于项目根目录
        model_path = os.path.join(Path(__file__).parent, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"指定的模型路径不存在: {model_path}")
    print(f"使用指定的模型路径: {model_path}")
else:
    # 自动查找模型
    # 在 main_unet2dcond.py 中，模型使用 save_pretrained 保存到 config.output_dir/unet 目录
    # config.output_dir 在 __post_init__ 中会被修改为 ./model_checkpoints/DDPM_attencond/sub{leave_sub}
    # 所以实际保存路径是 ./model_checkpoints/DDPM_attencond/sub1/unet/（假设 leave_sub=1）
    default_model_dir = Path(config.output_dir) / "unet"
    
    print(f"查找模型目录: {default_model_dir}")
    if default_model_dir.exists() and (default_model_dir / "config.json").exists():
        # 找到 save_pretrained 格式的模型（HuggingFace 格式，包含 config.json 和 diffusion_pytorch_model.bin）
        model_path = str(default_model_dir)
        print(f"✓ 找到 save_pretrained 格式模型: {model_path}")
    else:
        # 如果不存在，尝试查找旧的 checkpoint 文件（.pt 格式）
        checkpoint_dir = Path(config.output_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            model_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
            print(f"⚠ 未找到 save_pretrained 模型，使用旧的 checkpoint 格式: {model_path}")
        else:
            raise FileNotFoundError(
                f"未找到模型文件。\n"
                f"  期望路径: {default_model_dir}\n"
                f"  或 checkpoint 文件: {checkpoint_dir}/checkpoint_epoch_*.pt\n"
                f"  请使用 --checkpoint 参数指定模型路径"
            )

# 加载模型
print(f"加载模型: {model_path}")
if model_path.endswith('.pt'):
    # 旧的 checkpoint 格式：需要先创建模型结构，再加载权重
    print("检测到旧格式 checkpoint，创建模型结构...")
    unet = create_unet2dcond_model(config, num_classes=num_classes)
    unet = unet.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 尝试加载，允许部分键不匹配（因为可能是旧格式）
    missing_keys, unexpected_keys = unet.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"警告: 以下键未加载: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"警告: 以下键未加载: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 以下键未使用: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"警告: 以下键未使用: {unexpected_keys}")
else:
    # 新的 save_pretrained 格式：直接加载
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(model_path)
    unet = unet.to(device)

unet.eval()
print("模型加载完成！")

# 创建噪声调度器（需要与训练时一致）
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# 创建保存目录
save_dir = Path("./model_gen/DDPM_oricond")
save_dir.mkdir(parents=True, exist_ok=True)

# 从命令行参数获取生成参数
target_label = args.target_label
n_samples = args.n_samples
gen_batch_size = args.gen_batch_size
npy_path = str(save_dir / f"eeg_label{target_label}.npy")

# 验证参数
if target_label < 0 or target_label > 3:
    raise ValueError(f"target_label 必须在 0-3 之间，得到: {target_label}")
if n_samples <= 0:
    raise ValueError(f"n_samples 必须大于 0，得到: {n_samples}")
if gen_batch_size is not None and gen_batch_size <= 0:
    raise ValueError(f"gen_batch_size 必须大于 0，得到: {gen_batch_size}")

print(f"\n生成参数:")
print(f"  目标类别: {target_label}")
print(f"  生成样本数: {n_samples}")
if gen_batch_size is not None:
    print(f"  生成 batch size: {gen_batch_size}")
else:
    print(f"  生成 batch size: 一次性生成所有样本 ({n_samples})")

# 生成指定类别的 EEG 数据
print(f"\n{'='*60}")
print(f"生成类别 {target_label} 的 EEG 数据")
print(f"{'='*60}")
generate_eeg_for_label(
    config,
    unet,
    noise_scheduler,
    target_label=target_label,
    n_samples=n_samples,
    save_path=npy_path,
    gen_batch_size=gen_batch_size
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
