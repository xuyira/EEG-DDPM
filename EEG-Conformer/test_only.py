"""
只进行测试的脚本：加载已训练好的模型，对测试数据进行预测并保存 Excel
不需要重新训练
"""
import os
import numpy as np
import torch
import scipy.io
from conformer_multisub import Conformer, ExP


def test_only(nsub=1, model_path=None):
    """
    只进行测试，不训练
    
    参数:
        nsub: 测试受试者编号（留哪个受试者作为测试集）
        model_path: 模型权重路径，如果为 None，则使用默认路径
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建实验对象（用于加载数据）
    exp = ExP(nsub=nsub, total_sub=9)
    
    # 加载测试数据
    print(f"\n加载测试数据（留 sub{nsub}，使用 sub{','.join([str(i) for i in range(1, 10) if i != nsub])} 的 E 数据，每类10%）...")
    _, _, test_data, test_label = exp.get_source_data()
    
    # 转换为 tensor
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label - 1)  # 转换为 0-3
    
    # 转换为 Variable（与训练时一致）
    test_data = test_data.type(exp.Tensor)
    test_label = test_label.type(exp.LongTensor)
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试标签形状: {test_label.shape}")
    
    # 加载模型
    if model_path is None:
        model_path = f'best_model_subject{nsub}.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"\n加载模型: {model_path}")
    model = Conformer(emb_size=40, depth=6, n_classes=4).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.module.load_state_dict(state_dict)
    model.eval()
    print("模型加载成功！")
    
    # 进行预测
    print(f"\n开始对测试数据进行预测...")
    with torch.no_grad():
        _, test_outputs = model(test_data)
        # 计算 softmax 概率
        test_probabilities = torch.softmax(test_outputs, dim=1)
        test_predictions = torch.max(test_outputs, 1)[1]
    
    # 计算准确率
    correct = (test_predictions == test_label).sum().item()
    total = test_label.size(0)
    accuracy = correct / total
    
    print(f"\n预测结果:")
    print(f"  总样本数: {total}")
    print(f"  正确预测: {correct}")
    print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 打印每个类别的预测分布
    unique_preds, counts = torch.unique(test_predictions, return_counts=True)
    print(f"\n预测类别分布:")
    for pred_class, count in zip(unique_preds.cpu().numpy(), counts.cpu().numpy()):
        print(f"  类别 {pred_class}: {count} 个样本 ({count/total*100:.2f}%)")
    
    # 打印每个类别的真实分布
    unique_true, counts_true = torch.unique(test_label, return_counts=True)
    print(f"\n真实类别分布:")
    for true_class, count in zip(unique_true.cpu().numpy(), counts_true.cpu().numpy()):
        print(f"  类别 {true_class}: {count} 个样本 ({count/total*100:.2f}%)")
    
    # 保存 Excel 文件
    try:
        import pandas as pd
        
        results_df = pd.DataFrame({
            '真实标签': test_label.cpu().numpy(),
            '预测标签': test_predictions.cpu().numpy(),
            '预测为类0的概率': test_probabilities[:, 0].cpu().numpy(),
            '预测为类1的概率': test_probabilities[:, 1].cpu().numpy(),
            '预测为类2的概率': test_probabilities[:, 2].cpu().numpy(),
            '预测为类3的概率': test_probabilities[:, 3].cpu().numpy()
        })
        
        # 创建 results 目录（如果不存在）
        os.makedirs("./results", exist_ok=True)
        excel_path = f'./results/test_results_subject{nsub}.xlsx'
        results_df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f'\n测试结果已保存到: {excel_path}')
        print(f'测试样本数: {len(results_df)}')
        
    except ImportError:
        print('警告: pandas 或 openpyxl 未安装，无法保存 Excel 文件')
        print('请运行: pip install pandas openpyxl')
    
    return accuracy, test_predictions.cpu().numpy(), test_probabilities.cpu().numpy(), test_label.cpu().numpy()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='只进行测试，不训练（加载已训练好的模型）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认模型路径测试 sub1
  python test_only.py --nsub 1
  
  # 使用自定义模型路径
  python test_only.py --nsub 1 --model_path best_model_subject1.pth
        """
    )
    parser.add_argument('--nsub', type=int, default=1,
                        help='测试受试者编号（留哪个受试者作为测试集）')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重路径（如果为 None，使用 best_model_subject{nsub}.pth）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"只进行测试（不训练）")
    print(f"留 sub{args.nsub} 作为测试集")
    print("=" * 60)
    
    accuracy, predictions, probabilities, true_labels = test_only(
        nsub=args.nsub,
        model_path=args.model_path
    )
    
    print(f"\n" + "=" * 60)
    print(f"最终测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"=" * 60)

