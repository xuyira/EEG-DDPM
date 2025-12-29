"""
检查 LabelEmbedder 的类别区分效果

1. LabelEmbedder 是否在训练？
2. 不同类别的 embedding 是否有差异？
3. 如何判断类别区分效果？
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def check_label_embedder_training(model, optimizer):
    """
    检查 LabelEmbedder 是否在优化器中（即是否参与训练）
    """
    print("=" * 80)
    print("检查 LabelEmbedder 是否参与训练")
    print("=" * 80)
    
    # 获取优化器中的所有参数组
    all_param_names = set()
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            # 获取参数的名字（通过查找在哪个模块中）
            for name, module_param in model.named_parameters():
                if param is module_param:
                    all_param_names.add(name)
                    break
    
    # 检查 label_embedder 的参数
    label_embedder_params = []
    for name in all_param_names:
        if 'label_embedder' in name.lower() or 'embed' in name.lower():
            label_embedder_params.append(name)
    
    if label_embedder_params:
        print(f"\n✓ LabelEmbedder 的参数在优化器中（参与训练）:")
        for name in sorted(label_embedder_params):
            print(f"  - {name}")
    else:
        print(f"\n✗ 未找到 LabelEmbedder 的参数！可能没有参与训练。")
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    label_embedder_params_count = 0
    for name, param in model.named_parameters():
        if 'label_embedder' in name.lower() or 'embed' in name.lower():
            label_embedder_params_count += param.numel()
            print(f"  {name}: {param.shape} ({param.numel()} 个参数)")
    
    print(f"\n参数统计:")
    print(f"  LabelEmbedder 参数数量: {label_embedder_params_count:,}")
    print(f"  模型总参数数量: {total_params:,}")
    print(f"  LabelEmbedder 占比: {label_embedder_params_count/total_params*100:.4f}%")
    
    return label_embedder_params_count > 0


def check_embedding_differences(label_embedder, num_classes, device='cpu'):
    """
    检查不同类别的 embedding 是否有差异
    
    参数:
        label_embedder: LabelEmbedder 模型
        num_classes: 类别数量
        device: 设备
    """
    print("\n" + "=" * 80)
    print("检查不同类别的 Embedding 差异")
    print("=" * 80)
    
    label_embedder.eval()
    with torch.no_grad():
        # 获取所有类别的 embedding
        all_labels = torch.arange(num_classes, dtype=torch.long, device=device)
        embeddings = label_embedder(all_labels)  # (num_classes, emb_dim)
        
        print(f"\n1. Embedding 基本信息:")
        print(f"   形状: {embeddings.shape}  # (num_classes={num_classes}, emb_dim={embeddings.shape[1]})")
        print(f"   数据类型: {embeddings.dtype}")
        print(f"   值范围: [{embeddings.min().item():.6f}, {embeddings.max().item():.6f}]")
        print(f"   均值: {embeddings.mean().item():.6f}")
        print(f"   标准差: {embeddings.std().item():.6f}")
        
        print(f"\n2. 每个类别的 Embedding 统计:")
        for i in range(num_classes):
            emb = embeddings[i]
            print(f"   类别 {i}:")
            print(f"     均值: {emb.mean().item():.6f}")
            print(f"     标准差: {emb.std().item():.6f}")
            print(f"     最小值: {emb.min().item():.6f}")
            print(f"     最大值: {emb.max().item():.6f}")
            print(f"     前5个值: {emb[:5].cpu().numpy()}")
        
        print(f"\n3. 类别之间的 Embedding 差异:")
        # 计算类别之间的余弦相似度
        from torch.nn.functional import cosine_similarity
        
        similarities = []
        distances = []
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                emb_i = embeddings[i].unsqueeze(0)  # (1, emb_dim)
                emb_j = embeddings[j].unsqueeze(0)  # (1, emb_dim)
                
                # 余弦相似度
                cos_sim = cosine_similarity(emb_i, emb_j, dim=1).item()
                similarities.append((i, j, cos_sim))
                
                # 欧氏距离
                euclidean_dist = torch.norm(emb_i - emb_j).item()
                distances.append((i, j, euclidean_dist))
        
        print(f"\n   余弦相似度 (越接近1越相似，越接近-1越不同):")
        for i, j, sim in similarities:
            print(f"     类别 {i} vs 类别 {j}: {sim:.6f}")
        
        print(f"\n   欧氏距离 (越大越不同):")
        for i, j, dist in distances:
            print(f"     类别 {i} vs 类别 {j}: {dist:.6f}")
        
        # 计算平均差异
        avg_sim = np.mean([sim for _, _, sim in similarities])
        avg_dist = np.mean([dist for _, _, dist in distances])
        
        print(f"\n   平均余弦相似度: {avg_sim:.6f}")
        print(f"   平均欧氏距离: {avg_dist:.6f}")
        
        # 判断标准
        print(f"\n4. 类别区分效果判断:")
        if avg_sim < 0.5:
            print(f"   ✓ 余弦相似度较低 ({avg_sim:.6f} < 0.5)，类别区分度较好")
        elif avg_sim < 0.8:
            print(f"   ⚠ 余弦相似度中等 ({avg_sim:.6f})，类别区分度一般")
        else:
            print(f"   ✗ 余弦相似度较高 ({avg_sim:.6f} >= 0.8)，类别区分度较差")
        
        if avg_dist > 1.0:
            print(f"   ✓ 欧氏距离较大 ({avg_dist:.6f} > 1.0)，类别区分度较好")
        elif avg_dist > 0.5:
            print(f"   ⚠ 欧氏距离中等 ({avg_dist:.6f})，类别区分度一般")
        else:
            print(f"   ✗ 欧氏距离较小 ({avg_dist:.6f} <= 0.5)，类别区分度较差")
        
        """
        # 可视化
        print(f"\n5. 生成可视化图表...")
        try:
            # 使用 PCA 或 t-SNE 降维可视化（如果维度不太高）
            if embeddings.shape[1] <= 128:
                # 简单的 2D 可视化：选择前两个维度
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # 图1：前两个维度的散点图
                ax1 = axes[0]
                for i in range(num_classes):
                    ax1.scatter(embeddings[i, 0].item(), embeddings[i, 1].item(), 
                              label=f'Class {i}', s=100, alpha=0.7)
                ax1.set_xlabel('Embedding Dimension 0')
                ax1.set_ylabel('Embedding Dimension 1')
                ax1.set_title('Embedding Visualization (First 2 Dimensions)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 图2：所有类别的 embedding 热力图
                ax2 = axes[1]
                im = ax2.imshow(embeddings.cpu().numpy(), aspect='auto', cmap='viridis')
                ax2.set_xlabel('Embedding Dimension')
                ax2.set_ylabel('Class')
                ax2.set_title('Embedding Heatmap')
                ax2.set_yticks(range(num_classes))
                ax2.set_yticklabels([f'Class {i}' for i in range(num_classes)])
                plt.colorbar(im, ax=ax2)
                
                plt.tight_layout()
                save_path = Path("./embedding_visualization.png")
                plt.savefig(save_path, dpi=150)
                plt.close()
                print(f"   ✓ 可视化图表已保存到: {save_path}")
            else:
                print(f"   ⚠ Embedding 维度较高 ({embeddings.shape[1]})，跳过 2D 可视化")
        except Exception as e:
            print(f"   ✗ 可视化失败: {e}")
        """
    label_embedder.train()
    return embeddings


def check_embedding_gradients(model, labels, loss):
    """
    检查 LabelEmbedder 的梯度（用于验证是否在训练）
    """
    print("\n" + "=" * 80)
    print("检查 LabelEmbedder 的梯度")
    print("=" * 80)
    
    has_grad = False
    for name, param in model.named_parameters():
        if 'label_embedder' in name.lower() or 'embed' in name.lower():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                print(f"  {name}:")
                print(f"    梯度范数: {grad_norm:.6f}")
                print(f"    梯度均值: {grad_mean:.6f}")
                print(f"    梯度标准差: {grad_std:.6f}")
            else:
                print(f"  {name}: ✗ 没有梯度（可能未参与反向传播）")
    
    if has_grad:
        print(f"\n✓ LabelEmbedder 有梯度，说明正在训练")
    else:
        print(f"\n✗ LabelEmbedder 没有梯度，可能未参与训练")
    
    return has_grad


if __name__ == "__main__":
    print("这是一个工具函数，用于检查 LabelEmbedder 的训练状态和类别区分效果")
    print("\n使用方法:")
    print("1. 在训练循环中调用 check_label_embedder_training(model, optimizer)")
    print("2. 在训练过程中调用 check_embedding_differences(label_embedder, num_classes, device)")
    print("3. 在 backward 之后调用 check_embedding_gradients(model, labels, loss)")

