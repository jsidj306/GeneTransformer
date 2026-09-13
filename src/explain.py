"""
explain.py -- SHAP + attention visualization

此模块提供模型解释功能，包括SHAP分析和注意力可视化。
"""
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import load_config, get_device, ensure_dir

def shap_analysis(model, background_data, test_data, gene_names=None, save_dir='results/figures'):
    """
    使用SHAP分析模型

    Args:
        model: 训练好的模型
        background_data: 背景数据（用于SHAP explainer）
        test_data: 测试数据
        gene_names: 基因名称列表
        save_dir: 保存目录
    """
    print("开始SHAP分析...")

    device = get_device()
    model.eval()

    # 确保数据在正确的设备上
    background_data = torch.FloatTensor(background_data).to(device)
    test_data = torch.FloatTensor(test_data).to(device)

    try:
        # 创建DeepExplainer
        explainer = shap.DeepExplainer(model, background_data)

        # 计算SHAP值
        shap_values = explainer.shap_values(test_data)

        print("SHAP值计算完成")

        # 绘制summary plot
        ensure_dir(save_dir)

        # 为每个类别生成summary plot
        if isinstance(shap_values, list):  # 多分类
            for i, class_shap_values in enumerate(shap_values):
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    class_shap_values,
                    test_data.cpu().numpy(),
                    feature_names=gene_names,
                    show=False
                )
                plt.title(f'SHAP Summary Plot - Class {i}')
                plt.savefig(f'{save_dir}/shap_summary_class_{i}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"SHAP summary plot已保存: class_{i}")
        else:  # 二分类
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                test_data.cpu().numpy(),
                feature_names=gene_names,
                show=False
            )
            plt.title('SHAP Summary Plot')
            plt.savefig(f'{save_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("SHAP summary plot已保存")

        return shap_values

    except Exception as e:
        print(f"SHAP分析失败: {e}")
        print("尝试使用简化的解释方法...")

        # 简化的梯度分析
        return gradient_based_importance(model, test_data, gene_names, save_dir)

def gradient_based_importance(model, test_data, gene_names=None, save_dir='results/figures'):
    """
    基于梯度的特征重要性分析（SHAP备选方案）

    Args:
        model: 训练好的模型
        test_data: 测试数据
        gene_names: 基因名称列表
        save_dir: 保存目录

    Returns:
        重要性分数
    """
    print("使用基于梯度的方法计算特征重要性...")

    device = get_device()
    model.eval()

    test_data = torch.FloatTensor(test_data).to(device)
    test_data.requires_grad = True

    # 获取模型输出
    outputs = model(test_data)

    # 计算重要性（梯度）
    importance = torch.zeros_like(test_data)

    for class_idx in range(outputs.shape[1]):
        # 对每个类别计算梯度
        class_score = outputs[:, class_idx].sum()
        class_score.backward(retain_graph=True)

        if test_data.grad is not None:
            importance += torch.abs(test_data.grad)

        # 清零梯度
        model.zero_grad()

    importance = importance.cpu().detach().numpy()

    # 平均重要性
    avg_importance = importance.mean(axis=0)

    # 可视化
    if gene_names is not None:
        plt.figure(figsize=(12, 6))
        sorted_idx = np.argsort(avg_importance)[-20:]  # Top 20

        plt.barh(range(len(sorted_idx)), avg_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [gene_names[i] for i in sorted_idx])
        plt.xlabel('Importance Score')
        plt.title('Top 20 Important Genes (Gradient-based)')
        plt.tight_layout()

        save_path = f'{save_dir}/gradient_importance.png'
        ensure_dir(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征重要性图已保存: {save_path}")

    return avg_importance

def explain_single_sample(model, sample, gene_names=None, save_dir='results/figures'):
    """
    解释单个样本的预测

    Args:
        model: 训练好的模型
        sample: 单个样本
        gene_names: 基因名称列表
        save_dir: 保存目录
    """
    print("生成单样本解释...")

    device = get_device()
    model.eval()

    sample = torch.FloatTensor(sample).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = model(sample)
        predicted_class = prediction.argmax(dim=1).item()
        probabilities = torch.softmax(prediction, dim=1)[0]

    print(f"预测类别: {predicted_class}")
    print(f"类别概率: {probabilities.cpu().numpy()}")

    # 基于梯度的单样本解释
    sample.requires_grad = True
    outputs = model(sample)

    target_class = outputs.argmax(dim=1).item()
    target_score = outputs[0, target_class]

    target_score.backward()

    if sample.grad is not None:
        feature_importance = torch.abs(sample.grad[0]).cpu().detach().numpy()

        # 绘制特征重要性
        plt.figure(figsize=(12, 6))
        sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20

        if gene_names is not None:
            labels = [gene_names[i] for i in sorted_idx]
        else:
            labels = [f"Gene_{i}" for i in sorted_idx]

        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), labels)
        plt.xlabel('Contribution to Prediction')
        plt.title(f'Single Sample Explanation - Predicted Class: {target_class}')
        plt.tight_layout()

        save_path = f'{save_dir}/single_sample_explanation.png'
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"单样本解释图已保存: {save_path}")

    return predicted_class, probabilities.cpu().numpy()

def visualize_attention_weights(model, sample, gene_names=None, save_dir='results/figures'):
    """
    可视化注意力权重

    Args:
        model: 训练好的模型
        sample: 输入样本
        gene_names: 基因名称列表
        save_dir: 保存目录
    """
    print("生成注意力权重可视化...")

    # 注意：标准TransformerEncoder不直接返回注意力权重
    # 这里提供一个框架，实际实现需要自定义Encoder层

    print("注意：注意力权重可视化需要自定义Transformer编码器层")
    print("建议使用模型内部的注意力机制或使用SHAP作为替代解释方法")

    # 备选：显示哪些基因对预测最重要
    explain_single_sample(model, sample, gene_names, save_dir)

def generate_interpretation_report(model, test_data, gene_names=None, save_dir='results'):
    """
    生成完整的解释报告

    Args:
        model: 训练好的模型
        test_data: 测试数据
        gene_names: 基因名称列表
        save_dir: 保存目录
    """
    print("生成解释报告...")

    device = get_device()
    model.eval()

    # 背景数据（使用测试数据的一个子集）
    n_background = min(100, len(test_data))
    background_data = test_data[:n_background]

    # 分析数据
    n_samples_to_explain = min(20, len(test_data))
    samples_to_explain = test_data[n_background:n_background + n_samples_to_explain]

    # SHAP分析
    shap_values = shap_analysis(
        model, background_data, samples_to_explain,
        gene_names=gene_names, save_dir=f'{save_dir}/figures'
    )

    # 单样本解释
    explain_single_sample(
        model, test_data[n_background],
        gene_names=gene_names, save_dir=f'{save_dir}/figures'
    )

    # 注意力可视化
    visualize_attention_weights(
        model, test_data[n_background],
        gene_names=gene_names, save_dir=f'{save_dir}/figures'
    )

    print("解释报告生成完成")

if __name__ == "__main__":
    print("explain模块加载成功")
    print("注意：实际解释需要训练好的模型和测试数据")