"""
utils.py -- seed, metrics, plotting helpers

此模块提供通用工具函数，包括随机种子设置、评估指标计算和绘图功能。
"""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

def set_seed(seed=42):
    """
    设置全局随机种子以确保可复现性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path='config.yaml'):
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path='config.yaml'):
    """
    保存配置文件

    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def calculate_metrics(y_true, y_pred, y_prob=None, average='macro'):
    """
    计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率 (可选)
        average: 平均方式 (macro/micro/weighted)

    Returns:
        指标字典
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, classification_report
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # 如果有概率预测且是多分类，计算AUC
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2:
                # 多分类需要one-vs-rest
                metrics['auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average=average
                )
            else:
                # 二分类
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception as e:
            print(f"AUC计算失败: {e}")
            metrics['auc'] = None

    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")

    plt.close()

def plot_training_history(train_losses, val_losses, val_metrics=None,
                         metric_name='AUC', save_path=None):
    """
    绘制训练历史

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_metrics: 验证指标列表 (可选)
        metric_name: 指标名称
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 指标曲线
    if val_metrics:
        ax2.plot(epochs, val_metrics, 'g-', label=f'Val {metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'Validation {metric_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史已保存到: {save_path}")

    plt.close()

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_device():
    """
    获取可用的计算设备

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device

def format_time(seconds):
    """
    格式化时间显示

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

if __name__ == "__main__":
    print("utils模块加载成功")

    # 测试种子设置
    set_seed(42)
    print(f"随机种子已设置: 42")

    # 测试配置加载
    try:
        config = load_config()
        print(f"配置文件加载成功，模型类别数: {config['model']['n_classes']}")
    except Exception as e:
        print(f"配置文件加载失败: {e}")