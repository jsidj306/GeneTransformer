"""
dataset.py -- PyTorch Dataset / DataLoader

此模块将预处理后的数据封装为PyTorch Dataset和DataLoader。
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class GeneDataset(Dataset):
    """
    基因表达数据集

    Args:
        X: 特征矩阵 (样本 × 基因)，可以是DataFrame或ndarray
        y: 标签向量，可以是Series或ndarray
    """
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.X = torch.tensor(X.values, dtype=torch.float32)
            self.gene_names = X.columns.tolist()
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.gene_names = None

        if isinstance(y, pd.Series):
            self.y = torch.tensor(y.values, dtype=torch.long)
            self.sample_names = y.index.tolist()
        else:
            self.y = torch.tensor(y, dtype=torch.long)
            self.sample_names = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32,
                       num_workers=0, shuffle_train=True):
    """
    创建训练和测试DataLoader

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        batch_size: 批量大小
        num_workers: 数据加载工作进程数
        shuffle_train: 是否打乱训练数据

    Returns:
        train_loader, test_loader
    """
    train_dataset = GeneDataset(X_train, y_train)
    test_dataset = GeneDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=False  # CPU环境下不使用pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, test_loader

def get_class_weights(y):
    """
    计算类别权重（用于处理类别不平衡）

    Args:
        y: 标签向量

    Returns:
        类别权重张量
    """
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    # 反向频率权重
    weights = total / (len(unique) * counts)

    class_weights = {}
    for cls, weight in zip(unique, weights):
        class_weights[cls] = weight

    return class_weights

if __name__ == "__main__":
    # 测试代码
    print("dataset模块加载成功")

    # 创建测试数据
    X_train = np.random.randn(100, 5000)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.randn(20, 5000)
    y_test = np.random.randint(0, 3, 20)

    # 创建DataLoader
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=8
    )

    # 测试加载
    for batch_X, batch_y in train_loader:
        print(f"Batch shape: X={batch_X.shape}, y={batch_y.shape}")
        break