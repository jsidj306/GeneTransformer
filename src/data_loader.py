"""
data_loader.py -- download & parse GEO/TCGA expression matrices

此模块负责从GEO或TCGA数据源下载和解析基因表达数据。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import GEOparse

def load_geo(gse_id, destdir='data/raw'):
    """
    加载GEO数据集并解析表达矩阵

    Args:
        gse_id: GEO数据集编号，如 'GSE12345'
        destdir: 数据存储目录

    Returns:
        expr: 表达矩阵 (探针 × 样本)
        meta: 样本元数据 (样本 × 特征)
    """
    gse = GEOparse.get_GEO(geo=gse_id, destdir=destdir)
    expr = gse.pivot_samples('VALUE')  # 探针 × 样本
    meta = gse.phenotype_data          # 样本临床特征
    return expr, meta

def probe_to_gene(expr, probe2gene_map=None):
    """
    将探针ID映射到基因名并聚合

    Args:
        expr: 表达矩阵，行索引为探针ID
        probe2gene_map: 探针到基因的映射字典

    Returns:
        聚合后的基因表达矩阵
    """
    if probe2gene_map is None:
        # 如果没有提供映射，尝试使用GEO自带的注释
        expr['gene'] = expr.index  # 简化处理，实际需要映射
        return expr

    expr = expr.copy()
    expr['gene'] = expr.index.map(probe2gene_map)
    expr = expr.dropna(subset=['gene'])
    expr = expr.groupby('gene').mean()  # 同名基因取均值
    return expr

def load_tcga(manifest_file):
    """
    加载TCGA数据集（备选方案）

    Args:
        manifest_file: TCGA manifest文件路径

    Returns:
        expr: 表达矩阵
        meta: 样本元数据
    """
    # TODO: 实现TCGA数据加载逻辑
    raise NotImplementedError("TCGA数据加载待实现")

if __name__ == "__main__":
    # 测试代码
    print("data_loader模块加载成功")
    print("注意：实际使用时需要提供有效的GEO数据集ID")