# 把 Transformer 用在基因表达上：一个可解释的肿瘤/正常分类器

> 标签：深度学习 / PyTorch / Transformer / 生物信息学
> 项目地址：[GeneTransformer](https://github.com/jsidj306/GeneTransformer)
> 系列：第 2 篇 / 共 3 篇

---

## 为什么是"肿瘤 vs 正常"，而不是"哪种癌"

承接[上一篇](https://github.com/jsidj306/GeneTransformer)的批次泄漏复盘：三个公开数据集每个只含一种癌，导致「平台」和「癌种」1:1 混叠，多分类在 LODO 下 macro-F1 = 0.0。

但「肿瘤 vs 正常」是另一种信号——它在**每个**平台都存在，且不依赖样本属于哪种具体癌。于是我把任务重新定义为：留一数据集（LODO）下的肿瘤/正常二分类。这是诚实且困难的泛化基准。

## 数据与预处理

| 平台 (GSE) | 组织 | 样本 | 正常 | 肿瘤 |
|---|---|---|---|---|
| GSE31210 | 肺 | 246 | 20 | 226 |
| GSE39582 | 结直肠 | 585 | 19 | 566 |
| GSE45827 | 乳腺 | 141 | 11 | 130 |
| 合计 | | **972** | **50** | **922** |

类别比 ≈ 18:1（肿瘤:正常），所以主指标用**阈值无关**的 PR-AUC / ROC-AUC。

预处理三步：

1. `log2(x+1)` 抑制表达量的长尾；
2. 逐基因 z-score（**只在训练集拟合**均值/方差）；
3. 按折高变基因（HVG）选 top-5000，作为 Transformer 输入。

## 模型：GeneTransformer

思路是把「基因」当作「token」：

```python
class GeneTransformer(nn.Module):
    def __init__(self, n_genes, d_model=128, n_layers=2, n_heads=4, n_classes=2):
        self.gene_emb = nn.Embedding(n_genes, d_model)        # 每个基因一个 embedding
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))    # [CLS] token
        self.pos = nn.Parameter(torch.randn(1, n_genes + 1, d_model))
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, n_classes))

    def forward(self, x):            # x: (B, n_genes) 表达值
        emb = self.gene_emb.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        emb = emb * x.unsqueeze(-1)  # 表达值作为 embedding 的缩放系数
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), emb], dim=1) + self.pos
        return self.head(self.encoder(x)[:, 0])   # 取 [CLS] 输出分类
```

关键设计：表达值作为 gene embedding 的**缩放系数**——高表达的基因其 embedding 权重更大；`[CLS]` token 汇总整条样本后接 MLP 分类头。

## 结果：与 SVM 同档，且可解释

| 模型 (LODO) | PR-AUC | ROC-AUC |
|---|---|---|
| Random Forest | 0.9961 ± 0.0025 | 0.9486 ± 0.0250 |
| SVM (RBF) | 0.9986 ± 0.0012 | 0.9776 ± 0.0154 |
| **Gene Transformer** | **0.9963 ± 0.0029** | **0.9637 ± 0.0192** |

Transformer 在 PR-AUC 上与 SVM 持平、在 AUC 上超过 RF，同时保持端到端可微——这是后面做 SHAP/Attention 解释的前提。**这个项目要的不是「AUC 最高」，而是「能解释为什么」。**

## 一个必须解决的坑：跨平台分数偏移

排序可以跨平台迁移，但**分数尺度不能**。训练折上模型饱和（肿瘤≈1.0、正常≈0.0），但留出折的正常样本分数会向上漂进肿瘤区间（测试正常中位数≈0.85），于是任何在训练折上学的阈值都会把正常样本误判成肿瘤。

我的解法是**分位数对齐**：每个留出折在训练折的正常占比分位数处切分，正常召回率从 0 提到 0.54（RF）/ 0.59（SVM）。

> 关键结论：`class_weight`（哪怕 18:1 ≈ balanced）**修不了**这个问题——它是跨平台技术差异造成的分数尺度偏移，不是类别先验问题。别一遇到不平衡就无脑上 class_weight。

## 小结

- 跨平台生物数据，**主指标选阈值无关的 PR-AUC/AUC**，别被 accuracy 骗。
- 「表达值缩放 embedding」是个简单有效的「基因做 token」方案。
- 端到端可微的模型，才能做下一步的归因解释（见[第三篇](https://github.com/jsidj306/GeneTransformer)）。

代码、checkpoint、复现步骤都在 [GeneTransformer](https://github.com/jsidj306/GeneTransformer)。
