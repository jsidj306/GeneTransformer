# 模型到底"看"到了什么？SHAP + Attention + 通路富集的三重复盘

> 标签：可解释性 / SHAP / 深度学习 / 生物信息学
> 项目地址：[GeneTransformer](https://github.com/jsidj306/GeneTransformer)
> 系列：第 3 篇 / 共 3 篇

---

## 为什么需要解释

一个 Transformer 在肿瘤/正常分类上拿到 PR-AUC 0.9963，但它到底靠哪些基因判断？如果它靠的是噪声，这个模型在生物学上就毫无价值。所以我做了三层独立复盘，交叉印证。

## 第一层：Deep SHAP = gradient × input

对这类可微模型，`shap.DeepExplainer` 会退化成梯度×输入的线性化：

```text
SHAP[s, i] = (x[s, i] − mean(bg)[i]) · mean(∂ logit / ∂ x_i)
```

背景取训练集分层均值，归因目标 = `logit[tumor] − logit[normal]`。因为 5001 个 token 的自注意力是 O(n²)，在 5GB 内存下会 OOM，所以我按 batch=1 分块计算。

## 第二层：Attention —— 手算 [CLS] 行

Transformer 的注意力矩阵是 O(L²) 的，我只想要 `[CLS]` token 对每个基因的注意力（一行）。于是从每层 Q/K 投影权重**手算 CLS 这一行**，只物化 O(L)：

```python
q = x @ W_Q.T + b_Q          # 只算 CLS 的 query
k = x @ W_K.T + b_K
scores = (q_cls @ k.T) * d_head ** -0.5
attn = softmax(scores, dim=-1)   # (B, H, L) 只保留 CLS 行
```

和 `need_weights=True` 的官方实现比对，**精确一致到 1e-9**。

## 第三层：跨折共识基因 + 通路富集

单折的 top 基因很平台特异（GSE31210 → PLEKHH1/NEDD4，GSE39582 → GALNT7/SLC44A5……），这本身就是域偏移的写照。所以我用**基于排名 + 缺席惩罚**的跨折聚合：每个基因的逐折重要性转成排名，某折 HVG 缺失的基因给最差排名，两方法平均排名相加——奖励那些**跨平台稳定重要**的基因。

三折的共识基因（top20）：

> ZNF521, PFKFB3, COL4A5, LPL, AFAP1-AS1, CDO1, DPT, LCN2, COL11A1, ABCA8, ACADSB, CLTC, C2CD4A, EBF1, FLJ13744, LEPR, AR, EGR3, CA4, DCN

把它们丢进 Enrichr 做 GO/KEGG/Reactome 富集，结果是：

| 库 | 通路 | 校正 p |
|---|---|---|
| GO:CC | Collagen-Containing Extracellular Matrix | 3.9 × 10⁻⁶ |
| Reactome | Extracellular Matrix Organization | 0.043 |

**唯一 FDR 显著的是「细胞外基质 / 胶原重塑」**，而 KEGG、GO:BP、GO:MF 全部不显著。

## 这意味着什么：模型看到的是"基质"，不是"增殖"

共识基因是一组**肿瘤基质 / 细胞外基质（ECM）签名**——胶原蛋白（COL11A1、COL4A5、COL10A1）、decorin（DCN）、dermatopontin（DPT）、fibulin（FBLN2），外加一个代谢分量（PFKFB3 糖酵解、LPL、PDK4）。

这是肿瘤**促结缔组织增生（desmoplasia）**的标志，而不是经典的增殖程序。有意思的是：随机森林的 top 基因是增殖标志（TPX2、ECT2、UBE2C、UBE2T、PPAT、BOP1），和 Transformer 几乎不重叠。原因很直接：HVG 按**方差**选基因，低方差的增殖基因常被挤出 5000 基因的 Transformer 输入，而 RF 看的是全部 ~1.5 万基因。两个模型的归纳偏置不同，看到的东西自然不同。

## 三条 takeaway

1. **可解释性要"多视角交叉印证"**：SHAP（模型无关）+ Attention（模型内置）+ 富集（生物先验），三者指向同一结论才可信。
2. **跨折聚合要用 rank-based + 缺席惩罚**，别对重尾的原始 SHAP/注意力分数直接求平均。
3. **诚实汇报**：富集里唯一显著的是 ECM/胶原，而不是"癌症通路"——该是什么就是什么。

完整报告（英/中双语）与全部图都在 [GeneTransformer](https://github.com/jsidj306/GeneTransformer)。
