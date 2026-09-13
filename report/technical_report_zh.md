# 技术报告 —— 从批次泄漏到可解释的肿瘤/正常分类器

**项目：** GeneTransformer · **日期：** 2026-09-13 · **状态：** 已完成

> English version / 英文版：[technical_report.md](technical_report.md)

---

## 摘要

我们最初的目标是用可解释的 Transformer 从 bulk 基因表达数据中预测**癌症类型**。严格的
留一数据集（LODO）评估揭示了一个关键事实：之前"100% 准确率"其实是**批次泄漏**——每个公开
GEO 系列本身就是单一癌症类型，因此"平台"与"类别"完全混杂，随机划分让任何模型都能靠记住
平台拿到满分。在 LODO 下，多分类任务崩到随机水平（macro-F1 = 0.0），因为"未见过的癌症类型"
与"未见过的批次"无法区分。转向跨平台可泛化的**肿瘤 vs 正常**二分类任务后（这里的信号真实存在
且可迁移），Transformer 达到 **PR-AUC 0.9963 ± 0.0029 / ROC-AUC 0.9637 ± 0.0192**，与强基线
RBF-SVM 持平。SHAP 与 `[CLS]` 注意力归因揭示出一组跨折共识基因，以**肿瘤基质 / 细胞外基质
重塑**为主（胶原蛋白、*DCN*、*DPT*、*FBLN2*），并带有代谢分量（*PFKFB3*、*LPL*、*PDK4*）。
通路富集（Enrichr）证实这是**唯一**通过多重检验校正的信号（GO:CC "含胶原的细胞外基质"，
校正 p = 3.9 × 10⁻⁶）。本报告完整记录了混杂现象、任务转向、击败固定阈值的域偏移机制，以及
随机森林增殖签名与 Transformer 基质签名之间的分歧。

---

## 1. 引言与任务

**原始任务。** 用 Transformer 编码器从 bulk 微阵列表达数据中分类癌症类型（乳腺癌 BRCA、
肺癌 LUAD、结直肠癌 COAD），并解释每一个预测。

**泛化难题。** 这三类癌症可用的公开数据来自三个*不同*的 GEO 平台。由于每个平台恰好也是单一
癌症类型，类别标签与批次相互混叠。项目的核心方法论问题因此变成：*模型学到的是生物学信号，
还是批次？* 我们用留一数据集协议（§3）来回答这个问题；在该协议暴露混杂之后，我们把任务重新
聚焦到真正可迁移的信号上：肿瘤 vs 正常（§4）。

---

## 2. 数据

取交集到共同的 14,725 个基因后，共 972 个 bulk **肿瘤/正常**样本，来自三个独立的 Affymetrix
平台：

| 平台 (GSE) | 组织 | 样本数 | 正常 | 肿瘤 |
|---|---|---|---|---|
| GSE31210 | 肺 (LUAD) | 246 | 20 | 226 |
| GSE39582 | 结直肠 (COAD) | 585 | 19 | 566 |
| GSE45827 | 乳腺 (BRCA) | 141 | 11 | 130 |
| **合计** | | **972** | **50** | **922** |

类别不平衡约为 18:1（肿瘤:正常），因此我们以阈值无关指标为主（§5），并在支持类别权重的模型
中对正常类上采样权重。

**预处理。** `log2(x + 1)` → 逐基因 z-score → **按折高变基因（HVG）选择**（Transformer 取
训练集方差最大的 5000 个基因）。基线模型在相同的 HVG 特征上拟合，以保证公平对比。

---

## 3. 关键发现：批次泄漏

多分类任务存在致命的混杂。我们用三条相互独立的证据验证了这一点：

1. **批次身份可被平凡预测。** 一个预测样本来自*哪个 GSE 平台*的分类器，在批次校正前得分
   **1.0**，仅批次 ComBat 校正后为 0.980。癌症身份同样可被预测到 1.0——因为 *批次 == 类别*。
2. **随机划分对照组发生泄漏。** 用随机训练/测试划分（也就是当初产生"100% 准确率"的设置），
   RF 与 SVM 都达到 **macro-F1 = 1.0**。
3. **LODO 崩到随机。** 每次留出一整个平台（一个未见过的癌症类型），两个基线都掉到
   **macro-F1 = 0.0**——模型只能把留出样本判给*见过*的癌症类型，因此永远判错。

![批次与癌症类别混叠 —— 批次校正前的 PCA](../results/figures/pca_by_batch_before.png)

**结论。** 之前的 100% 准确率是批次效应，不是生物学信号。`类别 == 批次`，所以带癌症协变量的
ComBat 是奇异（共线）设计，而仅批次的 ComBat 会连同类别信号一起把批次抹掉。只有 LODO 能暴露
这个混杂，而它表明：**仅凭这三个系列，跨平台的多分类癌症类型识别不可行。** 这是决定性的负结果，
它塑造了项目后续的全部走向。

---

## 4. 转向：二分类肿瘤/正常

肿瘤 vs 正常是另一种信号：它在*每一个*平台都存在，且不依赖于样本属于哪种具体癌症（因而也不依赖
于批次）。因此我们把任务重新定义为 **LODO 下的二分类肿瘤/正常分类**——这是诚实且困难的泛化基准。
后续所有结果均基于此任务。

---

## 5. 方法

### 模型
- **随机森林（RF）** 与 **RBF-SVM** —— 在 HVG 特征上的强经典基线。
- **GeneTransformer** —— 每个基因有一个可学习的 `d=128` 嵌入，乘以其表达值缩放；前置一个可学习的
  `[CLS]` token；两层多头自注意力编码器（`n_heads=4`、`d_ff=256`、GELU、post-LN）聚合上下文；
  `[CLS]` 输出送入 MLP 分类头（`→ 64 → 2`）。

### 评估
- **主指标为阈值无关的 PR-AUC 与 ROC-AUC。** 本任务弃用准确率 / F1 / 平衡准确率：在 ≈ 18:1 的
  不平衡与域偏移下，它们会退化成误导性的"全判肿瘤" 0.5，即使排序信号是真实的。
- **硬标签（次要指标）用分位数对齐**：每个留出折的分数在*训练*折的正常占比分位数处切分——在测试集
  上无标签依赖，也是唯一能扛过跨平台分数偏移的阈值（§6）。
- **LODO** 每次留出一整个平台；每个测试折在训练中从未出现。

---

## 6. 结果 —— 二分类

| 模型 (LODO) | PR-AUC (均值 ± 标准差) | ROC-AUC (均值 ± 标准差) |
|---|---|---|
| 随机森林 | 0.9961 ± 0.0025 | 0.9486 ± 0.0250 |
| SVM (RBF) | 0.9986 ± 0.0012 | 0.9776 ± 0.0154 |
| **Gene Transformer** | **0.9963 ± 0.0029** | **0.9637 ± 0.0192** |

Transformer 在 PR-AUC 上与 SVM 持平，在 ROC-AUC 上超过 RF，同时保持端到端可微——这正是 §7 的
归因分析所依赖的性质。

![ROC / PR 曲线 (LODO)](../results/figures/tumor_normal_roc_pr.png)

### 域偏移击败固定阈值

肿瘤/正常的*排序*可以跨平台迁移，但*分数尺度*不能。在训练折上模型饱和（肿瘤 ≈ 1.0、正常 ≈ 0.0）；
在留出折上，正常分数向上漂进肿瘤区间（测试正常中位数 ≈ 0.85），于是任何在训练折上学到的阈值都会
把大多数留出正常样本误判。

![分数偏移：测试正常分数跨平台向肿瘤漂移](../results/figures/tumor_normal_score_shift.png)

分位数对齐解决了这个问题：正常召回率从固定阈值下的 ≈ 0 升到 0.54（RF）/ 0.59（SVM）。关键的是，
`class_weight`（即便是 18:1）≈ balanced **不能**解决它——这个偏移是跨平台技术差异造成的分数尺度
偏移，而不是类别先验问题。

---

## 7. 可解释性 —— SHAP 与 Attention

两种互补、纯推理的视角，回答模型*为什么*如此决策（无需重训；在 CPU 上从本地 checkpoint 运行）：

- **SHAP（Deep SHAP / Gradient × Input）。** 对该架构，`shap.DeepExplainer` 退化为
  `SHAP[s,i] = (x[s,i] − mean(bg)[i]) · mean(d logit / d x_i)`，背景 = 分层训练均值，目标 =
  `logit[tumor] − logit[normal]`。以 `batch=1` 分块计算，因为 5001 个 token 的完整自注意力是
  O(n²)，在 ~5 GB 内存下会 OOM。
- **Attention（`[CLS]` → 基因）。** 从每一层的 Q/K 投影权重重算 `[CLS]` token 的自注意力行
  （`_cls_attention`），只物化 CLS 行（O(L)）——与 `need_weights=True` 精确一致到 1e-9——再对层、
  头、测试样本取平均。

**跨折聚合**采用基于排名的、带缺席惩罚的方式：每个基因的逐折重要性转成排名（1 = 最重要）；某折
HVG 缺失的基因得最差排名（`n_genes+1`）；两种方法的平均排名相加。这奖励那些*跨平台稳定*重要的
基因，并对重尾的原始 SHAP/注意力分数稳健。

### 共识基因（前 20，基于排名，`n_folds=3`）

> **ZNF521, PFKFB3, COL4A5, LPL, AFAP1-AS1, CDO1, DPT, LCN2, COL11A1, ABCA8,
> ACADSB, CLTC, C2CD4A, EBF1, FLJ13744, LEPR, AR, EGR3, CA4, DCN**

它们聚成两类生物学分组：**细胞外基质 / 基质重塑**（*COL11A1、COL4A5、COL10A1、DCN、DPT、
FBLN2、EFEMP1、CCDC80、ABI3BP、OGN、CLEC3B、EDIL3*）与**代谢 / 信号**（*PFKFB3* 糖酵解、
*CDO1* 甲基化标志、*LCN2*、*LPL、PDK4、ACADSB、GPAM*）。逐折 top 基因明显平台特异（GSE31210 →
PLEKHH1/NEDD4；GSE39582 → GALNT7/SLC44A5；GSE45827 → CST1/GPR110/SCARA5），与域偏移相呼应
——共识基因是三个平台都留存下来的部分。

![SHAP 总结 —— GSE31210（最清晰图 #1）](../results/figures/tn_shap_summary_GSE31210.png)

![CLS 注意力 —— GSE31210（最清晰图 #2）](../results/figures/tn_attention_GSE31210.png)

![SHAP 总结 —— GSE39582（最清晰图 #3：主导基因不同）](../results/figures/tn_shap_summary_GSE39582.png)

---

## 8. 通路富集（GO / KEGG / Reactome）

前 100 共识基因 → Enrichr（`KEGG_2021_Human`、`GO_BP/MF/CC_2023`、`Reactome_2022`）。

![富集气泡图](../results/figures/enrichment_bubble.png)

**唯一 FDR 显著的信号是细胞外基质 / 胶原组织：**

| 库 | 通路 | 校正 p | 重叠 |
|---|---|---|---|
| GO:CC | Collagen-Containing Extracellular Matrix | 3.9 × 10⁻⁶ | 13 |
| GO:CC | Extracellular Vesicle / Membrane-Bounded Organelle | 0.0069 | 4 |
| Reactome | Assembly of Collagen Fibrils & Other Multimeric Structures | 0.043 | 4 |
| Reactome | Extracellular Matrix Organization | 0.043 | 7 |
| Reactome | Collagen Formation / Chain Trimerization | 0.043–0.049 | 3–4 |

KEGG、GO Biological Process 与 GO Molecular Function 均**没有**通过多重检验校正的项——它们最强的
名义显著命中（GO:BP "ECM Organization" 原始 p = 2.6 × 10⁻⁴、校正 p = 0.088；KEGG "AMPK signaling"
原始 p = 0.022、校正 p = 0.48；KEGG "Pathways in cancer" 校正 p = 0.48）只是趋势。我们**如实**
报告：共识基因集是一个**肿瘤基质 / ECM 重塑签名**，带有次级代谢分量——而不是经典的癌基因/抑癌
通路。

---

## 9. 讨论

1. **Transformer 看到的是肿瘤微环境。** 它最一致的基因是基质/ECM——胶原蛋白、decorin、
   dermatopontin、fibulin——这是肿瘤促结缔组织增生（desmoplasia）的标志，而不是增殖程序。
2. **与 RF 增殖签名重叠低。** RF 的 top 基因是增殖标志（*TPX2、ECT2、UBE2C、UBE2T、PPAT、
   BOP1*）；它们在 Transformer 中只排到约 1000–3000（SHAP）/ 400–900（注意力），且多个基因在
   若干折中缺失（`n_folds < 3`）。这是可预期的：**(a)** HVG 按*方差*选基因，低方差的增殖基因经常
   被挤出 5000 基因的 Transformer 输入，而 RF 看到全部 ~15k 基因；**(b)** 两个模型的归纳偏置不同。
   Transformer 的 HVG 输入使它偏向高方差的基质信号。
3. **支配硬标签的是域偏移，而不是类别先验。** 在 LODO 下分数尺度跨平台漂移；只有分位数对齐能给出
   可用的硬标签。这是数据/特征选择的属性，不是架构的属性。
4. **决定性的结果是那个负结果。** 仅凭这些系列，跨平台多分类癌症类型识别不可行（`批次 == 类别`）；
   肿瘤/正常信号才可行。任何后续工作都应先取得批次 ≠ 类别的数据（例如 TCGA——单一平台、多癌种 +
   配对正常）再回到多分类。

### 局限性
- 可解释性在每个折上只跑 30 个分层测试样本（受内存限制），因此逐折排名是指示性的，并非穷尽。
- 富集前景是前 100 共识基因（按*排名*），不是差异表达统计检验；背景是 Enrichr 默认（全体人类
  基因）。共识中的 lncRNA/loci（AFAP1-AS1、FLJ13744、LOC285628、…）无法映射到通路，被 Enrichr
  静默排除。
- Transformer 的 AUC 略逊于 SVM；我们未激进调参，因为项目目标是可解释性，而非在 AUC 上胜过 SVM。

### 可复现性
所有超参数与路径都在 `config.yaml` 中；随机种子固定为 42。
命令：`python -m src.baseline`、`python -m src.train`、`python -m src.explain_tn`、
`python -m src.enrichment`。checkpoint、预处理矩阵与原始数据均 gitignore（可复现、不提交）。
