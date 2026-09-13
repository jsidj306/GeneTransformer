"""
model.py -- GeneTransformer (embedding + encoder + head)

此模块实现了基于Transformer的基因表达分类模型。
"""
import torch
import torch.nn as nn
import math


def _cls_attention(sa_module, x):
    """[CLS] token's self-attention over all tokens for one attention layer.

    Recomputes the CLS query's softmax attention from the layer's own Q/K
    projection weights, materializing only the CLS row (O(L) instead of the
    O(L^2) full matrix). Exact to float32 precision vs ``sa_module`` with
    ``need_weights=True`` (verified 2026-09-13).

    Args:
        sa_module: nn.MultiheadAttention (self-attention, q/k/v same input)
        x: (B, L, d_model) the layer input used as query/key/value

    Returns:
        (B, n_heads, L) attention of the [CLS] query (token 0) to every token.
    """
    E = sa_module.embed_dim
    H = sa_module.num_heads
    d_head = E // H
    w = sa_module.in_proj_weight  # (3E, E)
    b = sa_module.in_proj_bias    # (3E,)

    q = x @ w[:E].T + b[:E]
    k = x @ w[E:2 * E].T + b[E:2 * E]

    B, L, _ = q.shape
    q = q.view(B, L, H, d_head).transpose(1, 2)   # (B, H, L, d_head)
    k = k.view(B, L, H, d_head).transpose(1, 2)
    q_cls = q[:, :, 0:1, :]                        # (B, H, 1, d_head)

    scores = (q_cls @ k.transpose(-2, -1)) * (d_head ** -0.5)  # (B, H, 1, L)
    attn = torch.softmax(scores, dim=-1)           # (B, H, 1, L)
    return attn.squeeze(2)                         # (B, H, L)


class GeneTransformer(nn.Module):
    """
    基因表达Transformer分类模型

    架构：
    - 每个基因有一个可学习的embedding
    - 表达值作为embedding的缩放系数
    - 使用[CLS] token汇总信息
    - 多头自注意力 + MLP分类头

    Args:
        n_genes: 基因数量
        d_model: 嵌入维度
        n_layers: Transformer编码器层数
        n_heads: 注意力头数
        dim_feedforward: 前馈网络隐藏维度
        n_classes: 分类类别数
        dropout: Dropout比率
    """
    def __init__(self, n_genes, d_model=128, n_layers=2, n_heads=4,
                 dim_feedforward=256, n_classes=3, dropout=0.1):
        super().__init__()

        self.n_genes = n_genes
        self.d_model = d_model
        self.n_classes = n_classes

        # 基因嵌入层
        self.gene_emb = nn.Embedding(n_genes, d_model)

        # [CLS] token
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

        # 位置编码
        self.pos = nn.Parameter(torch.randn(1, n_genes + 1, d_model))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 基因表达值 (batch_size, n_genes)

        Returns:
            logits: 分类输出 (batch_size, n_classes)
        """
        B = x.shape[0]

        # 基因嵌入 + 表达值缩放
        gene_embeddings = self.gene_emb.weight.unsqueeze(0).expand(B, -1, -1)
        scaled_embeddings = gene_embeddings * x.unsqueeze(-1)

        # 添加[CLS] token
        cls_tokens = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tokens, scaled_embeddings], dim=1)

        # 添加位置编码
        x = x + self.pos

        # Transformer编码
        encoded = self.encoder(x)

        # 使用[CLS]输出进行分类
        cls_output = encoded[:, 0]

        # 分类头
        logits = self.head(cls_output)

        return logits

    def forward_attention(self, x):
        """Forward pass + per-layer [CLS]->token attention (memory-safe).

        The [CLS] token's self-attention to every other token is the key
        interpretability signal. It is recomputed from each layer's Q/K
        projection weights on the layer input actually used (captured via a
        forward hook), so it is exact but only materializes the CLS row, not
        the O(n_genes^2) full attention matrix.

        Args:
            x: gene expression values (batch_size, n_genes)

        Returns:
            (logits, cls_attn): logits (B, n_classes); cls_attn is a list over
            layers of (B, n_heads, L) with L = n_genes + 1 (token 0 = [CLS]).
        """
        captures = []
        hooks = []

        def make_hook(cap):
            def hook(module, args, _output):
                cap.append(_cls_attention(module, args[0]))
            return hook

        for layer in self.encoder.layers:
            cap = []
            hooks.append(layer.self_attn.register_forward_hook(make_hook(cap)))
            captures.append(cap)

        logits = self.forward(x)  # hooks fire inside the real forward

        for h in hooks:
            h.remove()

        return logits, [cap[0] for cap in captures]

    def get_attention_weights(self, x):
        """Mean [CLS]->gene attention across layers/heads (convenience wrapper).

        Returns (B, n_genes): for each sample, how much the [CLS] token attends
        to each gene, averaged over every layer and attention head.
        """
        with torch.no_grad():
            _, cls_attn = self.forward_attention(x)
        # cls_attn: list over layers of (B, n_heads, L); drop [CLS] self-token.
        per_layer = torch.stack([a[:, :, 1:] for a in cls_attn], dim=1)  # (B, L, H, G)
        return per_layer.mean(dim=(1, 2))  # (B, G)

def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model(config):
    """
    根据配置创建模型

    Args:
        config: 配置字典

    Returns:
        模型实例
    """
    model_config = config.get('model', {})

    model = GeneTransformer(
        n_genes=model_config.get('n_genes', 5000),
        d_model=model_config.get('d_model', 128),
        n_layers=model_config.get('n_layers', 2),
        n_heads=model_config.get('n_heads', 4),
        dim_feedforward=model_config.get('dim_feedforward', 256),
        n_classes=model_config.get('n_classes', 3),
        dropout=model_config.get('dropout', 0.1)
    )

    return model

if __name__ == "__main__":
    print("model模块加载成功")

    # 测试模型
    model = GeneTransformer(n_genes=5000, d_model=128, n_classes=3)

    # 测试前向传播
    x = torch.randn(4, 5000)  # batch_size=4
    output = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {count_parameters(model):,}")