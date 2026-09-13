"""Unit tests for GeneTransformer + the exact [CLS]-attention recomputation.

These are data-free: they only build small random models and check shapes and
the correctness of the memory-safe attention (the core interpretability code).
"""
import torch
import torch.nn as nn
import pytest

from src.model import GeneTransformer, _cls_attention, count_parameters, create_model


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return GeneTransformer(
        n_genes=16, d_model=32, n_layers=2, n_heads=4,
        dim_feedforward=64, n_classes=2, dropout=0.1,
    )


def test_forward_logits_shape(tiny_model):
    x = torch.randn(5, 16)
    logits = tiny_model(x)
    assert logits.shape == (5, 2)


def test_forward_attention_shapes(tiny_model):
    x = torch.randn(5, 16)
    logits, cls_attn = tiny_model.forward_attention(x)
    assert logits.shape == (5, 2)
    assert len(cls_attn) == 2  # n_layers
    for a in cls_attn:
        assert a.shape == (5, 4, 17)  # (B, H, L=n_genes+1)


def test_get_attention_weights_shape(tiny_model):
    x = torch.randn(5, 16)
    w = tiny_model.get_attention_weights(x)
    assert w.shape == (5, 16)


def test_cls_attention_matches_torch():
    """The hand-computed CLS row must equal MultiheadAttention(need_weights=True)."""
    torch.manual_seed(0)
    sa = nn.TransformerEncoderLayer(
        d_model=32, nhead=4, dim_feedforward=64, batch_first=True, dropout=0.1,
    ).self_attn
    sa.eval()
    x = torch.randn(2, 17, 32)  # (B, L, d_model)
    with torch.no_grad():
        manual = _cls_attention(sa, x)  # (B, H, L)
        _, ref = sa(x, x, x, need_weights=True, average_attn_weights=False)
    ref_cls = ref[:, :, 0, :]  # CLS query = token 0 -> (B, H, L)
    assert manual.shape == (2, 4, 17)
    assert torch.allclose(manual, ref_cls, atol=1e-6, rtol=1e-5)


def test_count_parameters(tiny_model):
    assert count_parameters(tiny_model) > 0


def test_create_model_from_config():
    cfg = {
        "model": {
            "n_genes": 8, "d_model": 16, "n_layers": 1, "n_heads": 2,
            "dim_feedforward": 32, "n_classes": 2, "dropout": 0.0,
        }
    }
    m = create_model(cfg)
    assert m.n_genes == 8
    assert m.n_classes == 2
    assert m(torch.randn(3, 8)).shape == (3, 2)
