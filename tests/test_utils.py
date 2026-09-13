"""Unit tests for config loading, seeding, and small helpers."""
import numpy as np

from src.utils import load_config, set_seed, get_device, ensure_dir, format_time


def test_load_config_keys():
    cfg = load_config("config.yaml")
    assert isinstance(cfg, dict)
    assert "model" in cfg
    assert "seed" in cfg


def test_load_config_values():
    cfg = load_config("config.yaml")
    m = cfg["model"]
    assert m["n_genes"] == 5000
    assert m["d_model"] == 128
    assert m["n_layers"] == 2
    assert m["n_heads"] == 4
    assert cfg["seed"] == 42


def test_set_seed_reproducible():
    set_seed(123)
    a = np.random.rand(4)
    set_seed(123)
    b = np.random.rand(4)
    assert np.array_equal(a, b)


def test_get_device():
    assert get_device().type in ("cpu", "cuda")


def test_ensure_dir(tmp_path):
    d = tmp_path / "sub" / "dir"
    ensure_dir(str(d))
    assert d.is_dir()


def test_format_time_is_str():
    assert isinstance(format_time(65), str)
