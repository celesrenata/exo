"""
Unit tests for LoadingMemoryMetrics and measure_loading_memory.

Tests that:
- LoadingMemoryMetrics fields are populated correctly
- peak_rss_mib property computes correctly
- rss_delta_mib property computes correctly
- tensor_mib_loaded property computes correctly
- measure_loading_memory returns both the shard and metrics

**Validates: Requirements 1.8**
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent
_PIPELINE_CONFIG_PATH = _ENGINE_DIR / "pipeline_config.py"
_LOCAL_SHARD_LOADER_PATH = _ENGINE_DIR / "local_shard_loader.py"


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    """Load a module directly from file, avoiding __init__.py."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load pipeline_config under its canonical import path
_config_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.pipeline_config", _PIPELINE_CONFIG_PATH
)
_loader_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.local_shard_loader",
    _LOCAL_SHARD_LOADER_PATH,
)

PipelineLayerDistribution = _config_mod.PipelineLayerDistribution
LoadingMemoryMetrics = _loader_mod.LoadingMemoryMetrics
measure_loading_memory = _loader_mod.measure_loading_memory


# ---------------------------------------------------------------------------
# Tests for LoadingMemoryMetrics dataclass
# ---------------------------------------------------------------------------


class TestLoadingMemoryMetrics:
    """Test LoadingMemoryMetrics fields and computed properties."""

    def test_fields_populated_correctly(self) -> None:
        """All fields are stored with the values provided at construction."""
        metrics = LoadingMemoryMetrics(
            peak_rss_bytes=2_147_483_648,  # 2 GiB
            rss_before_bytes=1_073_741_824,  # 1 GiB
            rss_after_bytes=2_147_483_648,  # 2 GiB
            rss_delta_bytes=1_073_741_824,  # 1 GiB
            tensor_bytes_loaded=536_870_912,  # 512 MiB
            loading_duration_seconds=3.5,
        )

        assert metrics.peak_rss_bytes == 2_147_483_648
        assert metrics.rss_before_bytes == 1_073_741_824
        assert metrics.rss_after_bytes == 2_147_483_648
        assert metrics.rss_delta_bytes == 1_073_741_824
        assert metrics.tensor_bytes_loaded == 536_870_912
        assert metrics.loading_duration_seconds == 3.5

    def test_peak_rss_mib_property(self) -> None:
        """peak_rss_mib converts bytes to mebibytes correctly."""
        # 1048576 bytes = 1 MiB
        metrics = LoadingMemoryMetrics(
            peak_rss_bytes=1048576,
            rss_before_bytes=0,
            rss_after_bytes=1048576,
            rss_delta_bytes=1048576,
            tensor_bytes_loaded=0,
            loading_duration_seconds=0.0,
        )
        assert metrics.peak_rss_mib == 1.0

    def test_peak_rss_mib_large_value(self) -> None:
        """peak_rss_mib handles large values (multi-GiB)."""
        # 4 GiB = 4096 MiB
        four_gib = 4 * 1024 * 1024 * 1024
        metrics = LoadingMemoryMetrics(
            peak_rss_bytes=four_gib,
            rss_before_bytes=0,
            rss_after_bytes=four_gib,
            rss_delta_bytes=four_gib,
            tensor_bytes_loaded=0,
            loading_duration_seconds=0.0,
        )
        assert metrics.peak_rss_mib == 4096.0

    def test_rss_delta_mib_property(self) -> None:
        """rss_delta_mib converts delta bytes to mebibytes correctly."""
        # 512 MiB delta
        delta = 512 * 1024 * 1024
        metrics = LoadingMemoryMetrics(
            peak_rss_bytes=1024 * 1024 * 1024,
            rss_before_bytes=512 * 1024 * 1024,
            rss_after_bytes=1024 * 1024 * 1024,
            rss_delta_bytes=delta,
            tensor_bytes_loaded=0,
            loading_duration_seconds=0.0,
        )
        assert metrics.rss_delta_mib == 512.0

    def test_tensor_mib_loaded_property(self) -> None:
        """tensor_mib_loaded converts tensor bytes to mebibytes correctly."""
        # 256 MiB of tensors
        tensor_bytes = 256 * 1024 * 1024
        metrics = LoadingMemoryMetrics(
            peak_rss_bytes=0,
            rss_before_bytes=0,
            rss_after_bytes=0,
            rss_delta_bytes=0,
            tensor_bytes_loaded=tensor_bytes,
            loading_duration_seconds=0.0,
        )
        assert metrics.tensor_mib_loaded == 256.0

    def test_zero_values(self) -> None:
        """Handles zero values without errors."""
        metrics = LoadingMemoryMetrics(
            peak_rss_bytes=0,
            rss_before_bytes=0,
            rss_after_bytes=0,
            rss_delta_bytes=0,
            tensor_bytes_loaded=0,
            loading_duration_seconds=0.0,
        )
        assert metrics.peak_rss_mib == 0.0
        assert metrics.rss_delta_mib == 0.0
        assert metrics.tensor_mib_loaded == 0.0

    def test_is_frozen(self) -> None:
        """LoadingMemoryMetrics is immutable (frozen dataclass)."""
        metrics = LoadingMemoryMetrics(
            peak_rss_bytes=100,
            rss_before_bytes=50,
            rss_after_bytes=100,
            rss_delta_bytes=50,
            tensor_bytes_loaded=25,
            loading_duration_seconds=1.0,
        )
        with pytest.raises(AttributeError):
            metrics.peak_rss_bytes = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests for measure_loading_memory
# ---------------------------------------------------------------------------


def _has_safetensors_and_torch() -> bool:
    """Check if safetensors and torch are available."""
    try:
        import safetensors.torch  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


_requires_safetensors = pytest.mark.skipif(
    not _has_safetensors_and_torch(),
    reason="safetensors and torch required",
)


def _create_synthetic_safetensors_files(
    tmp_path: Path, num_layers: int = 4
) -> dict[str, str]:
    """Create actual safetensors files with known tensor values."""
    import torch
    from safetensors.torch import save_file

    weight_map: dict[str, str] = {}
    hidden_size = 16

    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}

    # Embedding — shard 1
    shard_file = "model-00001-of-00002.safetensors"
    embed_tensor = torch.randn(100, hidden_size, dtype=torch.float32)
    if shard_file not in shard_tensors:
        shard_tensors[shard_file] = {}
    shard_tensors[shard_file]["model.embed_tokens.weight"] = embed_tensor
    weight_map["model.embed_tokens.weight"] = shard_file

    # Layer tensors
    layer_suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "mlp.gate_proj.weight",
        "input_layernorm.weight",
    ]

    for layer_idx in range(num_layers):
        shard_num = 1 if layer_idx < num_layers // 2 else 2
        shard_file = f"model-{shard_num:05d}-of-00002.safetensors"
        if shard_file not in shard_tensors:
            shard_tensors[shard_file] = {}

        for suffix in layer_suffixes:
            tensor_name = f"model.layers.{layer_idx}.{suffix}"
            if "layernorm" in suffix:
                tensor = torch.randn(hidden_size, dtype=torch.float32)
            else:
                tensor = torch.randn(
                    hidden_size, hidden_size, dtype=torch.float32
                )
            shard_tensors[shard_file][tensor_name] = tensor
            weight_map[tensor_name] = shard_file

    # Final norm and lm_head — shard 2
    shard_file = "model-00002-of-00002.safetensors"
    if shard_file not in shard_tensors:
        shard_tensors[shard_file] = {}
    shard_tensors[shard_file]["model.norm.weight"] = torch.randn(
        hidden_size, dtype=torch.float32
    )
    weight_map["model.norm.weight"] = shard_file
    shard_tensors[shard_file]["lm_head.weight"] = torch.randn(
        100, hidden_size, dtype=torch.float32
    )
    weight_map["lm_head.weight"] = shard_file

    # Write safetensors files
    for filename, tensors in shard_tensors.items():
        save_file(tensors, str(tmp_path / filename))

    # Write the index file
    index_data = {
        "metadata": {"total_size": 1000000},
        "weight_map": weight_map,
    }
    index_path = tmp_path / "model.safetensors.index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f)

    return weight_map


@_requires_safetensors
class TestMeasureLoadingMemory:
    """Test measure_loading_memory returns shard and metrics."""

    def test_returns_shard_and_metrics(self, tmp_path: Path) -> None:
        """measure_loading_memory returns a tuple of (shard, metrics)."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)

        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )

        # Create a minimal model config
        model_config = MagicMock()
        model_config.hidden_size = 16
        model_config.vocab_size = 100
        model_config.num_hidden_layers = 4
        model_config.layer_types = [
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
        ]
        model_config.head_dim = None
        model_config.num_attention_heads = None

        result = measure_loading_memory(
            model_path=tmp_path,
            rank=0,
            layer_distribution=distribution,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        shard, metrics = result
        assert shard is not None
        assert isinstance(metrics, LoadingMemoryMetrics)

    def test_metrics_have_positive_peak_rss(self, tmp_path: Path) -> None:
        """Peak RSS is positive (process always uses some memory)."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)

        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )

        model_config = MagicMock()
        model_config.hidden_size = 16
        model_config.vocab_size = 100
        model_config.num_hidden_layers = 4
        model_config.layer_types = [
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
        ]
        model_config.head_dim = None
        model_config.num_attention_heads = None

        _, metrics = measure_loading_memory(
            model_path=tmp_path,
            rank=0,
            layer_distribution=distribution,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        assert metrics.peak_rss_bytes > 0
        assert metrics.peak_rss_mib > 0.0

    def test_metrics_have_positive_duration(self, tmp_path: Path) -> None:
        """Loading duration is positive (loading takes nonzero time)."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)

        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )

        model_config = MagicMock()
        model_config.hidden_size = 16
        model_config.vocab_size = 100
        model_config.num_hidden_layers = 4
        model_config.layer_types = [
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
        ]
        model_config.head_dim = None
        model_config.num_attention_heads = None

        _, metrics = measure_loading_memory(
            model_path=tmp_path,
            rank=0,
            layer_distribution=distribution,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        assert metrics.loading_duration_seconds > 0.0

    def test_rss_before_is_positive(self, tmp_path: Path) -> None:
        """RSS before loading is positive (process uses memory at startup)."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)

        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )

        model_config = MagicMock()
        model_config.hidden_size = 16
        model_config.vocab_size = 100
        model_config.num_hidden_layers = 4
        model_config.layer_types = [
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
        ]
        model_config.head_dim = None
        model_config.num_attention_heads = None

        _, metrics = measure_loading_memory(
            model_path=tmp_path,
            rank=0,
            layer_distribution=distribution,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        assert metrics.rss_before_bytes > 0

    def test_rss_after_gte_rss_before(self, tmp_path: Path) -> None:
        """RSS after loading is >= RSS before (peak is monotonic)."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)

        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )

        model_config = MagicMock()
        model_config.hidden_size = 16
        model_config.vocab_size = 100
        model_config.num_hidden_layers = 4
        model_config.layer_types = [
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
        ]
        model_config.head_dim = None
        model_config.num_attention_heads = None

        _, metrics = measure_loading_memory(
            model_path=tmp_path,
            rank=0,
            layer_distribution=distribution,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        # ru_maxrss is cumulative peak, so after >= before
        assert metrics.rss_after_bytes >= metrics.rss_before_bytes
