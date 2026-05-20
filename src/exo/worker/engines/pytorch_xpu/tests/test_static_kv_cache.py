"""Unit tests for StaticKVCache.

Validates core functionality:
- Pre-allocation at construction (Req 2.1)
- In-place update without reallocation (Req 2.2)
- Position tracking (Req 2.3)
- Reset for reuse (Req 2.4)
- GatedDeltaNet state support (Req 2.5)
"""

from __future__ import annotations

import torch

from exo.worker.engines.pytorch_xpu.static_kv_cache import (
    GatedDeltaNetLayerConfig,
    StaticKVCache,
)


class TestStaticKVCacheConstruction:
    """Tests for cache pre-allocation at construction."""

    def test_creates_key_value_caches_for_each_layer(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=8,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=2048,
            device=torch.device("cpu"),
        )
        assert cache.num_attention_layers == 8
        assert cache.max_seq_len == 2048

    def test_initial_position_is_zero(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=4,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=128,
            device=torch.device("cpu"),
        )
        assert cache.position == 0

    def test_cache_tensors_have_correct_shape(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
        )
        key = cache.get_key_cache(0)
        value = cache.get_value_cache(0)
        assert key.shape == (1, 64, 4, 80)
        assert value.shape == (1, 64, 4, 80)

    def test_cache_tensors_are_on_specified_device(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
        )
        assert cache.get_key_cache(0).device == torch.device("cpu")
        assert cache.get_value_cache(0).device == torch.device("cpu")

    def test_cache_tensors_use_specified_dtype(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert cache.get_key_cache(0).dtype == torch.float32
        assert cache.dtype == torch.float32

    def test_default_dtype_is_bfloat16(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
        )
        assert cache.dtype == torch.bfloat16


class TestStaticKVCacheUpdate:
    """Tests for in-place cache update."""

    def test_update_writes_key_value_at_position(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        key = torch.ones(1, 1, 4, 80)
        value = torch.ones(1, 1, 4, 80) * 2.0

        keys_out, values_out = cache.update(0, key, value)

        assert keys_out.shape == (1, 1, 4, 80)
        assert values_out.shape == (1, 1, 4, 80)
        assert torch.allclose(keys_out[:, 0:1, :, :], key)
        assert torch.allclose(values_out[:, 0:1, :, :], value)

    def test_update_preserves_data_ptr(self) -> None:
        """data_ptr() must remain constant across updates (Property 3)."""
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        key_ptr_before = cache.get_key_cache(0).data_ptr()
        value_ptr_before = cache.get_value_cache(0).data_ptr()

        key = torch.randn(1, 1, 4, 80)
        value = torch.randn(1, 1, 4, 80)
        _keys_out, _values_out = cache.update(0, key, value)

        assert cache.get_key_cache(0).data_ptr() == key_ptr_before
        assert cache.get_value_cache(0).data_ptr() == value_ptr_before

    def test_multiple_updates_accumulate(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=1,
            num_kv_heads=2,
            head_dim=4,
            max_seq_len=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # First decode step
        key1 = torch.ones(1, 1, 2, 4) * 1.0
        value1 = torch.ones(1, 1, 2, 4) * 10.0
        keys_out, _values_out = cache.update(0, key1, value1)
        cache.advance_position()

        assert keys_out.shape == (1, 1, 2, 4)
        assert cache.position == 1

        # Second decode step
        key2 = torch.ones(1, 1, 2, 4) * 2.0
        value2 = torch.ones(1, 1, 2, 4) * 20.0
        keys_out, _values_out2 = cache.update(0, key2, value2)
        cache.advance_position()

        assert keys_out.shape == (1, 2, 2, 4)
        assert cache.position == 2
        # Verify both entries are present
        assert torch.allclose(keys_out[:, 0:1, :, :], key1)
        assert torch.allclose(keys_out[:, 1:2, :, :], key2)

    def test_update_raises_on_invalid_layer_idx(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
        )
        key = torch.randn(1, 1, 4, 80, dtype=torch.bfloat16)
        value = torch.randn(1, 1, 4, 80, dtype=torch.bfloat16)

        try:
            cache.update(5, key, value)
            assert False, "Should have raised IndexError"
        except IndexError:
            pass

    def test_update_raises_when_position_at_max(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=1,
            num_kv_heads=2,
            head_dim=4,
            max_seq_len=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        key = torch.randn(1, 1, 2, 4)
        value = torch.randn(1, 1, 2, 4)

        # Fill to max
        cache.update(0, key, value)
        cache.advance_position()
        cache.update(0, key, value)
        cache.advance_position()

        # Next update should raise
        try:
            cache.update(0, key, value)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass


class TestStaticKVCacheReset:
    """Tests for cache reset."""

    def test_reset_sets_position_to_zero(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        key = torch.randn(1, 1, 4, 80)
        value = torch.randn(1, 1, 4, 80)
        cache.update(0, key, value)
        cache.advance_position()
        assert cache.position == 1

        cache.reset()
        assert cache.position == 0

    def test_reset_preserves_data_ptr(self) -> None:
        """After reset, same tensor storage is reused (no deallocation)."""
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        key_ptr = cache.get_key_cache(0).data_ptr()
        value_ptr = cache.get_value_cache(0).data_ptr()

        key = torch.randn(1, 1, 4, 80)
        value = torch.randn(1, 1, 4, 80)
        cache.update(0, key, value)
        cache.advance_position()

        cache.reset()

        assert cache.get_key_cache(0).data_ptr() == key_ptr
        assert cache.get_value_cache(0).data_ptr() == value_ptr

    def test_reset_zeros_cache_contents(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=1,
            num_kv_heads=2,
            head_dim=4,
            max_seq_len=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        key = torch.ones(1, 1, 2, 4)
        value = torch.ones(1, 1, 2, 4)
        cache.update(0, key, value)
        cache.advance_position()

        cache.reset()

        # All values should be zero after reset
        assert torch.all(cache.get_key_cache(0) == 0)
        assert torch.all(cache.get_value_cache(0) == 0)

    def test_cache_reusable_after_reset(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=1,
            num_kv_heads=2,
            head_dim=4,
            max_seq_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        key = torch.ones(1, 1, 2, 4)
        value = torch.ones(1, 1, 2, 4)

        # Fill partially
        cache.update(0, key, value)
        cache.advance_position()
        cache.update(0, key * 2, value * 2)
        cache.advance_position()

        # Reset and reuse
        cache.reset()
        assert cache.position == 0

        new_key = torch.ones(1, 1, 2, 4) * 3.0
        new_value = torch.ones(1, 1, 2, 4) * 30.0
        keys_out, _values_out = cache.update(0, new_key, new_value)
        cache.advance_position()

        assert cache.position == 1
        assert torch.allclose(keys_out[:, 0:1, :, :], new_key)


class TestStaticKVCacheGatedDeltaNet:
    """Tests for GatedDeltaNet state support."""

    def test_creates_gated_deltanet_slots(self) -> None:
        configs = [
            GatedDeltaNetLayerConfig(conv_size=4, hidden_size=2560, num_heads=32, head_dim=128),
            GatedDeltaNetLayerConfig(conv_size=4, hidden_size=2560, num_heads=32, head_dim=128),
        ]
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
            gated_deltanet_configs=configs,
        )
        assert cache.num_gated_deltanet_layers == 2

    def test_gated_deltanet_slot_shapes(self) -> None:
        configs = [
            GatedDeltaNetLayerConfig(conv_size=4, hidden_size=2560, num_heads=32, head_dim=128),
        ]
        cache = StaticKVCache(
            num_attention_layers=1,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
            gated_deltanet_configs=configs,
        )
        slot = cache.get_gated_deltanet_slot(0)
        assert slot.conv_state.shape == (1, 4, 2560)
        assert slot.recurrent_state.shape == (1, 32, 128, 128)

    def test_gated_deltanet_recurrent_state_is_fp32(self) -> None:
        configs = [
            GatedDeltaNetLayerConfig(conv_size=4, hidden_size=2560, num_heads=32, head_dim=128),
        ]
        cache = StaticKVCache(
            num_attention_layers=1,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
            gated_deltanet_configs=configs,
        )
        slot = cache.get_gated_deltanet_slot(0)
        assert slot.recurrent_state.dtype == torch.float32

    def test_gated_deltanet_slot_preserves_data_ptr_after_reset(self) -> None:
        configs = [
            GatedDeltaNetLayerConfig(conv_size=4, hidden_size=256, num_heads=4, head_dim=64),
        ]
        cache = StaticKVCache(
            num_attention_layers=1,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
            gated_deltanet_configs=configs,
        )
        slot = cache.get_gated_deltanet_slot(0)
        conv_ptr = slot.conv_state.data_ptr()
        recurrent_ptr = slot.recurrent_state.data_ptr()

        # Modify state
        slot.conv_state.fill_(1.0)
        slot.recurrent_state.fill_(2.0)

        # Reset
        cache.reset()

        # data_ptr unchanged, values zeroed
        assert slot.conv_state.data_ptr() == conv_ptr
        assert slot.recurrent_state.data_ptr() == recurrent_ptr
        assert torch.all(slot.conv_state == 0)
        assert torch.all(slot.recurrent_state == 0)

    def test_gated_deltanet_slot_raises_on_invalid_index(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
        )
        try:
            cache.get_gated_deltanet_slot(0)
            assert False, "Should have raised IndexError"
        except IndexError:
            pass

    def test_no_gated_deltanet_slots_by_default(self) -> None:
        cache = StaticKVCache(
            num_attention_layers=2,
            num_kv_heads=4,
            head_dim=80,
            max_seq_len=64,
            device=torch.device("cpu"),
        )
        assert cache.num_gated_deltanet_layers == 0
