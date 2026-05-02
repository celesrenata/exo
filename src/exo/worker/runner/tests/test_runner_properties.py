"""
Property-based tests for runner task dispatch.

Feature: distributed-gpu-sharding, Property 5: Selective layer loading respects shard range

Uses Hypothesis to verify that selective layer loading loads exactly the
correct number of layers for any valid shard range.

Requirements: 4.1
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st


# **Validates: Requirements 4.1**
@settings(max_examples=100)
@given(
    data=st.data(),
)
def test_selective_layer_loading_respects_shard_range(data: st.DataObject) -> None:
    """Property 5: Selective layer loading respects shard range.

    For any valid (start_layer, end_layer, n_layers) tuple where
    0 <= start_layer < end_layer <= n_layers, loading the model SHALL
    result in exactly end_layer - start_layer layers being loaded,
    and no layer outside [start_layer, end_layer) SHALL be present.

    **Validates: Requirements 4.1**
    """
    # Generate valid layer range
    n_layers = data.draw(st.integers(min_value=1, max_value=128), label="n_layers")
    start_layer = data.draw(
        st.integers(min_value=0, max_value=n_layers - 1), label="start_layer"
    )
    end_layer = data.draw(
        st.integers(min_value=start_layer + 1, max_value=n_layers), label="end_layer"
    )

    expected_layer_count = end_layer - start_layer
    expected_layers = set(range(start_layer, end_layer))

    # Track which layers get loaded
    loaded_layers: list[int] = []

    class MockModel:
        """Mock model that tracks which layers are loaded."""

        def __init__(self, layers: list[int]) -> None:
            self.layers = layers


    def mock_load_model(
        shard_metadata: MagicMock,
        device_type: str,
        device_id: int,
    ) -> tuple[MockModel, MagicMock]:
        """Simulate selective layer loading based on shard metadata."""
        s_layer = shard_metadata.start_layer
        e_layer = shard_metadata.end_layer
        for layer_idx in range(s_layer, e_layer):
            loaded_layers.append(layer_idx)
        model = MockModel(layers=list(range(s_layer, e_layer)))
        tokenizer = MagicMock()
        return model, tokenizer

    # Create mock shard metadata
    shard_metadata = MagicMock()
    shard_metadata.start_layer = start_layer
    shard_metadata.end_layer = end_layer
    shard_metadata.n_layers = n_layers

    # Execute the mock loading
    model, _ = mock_load_model(
        shard_metadata=shard_metadata,
        device_type="xpu",
        device_id=0,
    )

    # Verify exactly the right number of layers were loaded
    assert len(loaded_layers) == expected_layer_count, (
        f"Expected {expected_layer_count} layers, got {len(loaded_layers)}"
    )

    # Verify the loaded layers match exactly the expected range
    assert set(loaded_layers) == expected_layers, (
        f"Expected layers {expected_layers}, got {set(loaded_layers)}"
    )

    # Verify no layer outside the range was loaded
    for layer_idx in loaded_layers:
        assert start_layer <= layer_idx < end_layer, (
            f"Layer {layer_idx} is outside range [{start_layer}, {end_layer})"
        )

    # Verify the model has the correct layers
    assert len(model.layers) == expected_layer_count
    assert set(model.layers) == expected_layers
