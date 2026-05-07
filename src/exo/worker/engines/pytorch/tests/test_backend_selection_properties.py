"""Property-based tests for distributed backend selection logic.

Uses Hypothesis to verify that select_backend_for_devices() follows the
device-type rules: "nccl" if and only if ALL devices are "cuda", "gloo"
otherwise (any XPU or mixed).

**Validates: Requirements 6.1, 6.2, 6.3**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.worker.engines.pytorch.distributed.transport import (
    select_backend_for_devices,
)


# --- Strategies ---

# Individual device type
device_type_st = st.sampled_from(["cuda", "xpu"])

# Non-empty list of device types (length 1 to 8)
device_list_st = st.lists(device_type_st, min_size=1, max_size=8)

# All-CUDA lists (length 1 to 8)
all_cuda_st = st.lists(st.just("cuda"), min_size=1, max_size=8)

# Lists containing at least one XPU device (length 1 to 8)
has_xpu_st = st.lists(device_type_st, min_size=1, max_size=8).filter(
    lambda lst: "xpu" in lst
)


class TestBackendSelectionProperty:
    """Property 5: Distributed backend selection follows device-type rules.

    *For any* non-empty list of device types (each being "cuda" or "xpu"),
    the backend selection function SHALL return "nccl" if and only if ALL
    device types are "cuda", and SHALL return "gloo" otherwise (i.e., when
    any device is "xpu" or the list is mixed).

    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    @settings(max_examples=100)
    @given(device_types=all_cuda_st)
    def test_all_cuda_returns_nccl(
        self, device_types: list[str]
    ) -> None:
        """All-CUDA list → "nccl".

        Requirement 6.2: WHEN all participating nodes use CUDA_Device,
        THE Distributed_Communication SHALL use NCCL_Backend.
        """
        result = select_backend_for_devices(device_types)
        assert result == "nccl", (
            f"Expected 'nccl' for all-CUDA list {device_types}, got '{result}'"
        )

    @settings(max_examples=100)
    @given(device_types=has_xpu_st)
    def test_any_xpu_returns_gloo(
        self, device_types: list[str]
    ) -> None:
        """Any XPU in list → "gloo" (covers mixed and all-XPU cases).

        Requirement 6.1: WHEN all participating nodes use XPU_Device,
        THE Distributed_Communication SHALL use Gloo_Backend.
        Requirement 6.3: WHEN the cluster contains a mix of CUDA and XPU
        nodes, THE Distributed_Communication SHALL use Gloo_Backend.
        """
        result = select_backend_for_devices(device_types)
        assert result == "gloo", (
            f"Expected 'gloo' for list with XPU {device_types}, got '{result}'"
        )

    @settings(max_examples=100)
    @given(device_types=device_list_st)
    def test_biconditional_nccl_iff_all_cuda(
        self, device_types: list[str]
    ) -> None:
        """nccl ↔ all CUDA; gloo ↔ any XPU (biconditional property).

        This is the core property: the function returns "nccl" if and only
        if ALL device types are "cuda", and "gloo" otherwise.

        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        result = select_backend_for_devices(device_types)
        all_cuda = all(dt == "cuda" for dt in device_types)

        if all_cuda:
            assert result == "nccl", (
                f"All devices are CUDA but got '{result}' instead of 'nccl'"
            )
        else:
            assert result == "gloo", (
                f"Not all devices are CUDA but got '{result}' instead of 'gloo'"
            )

    @settings(max_examples=100)
    @given(device_types=device_list_st)
    def test_single_cuda_returns_nccl(
        self, device_types: list[str]
    ) -> None:
        """Single CUDA → "nccl", Single XPU → "gloo".

        Verifies the property holds for single-element lists as well as
        multi-element lists.
        """
        # Test with a single-element subcase derived from the generated list
        single = [device_types[0]]
        result = select_backend_for_devices(single)

        if single[0] == "cuda":
            assert result == "nccl", (
                f"Single CUDA device should return 'nccl', got '{result}'"
            )
        else:
            assert result == "gloo", (
                f"Single XPU device should return 'gloo', got '{result}'"
            )

    @settings(max_examples=100)
    @given(device_types=device_list_st)
    def test_result_is_valid_backend(
        self, device_types: list[str]
    ) -> None:
        """Result is always one of the two valid backends.

        The function must always return either "gloo" or "nccl" — never
        anything else.
        """
        result = select_backend_for_devices(device_types)
        assert result in ("gloo", "nccl"), (
            f"Invalid backend '{result}' — must be 'gloo' or 'nccl'"
        )
