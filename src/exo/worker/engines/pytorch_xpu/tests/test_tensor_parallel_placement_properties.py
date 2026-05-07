"""Property-based tests for tensor-parallel placement module.

Feature: tensor-parallelism-xpu, Property 5: Symmetric Tensor-Parallel Group Validation

**Validates: Requirements 8.4**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.worker.engines.pytorch_xpu.tensor_parallel_placement import (
    validate_symmetric_groups,
)


# Strategy: generate random lists of group sizes (1-10 groups, sizes 1-16)
group_sizes_strategy = st.lists(
    st.integers(min_value=1, max_value=16),
    min_size=0,
    max_size=10,
)


@given(group_sizes=group_sizes_strategy)
@settings(max_examples=200)
def test_symmetric_group_validation_property(group_sizes: list[int]) -> None:
    """Property 5: Symmetric Tensor-Parallel Group Validation.

    For any set of proposed tensor-parallel groups, the placement validator
    SHALL accept the configuration if and only if all groups have the same
    number of nodes. For any configuration where group sizes differ, the
    validator SHALL reject it.

    **Validates: Requirements 8.4**
    """
    result = validate_symmetric_groups(group_sizes)

    if not group_sizes:
        # Empty list is vacuously symmetric
        assert result is True, "Empty group list should be accepted (vacuously true)"
    else:
        # Check if all sizes are equal
        all_equal = all(size == group_sizes[0] for size in group_sizes)
        assert result == all_equal, (
            f"validate_symmetric_groups({group_sizes}) returned {result}, "
            f"but all_equal={all_equal}"
        )


@given(
    uniform_size=st.integers(min_value=1, max_value=16),
    num_groups=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_uniform_groups_always_accepted(uniform_size: int, num_groups: int) -> None:
    """All-equal group sizes should always be accepted.

    **Validates: Requirements 8.4**
    """
    group_sizes = [uniform_size] * num_groups
    assert validate_symmetric_groups(group_sizes) is True, (
        f"Uniform groups {group_sizes} should be accepted"
    )


@given(
    base_size=st.integers(min_value=1, max_value=15),
    num_groups=st.integers(min_value=2, max_value=10),
    diff_index=st.integers(min_value=0, max_value=9),
)
@settings(max_examples=100)
def test_non_uniform_groups_always_rejected(
    base_size: int, num_groups: int, diff_index: int
) -> None:
    """Groups with at least one different size should always be rejected.

    **Validates: Requirements 8.4**
    """
    # Create a list with one element different from the rest
    group_sizes = [base_size] * num_groups
    actual_index = diff_index % num_groups
    # Make one element different (add 1, guaranteed to differ)
    group_sizes[actual_index] = base_size + 1

    assert validate_symmetric_groups(group_sizes) is False, (
        f"Non-uniform groups {group_sizes} should be rejected"
    )
