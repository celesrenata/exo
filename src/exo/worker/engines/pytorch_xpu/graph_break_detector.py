"""Graph break detection and logging for torch.compile tracing.

Detects graph breaks introduced during ``torch.compile()`` tracing by
inspecting ``torch._dynamo.utils.counters["graph_break"]`` before and after
compilation. When new graph breaks are detected, each break's location and
reason is logged at WARNING level.

**Validates: Requirement 3.4**
"""

from __future__ import annotations

from typing import final

from loguru import logger

_logger = logger.bind(module="pytorch_xpu.graph_break_detector")


@final
class GraphBreakReport:
    """Immutable report of graph breaks detected during a compilation pass.

    Attributes:
        breaks: Mapping of break reason to the count of occurrences.
        total_count: Total number of new graph breaks detected.
    """

    __slots__ = ("breaks", "total_count")

    def __init__(self, breaks: dict[str, int], total_count: int) -> None:
        self.breaks = breaks
        self.total_count = total_count


def _get_graph_break_counters() -> dict[str, int]:
    """Snapshot the current graph break counters from torch._dynamo.

    Returns an empty dict if torch._dynamo is unavailable (e.g., older
    PyTorch or CPU-only build without dynamo).
    """
    try:
        import torch._dynamo.utils as dynamo_utils

        counters = dynamo_utils.counters["graph_break"]
        # counters is a defaultdict(int) — copy to freeze the snapshot
        return dict(counters)
    except (ImportError, AttributeError, KeyError):
        return {}


def snapshot_graph_break_counters() -> dict[str, int]:
    """Take a snapshot of graph break counters before compilation.

    Call this immediately before invoking ``torch.compile()`` or triggering
    the first compiled invocation. The returned snapshot is passed to
    ``detect_graph_breaks`` after compilation completes.

    Returns:
        A frozen copy of the current graph break counter state.
    """
    return _get_graph_break_counters()


def detect_graph_breaks(before_snapshot: dict[str, int]) -> GraphBreakReport:
    """Detect and log graph breaks that occurred since the snapshot.

    Compares the current ``torch._dynamo.utils.counters["graph_break"]``
    state against the provided ``before_snapshot``. Any new or incremented
    counters represent graph breaks introduced during the compilation pass.

    Each detected break is logged at WARNING level with its reason and count.

    Args:
        before_snapshot: Counter state captured via ``snapshot_graph_break_counters()``
            before compilation.

    Returns:
        A ``GraphBreakReport`` containing the new breaks and total count.

    **Validates: Requirement 3.4**
    """
    after_snapshot = _get_graph_break_counters()

    new_breaks: dict[str, int] = {}
    for reason, count in after_snapshot.items():
        previous_count = before_snapshot.get(reason, 0)
        delta = count - previous_count
        if delta > 0:
            new_breaks[reason] = delta

    total_count = sum(new_breaks.values())

    if total_count > 0:
        _logger.warning(
            "Graph breaks detected during compilation: {count} total",
            count=total_count,
        )
        for reason, count in new_breaks.items():
            _logger.warning(
                "  Graph break: reason={reason}, occurrences={count}",
                reason=reason,
                count=count,
            )

    return GraphBreakReport(breaks=new_breaks, total_count=total_count)


def reset_graph_break_counters() -> None:
    """Clear all graph break counters.

    Useful for isolating graph break detection to a specific compilation
    pass without interference from prior compilations in the same process.
    """
    try:
        import torch._dynamo.utils as dynamo_utils

        dynamo_utils.counters["graph_break"].clear()
    except (ImportError, AttributeError, KeyError):
        pass
