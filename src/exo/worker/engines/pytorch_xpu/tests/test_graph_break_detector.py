"""Tests for graph break detection and logging.

Validates: Requirement 3.4 — IF a Graph_Break is detected during compilation,
THEN THE system SHALL log the break location and reason at WARNING level.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from unittest.mock import patch

import pytest

from exo.worker.engines.pytorch_xpu.graph_break_detector import (
    GraphBreakReport,
    detect_graph_breaks,
    reset_graph_break_counters,
    snapshot_graph_break_counters,
)


@pytest.fixture
def mock_dynamo_counters() -> defaultdict[str, int]:
    """Provide a mock counters dict that behaves like torch._dynamo.utils.counters["graph_break"]."""
    return defaultdict(int)


class TestSnapshotGraphBreakCounters:
    """Tests for snapshot_graph_break_counters."""

    def test_returns_empty_dict_when_dynamo_unavailable(self) -> None:
        """When torch._dynamo is not importable, returns empty dict."""
        with patch(
            "exo.worker.engines.pytorch_xpu.graph_break_detector._get_graph_break_counters",
            return_value={},
        ):
            result = snapshot_graph_break_counters()
            assert result == {}

    def test_returns_copy_of_current_counters(self, mock_dynamo_counters: defaultdict[str, int]) -> None:
        """Returns a frozen copy of the current counter state."""
        mock_dynamo_counters["unsupported_op"] = 3
        mock_dynamo_counters["data_dependent_control_flow"] = 1

        mock_counters: dict[str, Any] = {"graph_break": mock_dynamo_counters}

        with patch(
            "torch._dynamo.utils.counters",
            mock_counters,
        ):
            from exo.worker.engines.pytorch_xpu.graph_break_detector import (
                _get_graph_break_counters,  # pyright: ignore[reportPrivateUsage]
            )

            result = _get_graph_break_counters()
            assert result == {"unsupported_op": 3, "data_dependent_control_flow": 1}

    def test_snapshot_is_independent_of_later_mutations(self, mock_dynamo_counters: defaultdict[str, int]) -> None:
        """Snapshot is a copy — later mutations to counters do not affect it."""
        mock_dynamo_counters["some_break"] = 2

        with patch(
            "exo.worker.engines.pytorch_xpu.graph_break_detector._get_graph_break_counters",
            return_value=dict(mock_dynamo_counters),
        ):
            snapshot = snapshot_graph_break_counters()

        # Mutate the original
        mock_dynamo_counters["some_break"] = 99
        assert snapshot == {"some_break": 2}


class TestDetectGraphBreaks:
    """Tests for detect_graph_breaks."""

    def test_no_breaks_detected_when_counters_unchanged(self) -> None:
        """When counters are the same before and after, no breaks reported."""
        before = {"existing_break": 5}

        with patch(
            "exo.worker.engines.pytorch_xpu.graph_break_detector._get_graph_break_counters",
            return_value={"existing_break": 5},
        ):
            report = detect_graph_breaks(before)

        assert report.total_count == 0
        assert report.breaks == {}

    def test_detects_new_break_reasons(self) -> None:
        """New break reasons that appear after compilation are detected."""
        before: dict[str, int] = {}

        with patch(
            "exo.worker.engines.pytorch_xpu.graph_break_detector._get_graph_break_counters",
            return_value={"unsupported_builtin": 2, "data_dependent_guard": 1},
        ):
            report = detect_graph_breaks(before)

        assert report.total_count == 3
        assert report.breaks == {"unsupported_builtin": 2, "data_dependent_guard": 1}

    def test_detects_incremented_existing_breaks(self) -> None:
        """Incremented counts on existing break reasons are detected."""
        before = {"unsupported_op": 3}

        with patch(
            "exo.worker.engines.pytorch_xpu.graph_break_detector._get_graph_break_counters",
            return_value={"unsupported_op": 5},
        ):
            report = detect_graph_breaks(before)

        assert report.total_count == 2
        assert report.breaks == {"unsupported_op": 2}

    def test_ignores_unchanged_counters(self) -> None:
        """Counters that did not change are not included in the report."""
        before = {"old_break": 10, "another_old": 5}

        with patch(
            "exo.worker.engines.pytorch_xpu.graph_break_detector._get_graph_break_counters",
            return_value={"old_break": 10, "another_old": 5, "new_break": 1},
        ):
            report = detect_graph_breaks(before)

        assert report.total_count == 1
        assert report.breaks == {"new_break": 1}

    def test_logs_warning_when_breaks_detected(self) -> None:
        """WARNING-level log messages are emitted for each detected break."""
        before: dict[str, int] = {}
        warning_messages: list[str] = []

        def capture_warning(msg: str, **kwargs: str | int) -> None:
            warning_messages.append(msg)

        with patch(
            "exo.worker.engines.pytorch_xpu.graph_break_detector._get_graph_break_counters",
            return_value={"call_function_unimplemented": 3},
        ):
            with patch(
                "exo.worker.engines.pytorch_xpu.graph_break_detector._logger"
            ) as mock_logger:
                mock_logger.warning = capture_warning
                report = detect_graph_breaks(before)

                # Verify WARNING-level logging occurred: summary + per-break
                assert len(warning_messages) == 2
                assert "Graph breaks detected" in warning_messages[0]
                assert "Graph break" in warning_messages[1]

        assert report.total_count == 3

    def test_no_logging_when_no_breaks(self) -> None:
        """No log messages when no graph breaks are detected."""
        before = {"existing": 5}
        warning_messages: list[str] = []

        def capture_warning(msg: str, **kwargs: str | int) -> None:
            warning_messages.append(msg)

        with patch(
            "exo.worker.engines.pytorch_xpu.graph_break_detector._get_graph_break_counters",
            return_value={"existing": 5},
        ):
            with patch(
                "exo.worker.engines.pytorch_xpu.graph_break_detector._logger"
            ) as mock_logger:
                mock_logger.warning = capture_warning
                detect_graph_breaks(before)
                assert len(warning_messages) == 0

    def test_returns_graph_break_report_type(self) -> None:
        """Return value is a GraphBreakReport instance."""
        before: dict[str, int] = {}

        with patch(
            "exo.worker.engines.pytorch_xpu.graph_break_detector._get_graph_break_counters",
            return_value={},
        ):
            report = detect_graph_breaks(before)

        assert isinstance(report, GraphBreakReport)
        assert report.total_count == 0
        assert report.breaks == {}


class TestResetGraphBreakCounters:
    """Tests for reset_graph_break_counters."""

    def test_clears_counters(self) -> None:
        """Calling reset clears the graph break counters via torch._dynamo."""
        mock_graph_break_dict: defaultdict[str, int] = defaultdict(int)
        mock_graph_break_dict["some_break"] = 5

        mock_counters: dict[str, Any] = {"graph_break": mock_graph_break_dict}

        with patch("torch._dynamo.utils.counters", mock_counters):
            reset_graph_break_counters()

        assert len(mock_graph_break_dict) == 0

    def test_handles_missing_dynamo_gracefully(self) -> None:
        """Does not raise when torch._dynamo is unavailable."""
        with patch.dict("sys.modules", {"torch._dynamo.utils": None, "torch._dynamo": None}):
            # Should not raise
            reset_graph_break_counters()


class TestGraphBreakReport:
    """Tests for GraphBreakReport dataclass."""

    def test_attributes(self) -> None:
        """Report exposes breaks dict and total_count."""
        report = GraphBreakReport(breaks={"reason_a": 2, "reason_b": 1}, total_count=3)
        assert report.breaks == {"reason_a": 2, "reason_b": 1}
        assert report.total_count == 3

    def test_empty_report(self) -> None:
        """Empty report has zero count and empty breaks."""
        report = GraphBreakReport(breaks={}, total_count=0)
        assert report.breaks == {}
        assert report.total_count == 0
