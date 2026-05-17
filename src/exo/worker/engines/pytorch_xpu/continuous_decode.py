"""
Continuous decode loop — step-based orchestration for continuous batching.

Provides a synchronous, testable interface for driving the continuous decode
loop. The pipeline generator calls ``ContinuousDecodeStep`` to determine what
action to take next (prefill, decode, or idle), and reports results back after
each pipeline pass.

This module does NOT perform inference or distributed communication. It
coordinates the ContinuousBatchingEngine to determine batch composition and
route token results.

The async ``run_continuous_decode(...)`` function will integrate this with the
distributed pipeline in a later task.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 3.9, 3.10, 3.11
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, final

from exo.worker.engines.pytorch_xpu.continuous_batching import (
    DecodeMicrobatch,
    TokenResultBatch,
)
from exo.worker.engines.pytorch_xpu.continuous_batching_engine import (
    ContinuousBatchingEngine,
)


# ---------------------------------------------------------------------------
# ContinuousDecodeAction — what the pipeline should do next
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class ContinuousDecodeAction:
    """Describes the next action the pipeline should take.

    Returned by ``ContinuousDecodeStep.get_next_action()`` to tell the
    pipeline generator what to execute next.

    - ``"prefill"``: Run prefill for a new request. ``prefill_request_id``
      and ``prefill_tokens`` are populated.
    - ``"decode"``: Run a decode step for the active microbatch.
      ``decode_microbatch`` is populated.
    - ``"idle"``: No work available. The pipeline can sleep or poll.
    """

    action_type: Literal["prefill", "decode", "idle"]
    """Which action the pipeline should take."""

    prefill_request_id: str | None = None
    """Request ID to prefill (only set when action_type is 'prefill')."""

    prefill_tokens: list[int] | None = None
    """Prompt tokens for the prefill request (only set when action_type is 'prefill')."""

    decode_microbatch: DecodeMicrobatch | None = None
    """Microbatch for the decode step (only set when action_type is 'decode')."""


# ---------------------------------------------------------------------------
# ContinuousDecodeResult — summary of a continuous decode session
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class ContinuousDecodeResult:
    """Summary of a continuous decode session.

    Returned by ``run_continuous_decode(...)`` after the session ends
    (all requests complete, max steps reached, or explicit stop).
    """

    total_tokens_generated: int
    """Total number of tokens generated across all requests."""

    total_steps: int
    """Total number of decode steps executed."""

    completed_requests: list[str]
    """Request IDs that completed normally (reached max_tokens)."""

    cancelled_requests: list[str]
    """Request IDs that were cancelled during the session."""


# ---------------------------------------------------------------------------
# ContinuousDecodeStep — one step of the continuous decode loop
# ---------------------------------------------------------------------------


@final
class ContinuousDecodeStep:
    """Executes one step of the continuous decode loop.

    Each step:
    1. Gets the next action from the engine (prefill or decode)
    2. Returns a ``ContinuousDecodeAction`` for the pipeline to process
    3. After processing, the pipeline calls ``report_prefill_done()`` or
       ``report_decode_results()`` to advance state

    This class is synchronous and testable without distributed infrastructure.
    The async pipeline generator wraps this in its event loop.
    """

    def __init__(self, engine: ContinuousBatchingEngine) -> None:
        """Initialize with a reference to the continuous batching engine.

        Args:
            engine: The ContinuousBatchingEngine that manages request
                lifecycle and batch composition.
        """
        self._engine: ContinuousBatchingEngine = engine
        self._total_tokens_generated: int = 0
        self._total_steps: int = 0
        self._completed_requests: list[str] = []
        self._cancelled_requests: list[str] = []

    # -------------------------------------------------------------------
    # Public API: Action determination
    # -------------------------------------------------------------------

    def get_next_action(self) -> ContinuousDecodeAction:
        """Determine what the pipeline should do next.

        Priority order:
        1. If a request needs prefilling, return a prefill action.
        2. If requests are decode-ready, return a decode action.
        3. Otherwise, return idle.

        Returns:
            A ``ContinuousDecodeAction`` describing the next pipeline action.
        """
        # Check for prefill work first
        prefill_info = self._engine.get_prefill_request()
        if prefill_info is not None:
            request_id, prompt_tokens = prefill_info
            return ContinuousDecodeAction(
                action_type="prefill",
                prefill_request_id=request_id,
                prefill_tokens=prompt_tokens,
            )

        # Check for decode work
        microbatch = self._engine.get_decode_microbatch()
        if microbatch is not None:
            return ContinuousDecodeAction(
                action_type="decode",
                decode_microbatch=microbatch,
            )

        # No work available
        return ContinuousDecodeAction(action_type="idle")

    # -------------------------------------------------------------------
    # Public API: Result reporting
    # -------------------------------------------------------------------

    def report_prefill_done(self, request_id: str) -> None:
        """Report that a prefill has completed for a request.

        Moves the request from prefilling to decode-ready state in the
        engine. The next call to ``get_next_action()`` will include this
        request in the decode microbatch.

        Args:
            request_id: The request that completed prefill.
        """
        self._engine.mark_prefill_done(request_id)

    def report_decode_results(self, results: TokenResultBatch) -> list[str]:
        """Report decode results from a pipeline pass.

        Routes token results to the engine, which updates per-request
        state and detects completions.

        Args:
            results: Token results from the decode step.

        Returns:
            List of request IDs that completed in this step.
        """
        completed = self._engine.report_token_results(results)

        # Update session tracking
        self._total_tokens_generated += len(results.token_ids)
        self._total_steps += 1
        self._completed_requests.extend(completed)

        return completed

    def report_cancellation(self, request_id: str) -> None:
        """Report that a request was cancelled.

        Removes the request from the engine and tracks it in the session
        summary.

        Args:
            request_id: The request to cancel.
        """
        self._engine.cancel_request(request_id)
        self._cancelled_requests.append(request_id)

    # -------------------------------------------------------------------
    # Public API: Session state
    # -------------------------------------------------------------------

    @property
    def has_work(self) -> bool:
        """Whether the engine has any pending work."""
        return self._engine.has_work

    @property
    def session_result(self) -> ContinuousDecodeResult:
        """Get the current session summary.

        Can be called at any time to get a snapshot of progress.
        """
        return ContinuousDecodeResult(
            total_tokens_generated=self._total_tokens_generated,
            total_steps=self._total_steps,
            completed_requests=list(self._completed_requests),
            cancelled_requests=list(self._cancelled_requests),
        )


# ---------------------------------------------------------------------------
# run_continuous_decode_steps — synchronous multi-step driver
# ---------------------------------------------------------------------------


@final
@dataclass
class _StepContext:
    """Internal mutable context for the synchronous step driver."""

    steps_executed: int = 0
    tokens_generated: int = 0
    completed: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)


def run_continuous_decode_steps(
    *,
    step: ContinuousDecodeStep,
    execute_prefill: Callable[[str, list[int]], None] | None = None,
    execute_decode: Callable[[DecodeMicrobatch], None] | None = None,
    max_steps: int | None = None,
) -> ContinuousDecodeResult:
    """Run the continuous decode loop synchronously until completion.

    This is a test-friendly synchronous driver. The real async version
    will use distributed communication for actual inference.

    Args:
        step: The ContinuousDecodeStep instance to drive.
        execute_prefill: Callback invoked with (request_id, tokens) for
            prefill actions. Must call ``step.report_prefill_done()``.
            If None, prefill is auto-completed.
        execute_decode: Callback invoked with (microbatch,) for decode
            actions. Must call ``step.report_decode_results()``.
            If None, raises RuntimeError (decode requires external logic).
        max_steps: Maximum number of decode steps to execute. None means
            run until all requests complete.

    Returns:
        A ContinuousDecodeResult summarizing the session.
    """
    while step.has_work:
        if max_steps is not None and step._total_steps >= max_steps:
            break

        action = step.get_next_action()

        if action.action_type == "prefill":
            assert action.prefill_request_id is not None
            assert action.prefill_tokens is not None
            if execute_prefill is not None:
                execute_prefill(action.prefill_request_id, action.prefill_tokens)
            else:
                # Auto-complete prefill
                step.report_prefill_done(action.prefill_request_id)

        elif action.action_type == "decode":
            assert action.decode_microbatch is not None
            if execute_decode is not None:
                execute_decode(action.decode_microbatch)
            else:
                raise RuntimeError(
                    "Decode action received but no execute_decode callback "
                    "was provided"
                )

        elif action.action_type == "idle":
            # No work but has_work is True — this means requests are
            # in a transitional state. Break to avoid infinite loop.
            break

    return step.session_result
