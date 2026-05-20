"""
Async output streamer for decoupling token output from the decode loop.

Provides an asyncio-based queue between the decode loop and the API response
stream, with backpressure support to prevent unbounded memory growth when the
consumer (API layer) is slower than the producer (decode loop).

The streamer is single-threaded async (not multi-threaded) — it relies on
asyncio cooperative scheduling rather than thread locks.
"""

from __future__ import annotations

import asyncio
from typing import final


@final
class AsyncOutputStreamer:
    """Async queue between decode loop and API response stream.

    Decouples token production (decode loop) from token consumption (API
    response streaming / tokenizer decode). The decode loop calls
    ``put_token()`` synchronously after each forward pass, and the API
    consumer awaits ``get_token()`` to receive tokens for output.

    Backpressure is managed via ``should_pause`` and ``should_resume``
    properties. When the pending token count reaches ``max_pending``, the
    caller should pause production. When it drops below
    ``resume_threshold`` after a pause, production should resume.

    The queue is unbounded (``maxsize=0``) — backpressure is advisory,
    enforced by the caller checking ``should_pause`` after each
    ``put_token()`` call.
    """

    __slots__ = (
        "_queue",
        "_pending",
        "_max_pending",
        "_resume_threshold",
        "_paused",
    )

    def __init__(
        self, max_pending: int = 32, resume_threshold: int = 16
    ) -> None:
        """Initialize the async output streamer.

        Args:
            max_pending: Maximum pending tokens before backpressure
                activates. Defaults to 32.
            resume_threshold: Queue depth at which generation resumes
                after backpressure. Defaults to 16.
        """
        self._queue: asyncio.Queue[int | None] = asyncio.Queue(maxsize=0)
        self._pending: int = 0
        self._max_pending: int = max_pending
        self._resume_threshold: int = resume_threshold
        self._paused: bool = False

    def put_token(self, token_id: int) -> bool:
        """Enqueue a token for output.

        Non-blocking — uses ``put_nowait()`` on the unbounded queue.
        Increments the pending counter and checks backpressure.

        Args:
            token_id: The integer token ID to enqueue.

        Returns:
            True if the caller should continue producing tokens.
            False if backpressure is now active (caller should pause).
        """
        self._queue.put_nowait(token_id)
        self._pending += 1
        if self.should_pause:
            self._paused = True
            return False
        return True

    async def get_token(self) -> int | None:
        """Get next token for API output.

        Awaits the queue for the next token. Decrements the pending
        counter and clears the paused state if ``should_resume`` is True.

        Returns:
            The integer token ID, or None signaling end of generation.
        """
        token = await self._queue.get()
        if token is not None:
            self._pending -= 1
        else:
            # Sentinel value — don't decrement pending for None
            pass
        if self.should_resume:
            self._paused = False
        return token

    def signal_end(self) -> None:
        """Signal end of generation by putting None (sentinel) in the queue.

        Non-blocking. The consumer will receive None from ``get_token()``
        to indicate that no more tokens will be produced.
        """
        self._queue.put_nowait(None)

    @property
    def should_pause(self) -> bool:
        """Whether the decode loop should pause (queue >= max_pending)."""
        return self._pending >= self._max_pending

    @property
    def should_resume(self) -> bool:
        """Whether the decode loop should resume (queue < resume_threshold after pause)."""
        return self._paused and self._pending < self._resume_threshold

    @property
    def pending(self) -> int:
        """Current number of tokens in the queue awaiting consumption."""
        return self._pending

    @property
    def paused(self) -> bool:
        """Whether the streamer is currently in a paused (backpressure) state."""
        return self._paused
