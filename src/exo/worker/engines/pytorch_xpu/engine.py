"""PyTorch XPU Engine — implements the Engine protocol for the upstream runner.

Handles warmup, task submission, and step-based generation. For multi-node
(world_size > 1), uses tensor_parallel_generator. For single-node, uses the
local generator.

Gloo backend only. All broadcast tensors must be on CPU.
Device is xpu:0 (Intel Arc iGPU via torch.xpu).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from exo.shared.types.chunks import Chunk, ErrorChunk, TokenChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.runner.bootstrap import logger


@dataclass(eq=False)
class PyTorchXPUEngine(Engine):
    model: Any
    tokenizer: Any
    rank: int
    world_size: int
    device: str
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: tuple[TextGeneration, Generator[Any]] | None = field(
        default=None, init=False
    )

    def warmup(self) -> None:
        """Run a dummy forward pass to warm up the model."""
        import torch

        from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
            PipelineParallelShard,
        )

        logger.info(f"PyTorchXPUEngine.warmup: device={self.device}, rank={self.rank}")

        try:
            with torch.no_grad():
                if isinstance(self.model, PipelineParallelShard):
                    if self.model.config.is_first_stage:
                        # First stage: expects token IDs
                        dummy_input = torch.tensor(
                            [[1, 2, 3]], dtype=torch.long, device=self.device
                        )
                    else:
                        # Non-first stage: expects hidden_state
                        dummy_input = torch.randn(
                            1,
                            3,
                            self.model.config.hidden_size,
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                    _output, _kv = self.model.forward(input_data=dummy_input)
                    # Clear KV cache from dummy forward pass
                    self.model.reset_state()
                elif hasattr(self.model, "forward"):
                    # TensorParallelShard or TransformerShard
                    dummy_input = torch.tensor(
                        [[1, 2, 3]], dtype=torch.long, device=self.device
                    )
                    _logits, _kv = self.model.forward(
                        input_data=dummy_input,
                        past_key_values=None,
                    )
                    # Task 4.2: Call reset after warmup when the model supports it
                    if hasattr(self.model, "reset_generation_state"):
                        self.model.reset_generation_state(reason="warmup_complete")
                elif hasattr(self.model, "__call__"):
                    # Standard HuggingFace model
                    dummy_input = torch.tensor(
                        [[1, 2, 3]], dtype=torch.long, device=self.device
                    )
                    self.model(input_ids=dummy_input, use_cache=False)

            logger.info("PyTorchXPUEngine.warmup: complete")
        except Exception as e:
            logger.warning(f"PyTorchXPUEngine.warmup failed (non-fatal): {e}")

    def submit(self, task: GenerationTask) -> None:
        """Queue a generation task."""
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._queue.append(task)

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        """Execute one generation step, yielding (task_id, Chunk) tuples."""
        # Drain cancellations
        for task_id in self.cancel_receiver.collect():
            self._cancelled_tasks.add(task_id)

        output: list[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]] = []

        # Emit cancellations
        for task_id in list(self._cancelled_tasks):
            output.append((task_id, CancelledResponse()))
        self._cancelled_tasks.clear()

        # If no active generation, start next from queue
        if self._active is None:
            if not self._queue:
                return output
            task = self._queue.popleft()

            if self.should_cancel(task.task_id):
                output.append((task.task_id, CancelledResponse()))
                return output

            # Reset KV cache from previous request before starting new generation
            if hasattr(self.model, "reset_state"):
                self.model.reset_state()

            gen = self._build_generator(task)
            self._active = (task, gen)

        assert self._active is not None
        task, gen = self._active

        try:
            response = next(gen)

            # Convert GenerationResponse to TokenChunk
            model_id = ModelId(str(task.task_params.model))

            if response.finish_reason == "error":
                chunk: Chunk = ErrorChunk(
                    model=model_id,
                    error_message=response.text,
                )
            else:
                chunk = TokenChunk(
                    model=model_id,
                    text=response.text,
                    token_id=response.token,
                    usage=response.usage,
                    finish_reason=response.finish_reason,  # type: ignore[arg-type]
                    stats=response.stats,
                    logprob=response.logprob,
                    top_logprobs=response.top_logprobs,
                )

            # Only rank 0 emits chunks to avoid duplicates
            if self.rank == 0:
                output.append((task.task_id, chunk))

            # If finished, clean up
            if response.finish_reason is not None:
                # Task 4.5: Reset generation state on completion
                if hasattr(self.model, "reset_generation_state"):
                    self.model.reset_generation_state(reason="generation_complete")
                output.append((task.task_id, FinishedResponse()))
                self._active = None

        except StopIteration:
            # Task 4.5: Reset generation state on cancellation/stop
            if hasattr(self.model, "reset_generation_state"):
                self.model.reset_generation_state(reason="generation_stopped")
            output.append((task.task_id, FinishedResponse()))
            self._active = None

        except Exception as e:
            logger.error(f"PyTorchXPUEngine.step error: {e}", exc_info=True)
            # Task 4.5: Reset generation state on error
            if hasattr(self.model, "reset_generation_state"):
                self.model.reset_generation_state(reason="generation_error")
            model_id = ModelId(str(task.task_params.model))
            output.append(
                (
                    task.task_id,
                    ErrorChunk(model=model_id, error_message=str(e)),
                )
            )
            output.append((task.task_id, FinishedResponse()))
            self._active = None

        return output

    def _build_generator(self, task: TextGeneration) -> Generator[Any]:
        """Build the appropriate generator for this task."""
        import torch

        from exo.worker.engines.pytorch_xpu.generator import pytorch_xpu_generate
        from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
            PipelineParallelShard,
        )

        # Extract prompt from task params
        prompt = ""
        if task.task_params.input:
            prompt = task.task_params.input[0].content or ""

        # Build messages list from task params, including system prompt if present
        instructions = getattr(task.task_params, "instructions", None)
        messages: list[dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": str(instructions)})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template if tokenizer supports it.
        # enable_thinking=False tells Qwen3.5 to skip the <think>...</think> block.
        # Without this, the model spends all max_tokens on internal reasoning
        # that gets stripped by skip_special_tokens=True, producing no visible output.
        enable_thinking: bool = bool(getattr(task.task_params, "enable_thinking", None) or False)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    # Tokenizer doesn't support enable_thinking — fall back without it
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            except Exception:
                pass  # Fall back to raw prompt

        # Log the final prompt suffix for debugging
        from exo.worker.runner.bootstrap import logger as _runner_logger  # pyright: ignore[reportAny]
        _runner_logger.info(
            f"PyTorchXPUEngine._build_generator: enable_thinking={enable_thinking} "
            f"prompt_tail={repr(prompt[-120:]) if isinstance(prompt, str) else repr(prompt)}"
        )

        max_tokens = task.task_params.max_output_tokens or 100
        temperature = task.task_params.temperature or 1.0
        top_k = getattr(task.task_params, "top_k", None)
        top_p = task.task_params.top_p

        # Pipeline-parallel dispatch (check BEFORE tensor-parallel)
        if isinstance(self.model, PipelineParallelShard):
            from exo.worker.engines.pytorch_xpu.pipeline_generator import (
                pipeline_parallel_generate,
                pipeline_parallel_worker_loop,
            )

            if self.rank == 0:
                return pipeline_parallel_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            else:
                return self._pipeline_worker_loop_generator()

        if self.world_size > 1 and self.rank == 0:
            # Multi-node: use tensor parallel generator
            from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
                tensor_parallel_generate,
            )

            return tensor_parallel_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        elif self.world_size > 1 and self.rank != 0:
            # Non-rank-0 in multi-node: run worker loop as a generator
            return self._worker_loop_generator()
        else:
            # Single-node: use local generator
            device_parts = self.device.split(":")
            device_type = device_parts[0]
            device_id = int(device_parts[1]) if len(device_parts) > 1 else 0

            return pytorch_xpu_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device_type=device_type,
                device_id=device_id,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

    def _worker_loop_generator(self) -> Generator[Any]:
        """Wrap the tensor_parallel_worker_loop as a generator for non-rank-0 nodes.

        The worker loop is blocking, so we run it and yield a single
        FinishedResponse when it completes.
        """
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            tensor_parallel_worker_loop,
        )

        tensor_parallel_worker_loop(
            model=self.model,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Worker loop completed — yield nothing (StopIteration will signal finish)
        return
        yield  # Make this a generator function  # noqa: RET503

    def _pipeline_worker_loop_generator(self) -> Generator[Any]:
        """Wrap pipeline_parallel_worker_loop as a generator for non-rank-0 nodes.

        The pipeline worker loop is blocking — it receives activations from the
        previous stage, forwards through local layers, and sends to the next stage
        (or samples and broadcasts if last stage). When the loop exits (EOS or
        termination signal), StopIteration signals the engine to finish.
        """
        from exo.worker.engines.pytorch_xpu.pipeline_generator import (
            pipeline_parallel_worker_loop,
        )

        pipeline_parallel_worker_loop(
            model=self.model,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
        )

        return
        yield  # Make this a generator function  # noqa: RET503

    def close(self) -> None:
        """Destroy process group and free model."""
        self._active = None
        self._queue.clear()

        try:
            import torch.distributed as dist

            if dist.is_initialized():
                from exo.worker.engines.pytorch_xpu.distributed import (
                    destroy_process_group,
                )

                destroy_process_group()
        except Exception as e:
            logger.warning(f"PyTorchXPUEngine.close: process group cleanup error: {e}")

        try:
            del self.model
        except Exception:
            pass

        logger.info("PyTorchXPUEngine.close: cleanup complete")

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        """Serve a prefill request. Not implemented for XPU engine."""
        raise NotImplementedError(
            "serve_prefill is not implemented for PyTorchXPUEngine"
        )
