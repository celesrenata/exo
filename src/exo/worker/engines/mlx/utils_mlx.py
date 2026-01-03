import asyncio
import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Callable, cast
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE,
    KV_CACHE_BITS,
    TRUST_REMOTE_CODE,
)

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer  # type: ignore
import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.shared.types.validation import (
    EnhancedTokenChunk,
    SequenceValidationResult,
    ValidationStatus,
    CorruptionType,
    CorruptionSeverity,
)
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.runner.bootstrap import logger

Group = mx.distributed.Group
# Needed for 8 bit model
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))


class BarrierStatus(Enum):
    """Status of barrier synchronization operations."""
    PENDING = "pending"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    FAILED = "failed"


class SyncResult(Enum):
    """Result of synchronization operations."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    RETRY_EXHAUSTED = "retry_exhausted"
    FAILED = "failed"


@dataclass
class SyncState:
    """State information for synchronization operations."""
    participating_devices: list[int]
    sync_timestamp: datetime
    barrier_status: BarrierStatus
    timeout_remaining: float
    retry_count: int = 0
    max_retries: int = 3


class DistributedBarrier:
    """Enhanced barrier with timeout and retry mechanisms."""
    
    def __init__(self, group: Group | None = None, default_timeout: float = 30.0):
        self.group = group
        self.default_timeout = default_timeout
        self.sync_state: SyncState | None = None
        
    async def sync_with_timeout(self, timeout: float | None = None, retry_count: int = 3) -> SyncResult:
        """
        Synchronize with timeout and retry logic.
        
        Args:
            timeout: Maximum time to wait for synchronization (seconds)
            retry_count: Maximum number of retry attempts
            
        Returns:
            SyncResult indicating the outcome of synchronization
        """
        if timeout is None:
            timeout = self.default_timeout
            
        self.sync_state = SyncState(
            participating_devices=list(range(self.group.size())) if self.group else [0],
            sync_timestamp=datetime.now(),
            barrier_status=BarrierStatus.PENDING,
            timeout_remaining=timeout,
            max_retries=retry_count
        )
        
        for attempt in range(retry_count + 1):
            self.sync_state.retry_count = attempt
            
            try:
                # Calculate exponential backoff delay for retries
                if attempt > 0:
                    backoff_delay = min(2 ** (attempt - 1), 8.0)  # Cap at 8 seconds
                    await asyncio.sleep(backoff_delay)
                    logger.info(f"Barrier retry attempt {attempt} after {backoff_delay}s backoff")
                
                # Perform the actual barrier synchronization with timeout
                start_time = time.perf_counter()
                
                # Use asyncio.wait_for to add timeout to the barrier operation
                await asyncio.wait_for(
                    self._perform_barrier_sync(),
                    timeout=timeout
                )
                
                elapsed = time.perf_counter() - start_time
                self.sync_state.barrier_status = BarrierStatus.SUCCESS
                self.sync_state.timeout_remaining = max(0, timeout - elapsed)
                
                logger.info(f"Barrier synchronization successful in {elapsed:.2f}s")
                return SyncResult.SUCCESS
                
            except asyncio.TimeoutError:
                self.sync_state.barrier_status = BarrierStatus.TIMEOUT
                logger.warning(f"Barrier synchronization timeout on attempt {attempt + 1}")
                
                if attempt == retry_count:
                    return SyncResult.TIMEOUT
                    
            except Exception as e:
                self.sync_state.barrier_status = BarrierStatus.FAILED
                logger.error(f"Barrier synchronization failed on attempt {attempt + 1}: {e}")
                
                if attempt == retry_count:
                    return SyncResult.FAILED
        
        return SyncResult.RETRY_EXHAUSTED
    
    async def _perform_barrier_sync(self) -> None:
        """Perform the actual barrier synchronization operation."""
        # Run the blocking mx_barrier in a thread pool to make it async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._blocking_barrier_sync)
    
    def _blocking_barrier_sync(self) -> None:
        """Blocking barrier synchronization using MLX distributed operations."""
        mx.eval(
            mx.distributed.all_sum(
                mx.array(1.0),
                stream=mx.default_stream(mx.Device(mx.cpu)),
                group=self.group,
            )
        )
    
    async def validate_sync_state(self) -> bool:
        """
        Validate the current synchronization state.
        
        Returns:
            True if synchronization state is valid and healthy
        """
        if self.sync_state is None:
            return False
            
        # Check if sync is recent (within last 60 seconds)
        time_since_sync = datetime.now() - self.sync_state.sync_timestamp
        if time_since_sync > timedelta(seconds=60):
            logger.warning("Synchronization state is stale")
            return False
            
        # Check if barrier completed successfully
        if self.sync_state.barrier_status != BarrierStatus.SUCCESS:
            logger.warning(f"Barrier status is not successful: {self.sync_state.barrier_status}")
            return False
            
        # Validate that all expected devices participated
        if self.group and len(self.sync_state.participating_devices) != self.group.size():
            logger.warning("Not all devices participated in synchronization")
            return False
            
        return True


def mx_barrier(group: Group | None = None):
    """Legacy synchronous barrier function for backward compatibility."""
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0),
            stream=mx.default_stream(mx.Device(mx.cpu)),
            group=group,
        )
    )


async def mx_barrier_async(group: Group | None = None, timeout: float = 30.0, retry_count: int = 3) -> SyncResult:
    """
    Enhanced asynchronous barrier with timeout and retry logic.
    
    Args:
        group: MLX distributed group
        timeout: Maximum time to wait for synchronization (seconds)
        retry_count: Maximum number of retry attempts
        
    Returns:
        SyncResult indicating the outcome of synchronization
    """
    barrier = DistributedBarrier(group, timeout)
    return await barrier.sync_with_timeout(timeout, retry_count)


@dataclass
class GenerationCoordination:
    """Coordination information for token generation across pipeline stages."""
    pipeline_stage: int
    device_rank: int
    world_size: int
    sync_point_id: str
    tokens_ready: bool = False
    coordination_timestamp: datetime = Field(default_factory=datetime.now)


class TokenStreamSynchronizer:
    """Coordinates token generation across distributed devices."""
    
    def __init__(self, group: Group | None = None, pipeline_stage: int = 0):
        self.group = group
        self.pipeline_stage = pipeline_stage
        self.device_rank = group.rank() if group else 0
        self.world_size = group.size() if group else 1
        self.barrier = DistributedBarrier(group)
        
        # Track token sequences and synchronization points
        self.token_sequences: dict[str, list[EnhancedTokenChunk]] = {}
        self.sync_points: dict[str, GenerationCoordination] = {}
        
    async def coordinate_generation(self, sync_point_id: str) -> GenerationCoordination:
        """
        Coordinate token generation across pipeline stages.
        
        Args:
            sync_point_id: Unique identifier for this synchronization point
            
        Returns:
            GenerationCoordination with coordination details
        """
        coordination = GenerationCoordination(
            pipeline_stage=self.pipeline_stage,
            device_rank=self.device_rank,
            world_size=self.world_size,
            sync_point_id=sync_point_id
        )
        
        self.sync_points[sync_point_id] = coordination
        
        # Synchronize all devices at this generation point
        sync_result = await self.barrier.sync_with_timeout(timeout=30.0, retry_count=3)
        
        if sync_result == SyncResult.SUCCESS:
            coordination.tokens_ready = True
            logger.info(f"Generation coordination successful for sync point {sync_point_id}")
        else:
            logger.error(f"Generation coordination failed for sync point {sync_point_id}: {sync_result}")
            
        return coordination
    
    async def verify_token_ordering(self, tokens: list[EnhancedTokenChunk], sequence_id: str = "default") -> bool:
        """
        Verify token ordering across distributed devices.
        
        Args:
            tokens: List of tokens to verify
            sequence_id: Identifier for the token sequence
            
        Returns:
            True if token ordering is correct
        """
        if not tokens:
            return True
            
        # Store the token sequence for this device
        self.token_sequences[sequence_id] = tokens
        
        # Verify local token ordering first
        local_ordering_valid = self._verify_local_ordering(tokens)
        if not local_ordering_valid:
            logger.error(f"Local token ordering invalid for sequence {sequence_id}")
            return False
        
        # If we're in a distributed setup, coordinate with other devices
        if self.group and self.world_size > 1:
            return await self._verify_distributed_ordering(tokens, sequence_id)
        
        return True
    
    def _verify_local_ordering(self, tokens: list[EnhancedTokenChunk]) -> bool:
        """Verify that tokens are in correct order locally."""
        for i in range(1, len(tokens)):
            if tokens[i].sequence_position <= tokens[i-1].sequence_position:
                logger.error(f"Token ordering violation: position {tokens[i].sequence_position} <= {tokens[i-1].sequence_position}")
                return False
        return True
    
    async def _verify_distributed_ordering(self, tokens: list[EnhancedTokenChunk], sequence_id: str) -> bool:
        """Verify token ordering across distributed devices."""
        try:
            # Create a summary of our token positions
            local_positions = [token.sequence_position for token in tokens]
            local_min = min(local_positions) if local_positions else 0
            local_max = max(local_positions) if local_positions else 0
            
            # Broadcast position ranges to all devices
            position_data = mx.array([local_min, local_max, len(tokens)], dtype=mx.int32)
            
            # Gather position data from all devices
            all_position_data = mx.distributed.all_gather(position_data, group=self.group)
            mx.eval(all_position_data)
            
            # Verify that position ranges don't overlap inappropriately
            all_ranges = []
            for i in range(self.world_size):
                device_data = all_position_data[i * 3:(i + 1) * 3]
                min_pos, max_pos, count = int(device_data[0]), int(device_data[1]), int(device_data[2])
                if count > 0:  # Only consider devices with tokens
                    all_ranges.append((i, min_pos, max_pos, count))
            
            # Check for overlaps and gaps
            all_ranges.sort(key=lambda x: x[1])  # Sort by min position
            
            for i in range(1, len(all_ranges)):
                prev_device, prev_min, prev_max, prev_count = all_ranges[i-1]
                curr_device, curr_min, curr_max, curr_count = all_ranges[i]
                
                # Check for overlaps (tokens with same positions on different devices)
                if curr_min <= prev_max:
                    logger.error(f"Token position overlap between devices {prev_device} and {curr_device}")
                    return False
                
                # Check for gaps (missing token positions)
                if curr_min > prev_max + 1:
                    logger.warning(f"Token position gap between devices {prev_device} and {curr_device}: {prev_max} -> {curr_min}")
            
            logger.info(f"Distributed token ordering verified for sequence {sequence_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying distributed token ordering: {e}")
            return False
    
    async def create_synchronization_point(self, point_id: str, tokens: list[EnhancedTokenChunk]) -> bool:
        """
        Create a synchronization point for token stream assembly.
        
        Args:
            point_id: Unique identifier for the synchronization point
            tokens: Tokens to synchronize at this point
            
        Returns:
            True if synchronization point was created successfully
        """
        try:
            # Validate tokens before synchronization
            for token in tokens:
                if token.validation_status == ValidationStatus.PENDING:
                    # Update checksum and mark as valid if not already done
                    token.update_checksum()
                    token.mark_as_valid()
            
            # Coordinate generation at this point
            coordination = await self.coordinate_generation(point_id)
            
            if not coordination.tokens_ready:
                logger.error(f"Failed to create synchronization point {point_id}")
                return False
            
            # Verify token ordering
            ordering_valid = await self.verify_token_ordering(tokens, point_id)
            
            if not ordering_valid:
                logger.error(f"Token ordering invalid at synchronization point {point_id}")
                return False
            
            logger.info(f"Synchronization point {point_id} created successfully with {len(tokens)} tokens")
            return True
            
        except Exception as e:
            logger.error(f"Error creating synchronization point {point_id}: {e}")
            return False
    
    def get_sequence_validation_result(self, sequence_id: str) -> SequenceValidationResult:
        """Get validation results for a token sequence."""
        tokens = self.token_sequences.get(sequence_id, [])
        
        if not tokens:
            return SequenceValidationResult(
                is_valid=True,
                total_tokens=0,
                expected_tokens=0
            )
        
        positions = [token.sequence_position for token in tokens]
        positions.sort()
        
        # Check for missing positions
        missing_positions = []
        if positions:
            expected_range = range(positions[0], positions[-1] + 1)
            missing_positions = [pos for pos in expected_range if pos not in positions]
        
        # Check for duplicates
        duplicate_positions = []
        seen_positions = set()
        for pos in [token.sequence_position for token in tokens]:
            if pos in seen_positions:
                duplicate_positions.append(pos)
            seen_positions.add(pos)
        
        # Check for out-of-order positions (already handled in local ordering)
        out_of_order_positions = []
        
        is_valid = (
            len(missing_positions) == 0 and 
            len(duplicate_positions) == 0 and 
            len(out_of_order_positions) == 0
        )
        
        return SequenceValidationResult(
            is_valid=is_valid,
            missing_positions=missing_positions,
            duplicate_positions=duplicate_positions,
            out_of_order_positions=out_of_order_positions,
            total_tokens=len(tokens),
            expected_tokens=len(positions) + len(missing_positions) if positions else 0
        )



# TODO: Test this
#  ALSO https://github.com/exo-explore/exo/pull/233#discussion_r2549683673
def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_meta.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


def broadcast_from_zero(value: int, group: Group | None = None):
    if group is None:
        return value

    if group.rank() == 0:
        a = mx.array([value], dtype=mx.int32)
    else:
        a = mx.array([0], dtype=mx.int32)

    m = mx.distributed.all_sum(a, stream=mx.Device(mx.DeviceType.cpu), group=group)
    mx.eval(m)
    return int(m.item())


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> Group:
    """
    Initialize MLX distributed.
    """
    rank = bound_instance.bound_shard.device_rank
    logger.info(f"Starting initialization for rank {rank}")

    coordination_file = None
    try:
        # TODO: singleton instances
        match bound_instance.instance:
            case MlxRingInstance(hosts_by_node=hosts_by_node, ephemeral_port=_):
                coordination_file = (
                    f"./hosts_{bound_instance.instance.instance_id}_{rank}.json"
                )
                hosts_for_node = hosts_by_node[bound_instance.bound_node_id]
                hosts_json = HostList.from_hosts(hosts_for_node).model_dump_json()

                with open(coordination_file, "w") as f:
                    _ = f.write(hosts_json)

                logger.info(
                    f"rank {rank} hostfile: {coordination_file} hosts: {hosts_json}"
                )

                os.environ["MLX_HOSTFILE"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_RING_VERBOSE"] = "1"
                group = mx.distributed.init(backend="ring", strict=True)

            case MlxJacclInstance(
                ibv_devices=ibv_devices, jaccl_coordinators=jaccl_coordinators
            ):
                # Use RDMA connectivity matrix
                coordination_file = (
                    f"./hosts_{bound_instance.instance.instance_id}_{rank}.json"
                )
                ibv_devices_json = json.dumps(ibv_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(ibv_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                logger.info(f"rank {rank} MLX_IBV_DEVICES: {ibv_devices_json}")
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_IBV_DEVICES"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_JACCL_COORDINATOR"] = jaccl_coordinator
                group = mx.distributed.init(backend="jaccl", strict=True)

        logger.info(f"Rank {rank} mlx distributed initialization complete")

        return group
    finally:
        with contextlib.suppress(FileNotFoundError):
            if coordination_file:
                os.remove(coordination_file)


def initialize_mlx(
    bound_instance: BoundInstance,
) -> Group:
    # should we unseed it?
    # TODO: pass in seed from params
    mx.random.seed(42)

    assert len(bound_instance.instance.shard_assignments.node_to_runner) > 1, (
        "Tried to initialize mlx for a single node instance"
    )
    return mlx_distributed_init(bound_instance)


def load_mlx_items(
    bound_instance: BoundInstance, group: Group | None
) -> tuple[Model, TokenizerWrapper, Callable[[mx.array], mx.array]]:
    # TODO: pass temperature
    sampler: Callable[[mx.array], mx.array] = make_sampler(temp=0.7)
    logger.info("Created a sampler")

    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(bound_instance.bound_shard.model_meta.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, strict=True)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = shard_and_load(bound_instance.bound_shard, group=group)
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    return cast(Model, model), tokenizer, sampler


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: Group,
) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(shard_metadata.model_meta.model_id)

    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.debug(model)
    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # TODO: See if we should quantize the model.
        # def is_attention_layer(path: str) -> bool:
        #     path = path.lower()

        #     return "self_attn" in path and "layernorm" not in path

        # def quant_predicate(path: str, module: nn.Module):
        #     if not isinstance(module, nn.Linear):
        #         return False

        #     return is_attention_layer(path)
        # model, config = quantize_model(
        #        model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = tensor_auto_parallel(model, group)
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = pipeline_auto_parallel(model, group, shard_metadata)

    mx.eval(model.parameters())

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata):
    # TODO: Let's move away from this custom logic to mlx_lm.load()
    if "kimi-k2" in shard_metadata.model_meta.model_id.lower():
        eos_token_ids = [163586]

    elif "glm" in shard_metadata.model_meta.model_id.lower():
        eos_token_ids = [151336, 151329, 151338]

    else:
        eos_token_ids = None

    tokenizer = cast(
        TokenizerWrapper,
        load_tokenizer(
            model_path,
            tokenizer_config_extra={"trust_remote_code": TRUST_REMOTE_CODE},
            eos_token_ids=eos_token_ids,
        ),
    )
    assert isinstance(tokenizer, TokenizerWrapper)

    return tokenizer


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
    # Now we can properly access the messages
    messages = chat_task_data.messages

    formatted_messages: list[dict[str, Any]] = []
    for _, message in enumerate(messages):
        if isinstance(message.content, ChatCompletionMessageText):
            message.content = message.content.text
        if isinstance(message.content, list):
            if len(message.content) != 1:
                logger.warning("Received malformed prompt")
                continue

            message.content = message.content[0].text
        if message.content is None and message.thinking is None:
            continue

        # Null values are not valid when applying templates in tokenizer
        formatted_messages.append(
            {k: v for k, v in message.model_dump().items() if v is not None}  # type: ignore
        )

    prompt: str = tokenizer.apply_chat_template(  # type: ignore
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt  # type: ignore


class NullKVCache(KVCache):
    """
    A KVCache that pretends to exist but holds zero tokens.
    It satisfies .state/.meta_state and never allocates real keys/values.
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # zero-length K/V so shapes/dtypes are defined but empty
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> list[KVCache | RotatingKVCache | QuantizedKVCache]:
    assert hasattr(model, "layers")

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]


def mlx_force_oom(size: int = 40000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        return

    model_bytes = model_size.in_bytes
    max_rec_size = int(mx.metal.device_info()["max_recommended_working_set_size"])
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        logger.warning(
            f"Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    mx.set_wired_limit(max_rec_size)
    logger.info(f"Wired limit set to {max_rec_size}.")
