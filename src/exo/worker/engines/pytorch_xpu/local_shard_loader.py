"""
Local-shard model loading for pipeline-parallel inference.

Provides safetensors index parsing, tensor-name ownership resolution,
and local tensor manifest creation for Qwen3.5 pipeline-parallel loading.

Each rank loads only the tensors required by its assigned pipeline stage,
avoiding full-model materialization on every node.

Requirements addressed:
- 1.1: Each rank loads only tensors required by its assigned pipeline stage
- 1.3: Loading reads tensors directly from safetensors files by tensor name
- 1.4: Rank ownership is deterministic and configurable
- 1.5: Default four-stage ownership
- 1.9: Loader exposes a manifest of loaded tensor names for diagnostics
- 1.10: Missing, duplicate, or unassigned layer tensors produce clear startup errors
"""

from __future__ import annotations

import json
import logging
import platform
import re
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, final

from pydantic import BaseModel, ConfigDict

from exo.worker.engines.pytorch_xpu.pipeline_config import (
    PipelineLayerDistribution,
    PipelineStageAssignment,
)

if TYPE_CHECKING:
    import torch

    from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
        PipelineParallelShard,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tensor name pattern for Qwen3.5 layer tensors
# ---------------------------------------------------------------------------

_LAYER_TENSOR_PATTERN: re.Pattern[str] = re.compile(
    r"^model\.layers\.(\d+)\."
)


# ---------------------------------------------------------------------------
# LocalShardManifest — describes what tensors a rank owns
# ---------------------------------------------------------------------------


@final
class LocalShardManifest(BaseModel):
    """
    Immutable manifest describing which tensors a rank owns and where they live.

    Created by the local shard loader after parsing the safetensors index
    and resolving tensor ownership for a specific pipeline stage assignment.
    """

    model_config = ConfigDict(frozen=True)

    tensor_names: list[str]
    """Names of tensors owned by this rank, in sorted order."""

    tensor_to_file: dict[str, str]
    """Mapping from tensor name to safetensors shard filename."""

    stage_assignment: PipelineStageAssignment
    """The pipeline stage assignment used to resolve ownership."""

    total_tensor_count: int
    """Total number of tensors in the full model (for validation)."""

    local_tensor_count: int
    """Number of tensors owned by this rank."""


# ---------------------------------------------------------------------------
# parse_safetensors_index — parse model.safetensors.index.json
# ---------------------------------------------------------------------------


def parse_safetensors_index(model_path: Path) -> dict[str, str]:
    """Parse the safetensors index file and return the weight_map.

    The weight_map maps tensor names to their containing shard filenames.

    Args:
        model_path: Path to the model directory containing
            ``model.safetensors.index.json``.

    Returns:
        Dictionary mapping tensor name to shard filename.

    Raises:
        FileNotFoundError: If the index file does not exist.
        ValueError: If the index file is malformed or missing weight_map.
    """
    index_path = model_path / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Safetensors index not found at {index_path}"
        )

    with open(index_path, "r", encoding="utf-8") as f:
        try:
            index_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse safetensors index at {index_path}: {e}"
            ) from e

    if not isinstance(index_data, dict):
        raise ValueError(
            f"Safetensors index at {index_path} is not a JSON object"
        )

    weight_map = index_data.get("weight_map")
    if weight_map is None:
        raise ValueError(
            f"Safetensors index at {index_path} has no 'weight_map' key"
        )

    if not isinstance(weight_map, dict):
        raise ValueError(
            f"'weight_map' in safetensors index at {index_path} is not a dict"
        )

    # Validate all values are strings
    for tensor_name, shard_file in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(shard_file, str):
            raise ValueError(
                f"Invalid weight_map entry: {tensor_name!r} -> {shard_file!r}"
            )

    logger.debug(
        f"Parsed safetensors index: {len(weight_map)} tensors across "
        f"{len(set(weight_map.values()))} shard files"
    )

    return weight_map


# ---------------------------------------------------------------------------
# resolve_tensor_ownership — determine which tensors belong to a rank
# ---------------------------------------------------------------------------


def resolve_tensor_ownership(
    weight_map: dict[str, str],
    stage_assignment: PipelineStageAssignment,
) -> list[str]:
    """Determine which tensors from the weight map belong to this rank.

    Ownership rules for Qwen3.5:
    - Rank that owns_embedding gets: ``model.embed_tokens.weight``
    - Each rank gets tensors matching ``model.layers.{i}.*`` where
      ``i`` is in ``[start_layer, end_layer)``
    - Rank that owns_lm_head gets: ``model.norm.weight``, ``lm_head.weight``
    - Tied weights: if ``lm_head.weight`` is tied to ``model.embed_tokens.weight``
      (same shard file entry), the last rank still needs it

    Args:
        weight_map: Full tensor name to shard file mapping from the index.
        stage_assignment: This rank's pipeline stage assignment.

    Returns:
        Sorted list of tensor names owned by this rank.
    """
    owned_tensors: list[str] = []

    for tensor_name in weight_map:
        if _tensor_belongs_to_stage(tensor_name, stage_assignment):
            owned_tensors.append(tensor_name)

    owned_tensors.sort()
    return owned_tensors


def _tensor_belongs_to_stage(
    tensor_name: str,
    stage_assignment: PipelineStageAssignment,
) -> bool:
    """Check if a tensor belongs to the given pipeline stage.

    Args:
        tensor_name: Full tensor name from the weight map.
        stage_assignment: This rank's pipeline stage assignment.

    Returns:
        True if this rank owns the tensor.
    """
    # Embedding tensor — owned by the rank with owns_embedding
    if tensor_name == "model.embed_tokens.weight":
        return stage_assignment.owns_embedding

    # Final normalization — owned by the rank with owns_lm_head
    if tensor_name == "model.norm.weight":
        return stage_assignment.owns_lm_head

    # Language model head — owned by the rank with owns_lm_head
    if tensor_name == "lm_head.weight":
        return stage_assignment.owns_lm_head

    # Layer tensors — owned by the rank whose range includes the layer index
    layer_match = _LAYER_TENSOR_PATTERN.match(tensor_name)
    if layer_match is not None:
        layer_index = int(layer_match.group(1))
        return (
            stage_assignment.start_layer
            <= layer_index
            < stage_assignment.end_layer
        )

    # Unknown tensor pattern — not owned by any stage
    # This handles metadata tensors or unexpected entries gracefully
    logger.debug(
        f"Tensor '{tensor_name}' does not match any ownership rule, "
        f"skipping for rank {stage_assignment.rank}"
    )
    return False


# ---------------------------------------------------------------------------
# load_qwen_local_shard_from_safetensors — orchestrator function
# ---------------------------------------------------------------------------


def load_qwen_local_shard_from_safetensors(
    *,
    model_path: Path,
    stage_assignment: PipelineStageAssignment,
) -> LocalShardManifest:
    """Create a local tensor manifest for a pipeline stage.

    Orchestrates safetensors index parsing, tensor ownership resolution,
    and manifest creation. Does NOT load actual tensor data — only creates
    the manifest describing what to load.

    Args:
        model_path: Path to the model directory containing the safetensors
            index file and shard files.
        stage_assignment: This rank's pipeline stage assignment describing
            which layers and auxiliary modules it owns.

    Returns:
        A LocalShardManifest describing the tensors this rank needs to load.

    Raises:
        FileNotFoundError: If the safetensors index file is missing.
        ValueError: If the index is malformed or no tensors are assigned.
    """
    # Step 1: Parse the safetensors index
    weight_map = parse_safetensors_index(model_path)

    # Step 2: Resolve tensor ownership for this rank
    owned_tensor_names = resolve_tensor_ownership(weight_map, stage_assignment)

    if not owned_tensor_names:
        raise ValueError(
            f"No tensors assigned to rank {stage_assignment.rank} "
            f"with layer range [{stage_assignment.start_layer}, "
            f"{stage_assignment.end_layer}). "
            f"Check the stage assignment and model index."
        )

    # Step 3: Build the tensor-to-file mapping for owned tensors only
    tensor_to_file: dict[str, str] = {
        name: weight_map[name] for name in owned_tensor_names
    }

    # Step 4: Create the manifest
    manifest = LocalShardManifest(
        tensor_names=owned_tensor_names,
        tensor_to_file=tensor_to_file,
        stage_assignment=stage_assignment,
        total_tensor_count=len(weight_map),
        local_tensor_count=len(owned_tensor_names),
    )

    logger.info(
        f"Created local shard manifest for rank {stage_assignment.rank}: "
        f"{manifest.local_tensor_count}/{manifest.total_tensor_count} tensors, "
        f"layers [{stage_assignment.start_layer}, {stage_assignment.end_layer}), "
        f"shard files: {sorted(set(tensor_to_file.values()))}"
    )

    return manifest


# ---------------------------------------------------------------------------
# load_tensors_from_manifest — load actual tensor data from safetensors
# ---------------------------------------------------------------------------


def load_tensors_from_manifest(
    *,
    manifest: LocalShardManifest,
    model_path: Path,
    device: "torch.device",
    dtype: "torch.dtype | None" = None,
) -> "dict[str, torch.Tensor]":
    """Load only the tensors listed in the manifest from safetensors shard files.

    Uses ``safetensors.safe_open`` to load individual tensors by name without
    materializing the full file into memory. Each rank loads ONLY its owned
    tensors — no full model materialization.

    This function does NOT use ``transformers.AutoModel.from_pretrained()`` or
    any HuggingFace full-model loading path.

    Args:
        manifest: A LocalShardManifest describing which tensors to load and
            which shard files they reside in.
        model_path: Path to the model directory containing the safetensors
            shard files referenced by the manifest.
        device: Target torch device to place loaded tensors on.
        dtype: Target dtype for loaded tensors. Defaults to ``torch.bfloat16``
            if not specified.

    Returns:
        Dictionary mapping tensor names to loaded tensors on the target device.

    Raises:
        FileNotFoundError: If a referenced shard file does not exist.
        RuntimeError: If a tensor listed in the manifest is not found in its
            expected shard file.
    """
    import torch
    from safetensors import safe_open

    if dtype is None:
        dtype = torch.bfloat16

    start_time = time.perf_counter()
    loaded_tensors: dict[str, torch.Tensor] = {}
    total_bytes: int = 0

    # Group tensors by shard file for efficient file access
    file_to_tensors: dict[str, list[str]] = {}
    for tensor_name, shard_file in manifest.tensor_to_file.items():
        if shard_file not in file_to_tensors:
            file_to_tensors[shard_file] = []
        file_to_tensors[shard_file].append(tensor_name)

    # Load tensors from each shard file
    for shard_file, tensor_names in file_to_tensors.items():
        shard_path = model_path / shard_file

        if not shard_path.exists():
            raise FileNotFoundError(
                f"Shard file not found: {shard_path} "
                f"(referenced by {len(tensor_names)} tensors)"
            )

        # Use safe_open with framework="pt" to load individual tensors by name
        # without loading the entire file into memory
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            available_keys = set(f.keys())
            for tensor_name in tensor_names:
                if tensor_name not in available_keys:
                    raise RuntimeError(
                        f"Tensor '{tensor_name}' not found in shard file "
                        f"'{shard_file}'. Available keys: "
                        f"{sorted(available_keys)[:10]}..."
                    )

                # Load tensor from safetensors (loads to CPU first)
                tensor: torch.Tensor = f.get_tensor(tensor_name)

                # Cast to target dtype and move to target device
                tensor = tensor.to(dtype=dtype, device=device)

                loaded_tensors[tensor_name] = tensor
                total_bytes += tensor.nelement() * tensor.element_size()

        logger.debug(
            f"Loaded {len(tensor_names)} tensors from {shard_file}"
        )

    elapsed = time.perf_counter() - start_time
    total_mb = total_bytes / (1024 * 1024)

    logger.info(
        f"Loaded {len(loaded_tensors)} tensors "
        f"({total_mb:.1f} MiB) in {elapsed:.2f}s "
        f"({total_mb / elapsed:.1f} MiB/s) "
        f"for rank {manifest.stage_assignment.rank}"
    )

    return loaded_tensors


# ---------------------------------------------------------------------------
# validate_loaded_tensors — verify all expected tensors were loaded
# ---------------------------------------------------------------------------


def validate_loaded_tensors(
    manifest: LocalShardManifest,
    loaded_tensors: "dict[str, torch.Tensor]",
) -> None:
    """Validate that all expected tensors were loaded and none are missing.

    Compares the tensor names in the manifest against the keys in the loaded
    tensor dictionary. Raises a clear error identifying any missing tensors.

    Args:
        manifest: The LocalShardManifest describing expected tensors.
        loaded_tensors: Dictionary of actually loaded tensors.

    Raises:
        ValueError: If any tensor from the manifest is missing from the
            loaded dict, with a message listing all missing tensor names.
    """
    expected = set(manifest.tensor_names)
    actual = set(loaded_tensors.keys())

    missing = expected - actual
    if missing:
        missing_sorted = sorted(missing)
        raise ValueError(
            f"Missing {len(missing_sorted)} tensors for rank "
            f"{manifest.stage_assignment.rank}: "
            f"{missing_sorted[:10]}"
            f"{'...' if len(missing_sorted) > 10 else ''}"
        )

    logger.debug(
        f"Validation passed: all {len(expected)} tensors loaded "
        f"for rank {manifest.stage_assignment.rank}"
    )


# ---------------------------------------------------------------------------
# LocalShardModules — built modules for a local pipeline stage
# ---------------------------------------------------------------------------


@final
class LocalShardModules(BaseModel):
    """
    Built PyTorch modules for a local pipeline stage.

    Contains only the layers and auxiliary modules assigned to this rank,
    with global layer indices preserved for correct rotary position encoding
    and cache key mapping.

    This is the bridge between raw loaded tensors and the PipelineParallelShard
    class that drives inference.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    layers: "torch.nn.ModuleList"
    """Only the local transformer layers for this rank."""

    embed_tokens: "torch.nn.Embedding | None"
    """Token embedding module — only present on rank 0."""

    lm_head: "torch.nn.Linear | None"
    """Language model head — only present on last rank."""

    final_norm: "torch.nn.Module | None"
    """Final RMSNorm — only present on last rank."""

    rotary_emb: "torch.nn.Module | None"
    """Rotary embedding module — present on all ranks for position encoding."""

    layer_types: list[str]
    """Layer type per local layer: 'full_attention' or 'linear_attention'."""

    global_layer_indices: list[int]
    """Global layer indices for position encoding and cache keys."""


# ---------------------------------------------------------------------------
# LayerBuilder protocol — dependency injection for layer instantiation
# ---------------------------------------------------------------------------


class LayerBuilder(Protocol):
    """Protocol for building a single transformer layer from loaded tensors.

    Implementations receive the layer type, global index, relevant tensors,
    device, dtype, and model config, and return an nn.Module.
    """

    def __call__(
        self,
        *,
        layer_type: str,
        global_layer_index: int,
        layer_tensors: "dict[str, torch.Tensor]",
        device: "torch.device",
        dtype: "torch.dtype",
        model_config: object,
    ) -> "torch.nn.Module": ...


# ---------------------------------------------------------------------------
# Default layer builder — uses transformers when available
# ---------------------------------------------------------------------------


def _default_layer_builder(
    *,
    layer_type: str,
    global_layer_index: int,
    layer_tensors: "dict[str, torch.Tensor]",
    device: "torch.device",
    dtype: "torch.dtype",
    model_config: object,
) -> "torch.nn.Module":
    """Default layer builder using HuggingFace transformers model classes.

    Attempts to import the appropriate Qwen3.5 decoder layer class and
    instantiate it with the provided config. Falls back to a simple
    nn.Module wrapper if transformers is not available.

    Args:
        layer_type: 'full_attention' or 'linear_attention'.
        global_layer_index: Global index of this layer in the full model.
        layer_tensors: Tensors for this layer (keys are suffixes after
            ``model.layers.{i}.``).
        device: Target device.
        dtype: Target dtype.
        model_config: HuggingFace model config object.

    Returns:
        An nn.Module representing the transformer layer.
    """
    import torch

    try:
        from transformers.models.qwen3_5 import (
            Qwen3_5DecoderLayer,  # type: ignore[import-untyped]
        )

        layer = Qwen3_5DecoderLayer(model_config, global_layer_index)  # type: ignore[arg-type]
        # Load state dict from layer_tensors
        state_dict: dict[str, torch.Tensor] = {}
        for suffix, tensor in layer_tensors.items():
            state_dict[suffix] = tensor
        layer.load_state_dict(state_dict, strict=False)
        layer = layer.to(device=device, dtype=dtype)
        return layer
    except (ImportError, Exception) as exc:
        logger.warning(
            f"Could not use transformers Qwen3_5DecoderLayer for layer "
            f"{global_layer_index} ({layer_type}): {exc}. "
            f"Using passthrough module."
        )
        # Fallback: create a simple module that holds the tensors as parameters
        module = torch.nn.Module()
        for suffix, tensor in layer_tensors.items():
            # Store as buffers (not parameters) to avoid optimizer registration
            safe_name = suffix.replace(".", "_")
            module.register_buffer(safe_name, tensor.to(device=device, dtype=dtype))
        return module


# ---------------------------------------------------------------------------
# build_local_shard_modules — instantiate local layers from loaded tensors
# ---------------------------------------------------------------------------


def build_local_shard_modules(
    *,
    loaded_tensors: "dict[str, torch.Tensor]",
    stage_assignment: PipelineStageAssignment,
    model_config: object,
    device: "torch.device",
    dtype: "torch.dtype | None" = None,
    layer_builder: "Callable[..., torch.nn.Module] | None" = None,
) -> LocalShardModules:
    """Build local PyTorch modules from loaded tensors for a pipeline stage.

    Given pre-loaded tensors (from ``load_tensors_from_manifest``) and a model
    config, instantiates only the layers in ``[start_layer, end_layer)`` using
    the appropriate layer type (full_attention or linear_attention) from the
    model config's ``layer_types`` list.

    Does NOT create layers outside the assigned range.

    Args:
        loaded_tensors: Dictionary of tensor name to tensor, as returned by
            ``load_tensors_from_manifest``.
        stage_assignment: This rank's pipeline stage assignment.
        model_config: HuggingFace model config (must have ``layer_types``
            attribute for Qwen3.5, or defaults to 'full_attention' for all).
        device: Target device for modules.
        dtype: Target dtype. Defaults to ``torch.bfloat16``.
        layer_builder: Optional callable for creating layer instances.
            Accepts dependency injection for testing. When None, uses the
            default builder that attempts transformers model classes.

    Returns:
        A ``LocalShardModules`` containing the built modules, layer types,
        and global layer indices.

    Raises:
        ValueError: If layer_types metadata is missing or invalid for the
            assigned layer range.
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16

    if layer_builder is None:
        layer_builder = _default_layer_builder

    # Determine layer types from model config
    all_layer_types: list[str] | None = getattr(model_config, "layer_types", None)

    start_layer = stage_assignment.start_layer
    end_layer = stage_assignment.end_layer

    # If no layer_types metadata, default all to full_attention
    if all_layer_types is None:
        logger.warning(
            "model_config has no 'layer_types' attribute; "
            "defaulting all layers to 'full_attention'"
        )
        local_layer_types = ["full_attention"] * (end_layer - start_layer)
    else:
        # Validate that we have enough entries
        if len(all_layer_types) < end_layer:
            raise ValueError(
                f"model_config.layer_types has {len(all_layer_types)} entries "
                f"but stage requires layers up to index {end_layer - 1}"
            )
        local_layer_types = list(all_layer_types[start_layer:end_layer])

    # Validate layer type values
    valid_types = {"full_attention", "linear_attention"}
    for idx, lt in enumerate(local_layer_types):
        if lt not in valid_types:
            raise ValueError(
                f"Invalid layer type '{lt}' at global index "
                f"{start_layer + idx}. Must be one of {valid_types}"
            )

    # Build global layer indices
    global_layer_indices = list(range(start_layer, end_layer))

    # Instantiate layers
    layers_list: list[torch.nn.Module] = []
    for local_idx, global_idx in enumerate(global_layer_indices):
        layer_type = local_layer_types[local_idx]

        # Extract tensors for this layer (prefix: model.layers.{global_idx}.)
        prefix = f"model.layers.{global_idx}."
        layer_tensors: dict[str, torch.Tensor] = {}
        for tensor_name, tensor in loaded_tensors.items():
            if tensor_name.startswith(prefix):
                suffix = tensor_name[len(prefix):]
                layer_tensors[suffix] = tensor

        layer_module = layer_builder(
            layer_type=layer_type,
            global_layer_index=global_idx,
            layer_tensors=layer_tensors,
            device=device,
            dtype=dtype,
            model_config=model_config,
        )
        layers_list.append(layer_module)

    layers = torch.nn.ModuleList(layers_list)

    # Build embedding (rank 0 only)
    embed_tokens: torch.nn.Embedding | None = None
    if stage_assignment.owns_embedding:
        embed_weight = loaded_tensors.get("model.embed_tokens.weight")
        if embed_weight is not None:
            vocab_size, embed_dim = embed_weight.shape
            embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)
            embed_tokens.weight = torch.nn.Parameter(
                embed_weight.to(device=device, dtype=dtype), requires_grad=False
            )
        else:
            logger.warning(
                "Rank 0 owns embedding but 'model.embed_tokens.weight' "
                "not found in loaded tensors"
            )

    # Build lm_head and final_norm (last rank only)
    lm_head: torch.nn.Linear | None = None
    final_norm: torch.nn.Module | None = None
    if stage_assignment.owns_lm_head:
        # lm_head
        lm_head_weight = loaded_tensors.get("lm_head.weight")
        if lm_head_weight is not None:
            out_features, in_features = lm_head_weight.shape
            lm_head = torch.nn.Linear(in_features, out_features, bias=False)
            lm_head.weight = torch.nn.Parameter(
                lm_head_weight.to(device=device, dtype=dtype), requires_grad=False
            )
        else:
            logger.warning(
                "Last rank owns lm_head but 'lm_head.weight' "
                "not found in loaded tensors"
            )

        # final_norm (RMSNorm)
        norm_weight = loaded_tensors.get("model.norm.weight")
        if norm_weight is not None:
            hidden_size = norm_weight.shape[0]
            # Use a simple RMSNorm implementation
            final_norm = _build_rms_norm(hidden_size, norm_weight, device, dtype)
        else:
            logger.warning(
                "Last rank owns final_norm but 'model.norm.weight' "
                "not found in loaded tensors"
            )

    # Build rotary embedding (all ranks need this for position encoding)
    rotary_emb = _build_rotary_embedding(model_config, device, dtype)

    logger.info(
        f"Built local shard modules for rank {stage_assignment.rank}: "
        f"{len(layers_list)} layers "
        f"(global indices {global_layer_indices}), "
        f"embed={embed_tokens is not None}, "
        f"lm_head={lm_head is not None}, "
        f"final_norm={final_norm is not None}, "
        f"rotary_emb={rotary_emb is not None}, "
        f"layer_types={local_layer_types}"
    )

    return LocalShardModules(
        layers=layers,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        final_norm=final_norm,
        rotary_emb=rotary_emb,
        layer_types=local_layer_types,
        global_layer_indices=global_layer_indices,
    )


# ---------------------------------------------------------------------------
# Helper: build RMSNorm module
# ---------------------------------------------------------------------------


def _build_rms_norm(
    hidden_size: int,
    weight: "torch.Tensor",
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.nn.Module":
    """Build a simple RMSNorm module with the given weight.

    Attempts to use the transformers RmsNorm if available, otherwise
    creates a minimal implementation.
    """
    import torch

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5RMSNorm,  # type: ignore[import-untyped]
        )

        norm = Qwen3_5RMSNorm(hidden_size)
        norm.weight = torch.nn.Parameter(
            weight.to(device=device, dtype=dtype), requires_grad=False
        )
        return norm
    except (ImportError, Exception):
        pass

    # Fallback: minimal RMSNorm
    class _RMSNorm(torch.nn.Module):
        def __init__(self, size: int, w: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(w, requires_grad=False)
            self.variance_epsilon = 1e-6

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return self.weight * hidden_states.to(input_dtype)

    return _RMSNorm(hidden_size, weight.to(device=device, dtype=dtype))


# ---------------------------------------------------------------------------
# Helper: build rotary embedding module
# ---------------------------------------------------------------------------


def _build_rotary_embedding(
    model_config: object,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.nn.Module | None":
    """Build a rotary embedding module from model config.

    Attempts to use transformers RotaryEmbedding if available. Returns None
    if the config does not provide enough information to build one.
    """
    import torch

    # Extract rotary parameters from config
    head_dim: int | None = getattr(model_config, "head_dim", None)
    hidden_size: int | None = getattr(model_config, "hidden_size", None)
    num_attention_heads: int | None = getattr(
        model_config, "num_attention_heads", None
    )

    if head_dim is None:
        if hidden_size is not None and num_attention_heads is not None:
            head_dim = hidden_size // num_attention_heads
        else:
            logger.debug(
                "Cannot determine head_dim for rotary embedding; skipping"
            )
            return None

    rope_theta: float = float(getattr(model_config, "rope_theta", 10000.0))
    max_position_embeddings: int = int(
        getattr(model_config, "max_position_embeddings", 32768)
    )
    partial_rotary_factor: float = float(
        getattr(model_config, "partial_rotary_factor", 1.0)
    )

    rotary_dim = int(head_dim * partial_rotary_factor)

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5RotaryEmbedding,  # type: ignore[import-untyped]
        )

        rotary_emb = Qwen3_5RotaryEmbedding(config=model_config)  # type: ignore[arg-type]
        rotary_emb = rotary_emb.to(device=device)
        return rotary_emb
    except (ImportError, Exception):
        pass

    # Fallback: minimal rotary embedding
    class _RotaryEmbedding(torch.nn.Module):
        def __init__(
            self,
            dim: int,
            max_pos: int,
            base: float,
        ) -> None:
            super().__init__()
            self.dim = dim
            self.max_position_embeddings = max_pos
            self.base = base
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(
            self, x: torch.Tensor, position_ids: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Compute cos and sin for rotary embedding
            inv_freq_expanded = self.inv_freq[None, :, None].to(
                device=x.device, dtype=torch.float32
            )
            position_ids_expanded = position_ids[None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype=x.dtype)
            sin = emb.sin().to(dtype=x.dtype)
            return cos, sin

    rotary = _RotaryEmbedding(rotary_dim, max_position_embeddings, rope_theta)
    rotary = rotary.to(device=device)
    return rotary


# ---------------------------------------------------------------------------
# validate_pipeline_distribution — strict startup validation
# ---------------------------------------------------------------------------


def validate_pipeline_distribution(
    *,
    weight_map: dict[str, str],
    layer_distribution: PipelineLayerDistribution,
    model_config: object,
) -> None:
    """Validate the full pipeline distribution across ALL ranks at startup.

    Performs comprehensive validation to catch configuration errors before
    inference begins. Raises ValueError with clear messages identifying
    the specific problem.

    Validations performed:
        1. Missing tensor names — for each rank's assignment, verify that
           every expected layer tensor exists in the weight_map.
        2. Duplicate owned tensors — verify no tensor is owned by more
           than one rank.
        3. Unassigned layer tensors — verify every layer/embedding/norm/
           lm_head tensor in the weight_map is assigned to exactly one rank.
        4. Unsupported layer type — if model_config.layer_types contains
           values other than "full_attention" or "linear_attention", raise
           an error identifying the invalid types and their layer indices.

    Args:
        weight_map: Full tensor name to shard file mapping from the
            safetensors index.
        layer_distribution: Layer distribution across pipeline ranks.
        model_config: HuggingFace model config object. Must have
            ``layer_types`` attribute if layer type validation is desired.

    Raises:
        ValueError: If any validation check fails, with a message
            identifying the specific problem.
    """
    valid_layer_types = {"full_attention", "linear_attention"}

    # --- Validation 4: Unsupported layer types ---
    all_layer_types: list[str] | None = getattr(model_config, "layer_types", None)
    if all_layer_types is not None:
        invalid_types: dict[int, str] = {}
        for idx, layer_type in enumerate(all_layer_types):
            if layer_type not in valid_layer_types:
                invalid_types[idx] = layer_type
        if invalid_types:
            details = ", ".join(
                f"layer {idx}: '{lt}'" for idx, lt in sorted(invalid_types.items())
            )
            raise ValueError(
                f"Unsupported layer types in model_config.layer_types: "
                f"{details}. "
                f"Supported types are: {sorted(valid_layer_types)}"
            )

    # --- Validation 1 & 2: Missing tensors and duplicate ownership ---
    # Resolve ownership for all ranks and check for issues
    tensor_to_rank: dict[str, int] = {}
    duplicate_tensors: dict[str, list[int]] = {}

    for rank in range(layer_distribution.rank_count):
        assignment = layer_distribution.get_stage_assignment(rank)

        # Determine expected tensors for this rank based on assignment
        expected_tensors = _compute_expected_tensors_for_stage(
            stage_assignment=assignment,
            total_layer_count=layer_distribution.total_layer_count,
        )

        # Check for missing tensors (expected but not in weight_map)
        missing_tensors = [t for t in expected_tensors if t not in weight_map]
        if missing_tensors:
            raise ValueError(
                f"Missing tensor names for rank {rank} "
                f"(layers [{assignment.start_layer}, {assignment.end_layer})): "
                f"{sorted(missing_tensors)}"
            )

        # Resolve actual ownership from weight_map
        owned = resolve_tensor_ownership(weight_map, assignment)

        # Check for duplicate ownership
        for tensor_name in owned:
            if tensor_name in tensor_to_rank:
                if tensor_name not in duplicate_tensors:
                    duplicate_tensors[tensor_name] = [tensor_to_rank[tensor_name]]
                duplicate_tensors[tensor_name].append(rank)
            else:
                tensor_to_rank[tensor_name] = rank

    if duplicate_tensors:
        details = ", ".join(
            f"'{name}' claimed by ranks {ranks}"
            for name, ranks in sorted(duplicate_tensors.items())
        )
        raise ValueError(
            f"Duplicate owned tensors across ranks: {details}"
        )

    # --- Validation 3: Unassigned layer tensors ---
    # Only layer/embedding/norm/lm_head tensors must be assigned.
    # Metadata tensors that don't match any ownership pattern are acceptable.
    unassigned_tensors: list[str] = []
    for tensor_name in weight_map:
        if tensor_name not in tensor_to_rank:
            # Check if this is a model tensor that should be assigned
            if _is_model_tensor(tensor_name):
                unassigned_tensors.append(tensor_name)

    if unassigned_tensors:
        raise ValueError(
            f"Unassigned layer tensors not claimed by any rank: "
            f"{sorted(unassigned_tensors)}"
        )

    logger.debug(
        f"Pipeline distribution validation passed: "
        f"{len(tensor_to_rank)} tensors assigned across "
        f"{layer_distribution.rank_count} ranks"
    )


def _compute_expected_tensors_for_stage(
    *,
    stage_assignment: PipelineStageAssignment,
    total_layer_count: int,
) -> list[str]:
    """Compute the tensor names expected for a given stage assignment.

    This generates the expected tensor name patterns based on the stage's
    layer range and auxiliary module ownership. Used for validation to
    detect missing tensors in the weight_map.

    Note: This returns a minimal set of expected structural tensors.
    The actual model may have additional tensors per layer, but at minimum
    each layer should have an input_layernorm.weight tensor.

    Args:
        stage_assignment: The pipeline stage assignment to check.
        total_layer_count: Total number of layers in the model.

    Returns:
        List of tensor names that must exist in the weight_map.
    """
    expected: list[str] = []

    # Embedding tensor
    if stage_assignment.owns_embedding:
        expected.append("model.embed_tokens.weight")

    # Layer tensors — at minimum, each layer must have input_layernorm.weight
    for layer_idx in range(stage_assignment.start_layer, stage_assignment.end_layer):
        expected.append(f"model.layers.{layer_idx}.input_layernorm.weight")

    # Final norm and lm_head
    if stage_assignment.owns_lm_head:
        expected.append("model.norm.weight")
        expected.append("lm_head.weight")

    return expected


def _is_model_tensor(tensor_name: str) -> bool:
    """Check if a tensor name represents a model tensor that must be assigned.

    Model tensors include:
    - model.embed_tokens.weight
    - model.layers.{i}.* (any layer tensor)
    - model.norm.weight
    - lm_head.weight

    Metadata tensors or other entries that don't match these patterns
    are considered acceptable to leave unassigned.

    Args:
        tensor_name: The tensor name to check.

    Returns:
        True if this is a model tensor that must be assigned to a rank.
    """
    if tensor_name == "model.embed_tokens.weight":
        return True
    if tensor_name == "model.norm.weight":
        return True
    if tensor_name == "lm_head.weight":
        return True
    if _LAYER_TENSOR_PATTERN.match(tensor_name) is not None:
        return True
    return False


# ---------------------------------------------------------------------------
# initialize_pipeline_shard — top-level orchestrator for engine startup
# ---------------------------------------------------------------------------


def initialize_pipeline_shard(
    *,
    model_path: Path,
    rank: int,
    layer_distribution: "PipelineLayerDistribution",
    model_config: object,
    device: "torch.device",
    dtype: "torch.dtype | None" = None,
    performance_recorder: object | None = None,
    optimization_config: object | None = None,
) -> "PipelineParallelShard":
    """Top-level orchestrator for local-shard pipeline initialization.

    This is the single entry point that the engine calls during startup to
    create a fully initialized PipelineParallelShard from a safetensors
    checkpoint. It replaces the old full-model loading path for
    pipeline-parallel mode.

    Steps:
        1. Get the PipelineStageAssignment from layer_distribution
        2. Create the local tensor manifest (no data loaded yet)
        3. Load actual tensors from safetensors shard files
        4. Validate that all expected tensors were loaded
        5. Build local PyTorch modules from loaded tensors
        6. Create a PipelineStageConfig from the assignment
        7. Create the PipelineParallelShard via from_local_shard_modules()
        8. Log a summary: rank, layer range, tensor count, memory, time

    Args:
        model_path: Path to the model directory containing safetensors
            index and shard files.
        rank: Pipeline stage rank (0-indexed).
        layer_distribution: Layer distribution across pipeline ranks.
        model_config: HuggingFace model config object (must have
            hidden_size, vocab_size, num_hidden_layers attributes).
        device: Target torch device for loaded tensors and modules.
        dtype: Target dtype. Defaults to torch.bfloat16.
        performance_recorder: Optional PerformanceRecorder for instrumentation.
        optimization_config: Optional PytorchXpuOptimizationConfiguration.

    Returns:
        A fully initialized PipelineParallelShard ready for inference.

    Raises:
        FileNotFoundError: If safetensors index or shard files are missing.
        ValueError: If no tensors are assigned, tensors are missing after
            load, or the model config lacks required attributes.
    """
    import torch

    from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
        PipelineParallelShard as _PipelineParallelShard,
    )
    from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
        PipelineStageConfig,
    )

    if dtype is None:
        dtype = torch.bfloat16

    start_time = time.perf_counter()

    # Step 1: Get the stage assignment for this rank
    assert isinstance(layer_distribution, PipelineLayerDistribution)
    stage_assignment = layer_distribution.get_stage_assignment(rank)

    logger.info(
        f"Initializing pipeline shard for rank {rank}: "
        f"layers [{stage_assignment.start_layer}, {stage_assignment.end_layer}), "
        f"owns_embedding={stage_assignment.owns_embedding}, "
        f"owns_lm_head={stage_assignment.owns_lm_head}"
    )

    # Step 1.5: Parse weight_map and validate pipeline distribution
    weight_map = parse_safetensors_index(model_path)
    validate_pipeline_distribution(
        weight_map=weight_map,
        layer_distribution=layer_distribution,
        model_config=model_config,
    )

    # Step 2: Create the local tensor manifest (determines what to load)
    manifest = load_qwen_local_shard_from_safetensors(
        model_path=model_path,
        stage_assignment=stage_assignment,
    )

    # Step 3: Load actual tensors from safetensors shard files
    loaded_tensors = load_tensors_from_manifest(
        manifest=manifest,
        model_path=model_path,
        device=device,
        dtype=dtype,
    )

    # Step 4: Validate that all expected tensors were loaded
    validate_loaded_tensors(manifest, loaded_tensors)

    # Step 5: Build local PyTorch modules from loaded tensors
    modules = build_local_shard_modules(
        loaded_tensors=loaded_tensors,
        stage_assignment=stage_assignment,
        model_config=model_config,
        device=device,
        dtype=dtype,
    )

    # Step 6: Create a PipelineStageConfig from the assignment
    hidden_size: int = int(getattr(model_config, "hidden_size", 0))
    vocab_size: int = int(getattr(model_config, "vocab_size", 0))
    num_layers: int = int(getattr(model_config, "num_hidden_layers", 0))

    if hidden_size == 0:
        raise ValueError(
            "model_config.hidden_size is missing or zero; "
            "cannot create PipelineStageConfig"
        )
    if vocab_size == 0:
        raise ValueError(
            "model_config.vocab_size is missing or zero; "
            "cannot create PipelineStageConfig"
        )
    if num_layers == 0:
        raise ValueError(
            "model_config.num_hidden_layers is missing or zero; "
            "cannot create PipelineStageConfig"
        )

    stage_config = PipelineStageConfig(
        rank=rank,
        world_size=layer_distribution.rank_count,
        start_layer=stage_assignment.start_layer,
        end_layer=stage_assignment.end_layer,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        device=str(device),
    )

    # Step 7: Create the PipelineParallelShard
    shard = _PipelineParallelShard.from_local_shard_modules(
        modules=modules,
        config=stage_config,
        text_model_config=model_config,
        performance_recorder=performance_recorder,  # type: ignore[arg-type]
        optimization_config=optimization_config,  # type: ignore[arg-type]
    )

    # Step 8: Log summary
    elapsed = time.perf_counter() - start_time

    # Estimate memory used by loaded tensors
    total_bytes: int = 0
    for tensor in loaded_tensors.values():
        total_bytes += tensor.nelement() * tensor.element_size()
    total_mib = total_bytes / (1024 * 1024)

    logger.info(
        f"Pipeline shard initialized for rank {rank}: "
        f"layers [{stage_assignment.start_layer}, {stage_assignment.end_layer}), "
        f"{manifest.local_tensor_count} tensors loaded "
        f"({total_mib:.1f} MiB), "
        f"elapsed {elapsed:.2f}s"
    )

    return shard


# ---------------------------------------------------------------------------
# LoadingMemoryMetrics — memory metrics captured during model loading
# ---------------------------------------------------------------------------

# On macOS, ru_maxrss is in bytes; on Linux, it's in kilobytes.
_RSS_UNIT_BYTES: int = 1 if platform.system() == "Darwin" else 1024


@dataclass(frozen=True)
class LoadingMemoryMetrics:
    """Memory metrics captured during model loading.

    Records peak resident set size (RSS), RSS before and after loading,
    the net memory increase, total tensor bytes loaded, and wall-clock
    loading duration.

    Used by bench_xpu.py to report memory overhead of local-shard loading
    and validate that peak RSS stays within the budget defined in
    Requirement 1.8.
    """

    peak_rss_bytes: int
    """Peak resident set size during loading."""

    rss_before_bytes: int
    """RSS before loading started."""

    rss_after_bytes: int
    """RSS after loading completed."""

    rss_delta_bytes: int
    """Net memory increase (after - before)."""

    tensor_bytes_loaded: int
    """Total bytes of loaded tensors."""

    loading_duration_seconds: float
    """Wall-clock time for loading."""

    @property
    def peak_rss_mib(self) -> float:
        """Peak RSS in mebibytes."""
        return self.peak_rss_bytes / (1024 * 1024)

    @property
    def rss_delta_mib(self) -> float:
        """Net RSS increase in mebibytes."""
        return self.rss_delta_bytes / (1024 * 1024)

    @property
    def tensor_mib_loaded(self) -> float:
        """Total tensor data loaded in mebibytes."""
        return self.tensor_bytes_loaded / (1024 * 1024)


# ---------------------------------------------------------------------------
# measure_loading_memory — wraps initialize_pipeline_shard with memory tracking
# ---------------------------------------------------------------------------


def measure_loading_memory(
    *,
    model_path: Path,
    rank: int,
    layer_distribution: "PipelineLayerDistribution",
    model_config: object,
    device: "torch.device",
    dtype: "torch.dtype | None" = None,
    performance_recorder: object | None = None,
    optimization_config: object | None = None,
) -> "tuple[PipelineParallelShard, LoadingMemoryMetrics]":
    """Measure peak resident memory during pipeline shard initialization.

    Wraps ``initialize_pipeline_shard()`` and captures RSS measurements
    before and after loading using ``resource.getrusage(RUSAGE_SELF)``.

    The peak RSS reported by the OS is the high-water mark for the process
    lifetime, so this function records the delta between the peak before
    and after loading to isolate the loading contribution.

    Args:
        model_path: Path to the model directory containing safetensors
            index and shard files.
        rank: Pipeline stage rank (0-indexed).
        layer_distribution: Layer distribution across pipeline ranks.
        model_config: HuggingFace model config object.
        device: Target torch device for loaded tensors and modules.
        dtype: Target dtype. Defaults to torch.bfloat16.
        performance_recorder: Optional PerformanceRecorder for instrumentation.
        optimization_config: Optional PytorchXpuOptimizationConfiguration.

    Returns:
        A tuple of (PipelineParallelShard, LoadingMemoryMetrics).
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16

    # Capture RSS before loading
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    rss_before_bytes = usage_before.ru_maxrss * _RSS_UNIT_BYTES

    start_time = time.perf_counter()

    # Run the full loading pipeline
    shard = initialize_pipeline_shard(
        model_path=model_path,
        rank=rank,
        layer_distribution=layer_distribution,
        model_config=model_config,
        device=device,
        dtype=dtype,
        performance_recorder=performance_recorder,
        optimization_config=optimization_config,
    )

    elapsed = time.perf_counter() - start_time

    # Capture RSS after loading
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    rss_after_bytes = usage_after.ru_maxrss * _RSS_UNIT_BYTES

    # Peak RSS is the max of before and after (ru_maxrss is cumulative peak)
    peak_rss_bytes = rss_after_bytes

    # Compute tensor bytes from the shard's loaded state
    # We re-derive this from the stage assignment and manifest
    tensor_bytes_loaded: int = 0
    if hasattr(shard, "_loaded_tensor_bytes"):
        tensor_bytes_loaded = int(shard._loaded_tensor_bytes)  # type: ignore[attr-defined]
    else:
        # Estimate from manifest: count tensors and approximate sizes
        # This is a fallback — the actual bytes are computed during loading
        # Use the manifest local_tensor_count as a proxy
        # Better: re-read from the loading log or compute from modules
        tensor_bytes_loaded = _estimate_tensor_bytes_from_modules(shard)

    rss_delta_bytes = rss_after_bytes - rss_before_bytes

    metrics = LoadingMemoryMetrics(
        peak_rss_bytes=peak_rss_bytes,
        rss_before_bytes=rss_before_bytes,
        rss_after_bytes=rss_after_bytes,
        rss_delta_bytes=rss_delta_bytes,
        tensor_bytes_loaded=tensor_bytes_loaded,
        loading_duration_seconds=elapsed,
    )

    logger.info(
        f"Memory metrics for rank {rank}: "
        f"peak_rss={metrics.peak_rss_mib:.1f} MiB, "
        f"rss_delta={metrics.rss_delta_mib:.1f} MiB, "
        f"tensors={metrics.tensor_mib_loaded:.1f} MiB, "
        f"duration={elapsed:.2f}s"
    )

    return shard, metrics


def _estimate_tensor_bytes_from_modules(
    shard: "PipelineParallelShard",
) -> int:
    """Estimate total tensor bytes from a PipelineParallelShard's modules.

    Iterates over all parameters and buffers in the shard's underlying
    modules to compute total memory footprint.

    Args:
        shard: A fully initialized PipelineParallelShard.

    Returns:
        Total bytes across all parameters and buffers.
    """
    import torch

    total_bytes: int = 0

    # Try to access the underlying nn.Module components
    for attr_name in ("_layers", "_embed_tokens", "_lm_head", "_final_norm"):
        module = getattr(shard, attr_name, None)
        if module is not None and isinstance(module, torch.nn.Module):
            for param in module.parameters():
                total_bytes += param.nelement() * param.element_size()
            for buf in module.buffers():
                total_bytes += buf.nelement() * buf.element_size()

    return total_bytes
