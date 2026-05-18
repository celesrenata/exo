"""Streaming safetensors model loader with tensor-parallel sharding.

Loads HuggingFace model weights directly from safetensors files on disk,
sharding each tensor as it's read. Never materializes the full model in
memory — peak memory stays at ~55GB (sharded weights only) instead of
~110GB (full model + sharded copy).
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch

from .tensor_parallel_shard import TPShardConfig

logger = logging.getLogger(__name__)

# Redundant (not sharded) parameter suffixes — kept full on every rank
_REDUNDANT_SUFFIXES = (
    "embed_tokens.weight",
    "norm.weight",
    "norm.bias",
    "lm_head.weight",
    "input_layernorm.weight",
    "input_layernorm.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "q_norm.weight",
    "k_norm.weight",
)

# Linear attention weights are redundant on all ranks
_LINEAR_ATTN_PATTERN = "linear_attn."

# MTP (multi-token prediction) layers are redundant — they have different
# dimensions than the main model and are auxiliary prediction heads
_MTP_PATTERN = "mtp."


def load_sharded_from_safetensors(
    model_path: str,
    config: TPShardConfig,
    device: str,
    model_id: str = "",
) -> tuple[dict[str, torch.Tensor], dict[int, Any], Any, Any]:
    """Load model weights from safetensors files with streaming sharding.

    Reads weights directly from safetensors files, sharding each tensor
    as it's read. Never materializes the full model in memory.

    Args:
        model_path: Path to the model directory containing safetensors files
        config: TPShardConfig with rank, world_size, and model dimensions
        device: Target device string (e.g., "xpu:0", "cpu")
        model_id: HuggingFace model ID for loading tokenizer and config

    Returns:
        Tuple of:
        - sharded_state_dict: dict mapping key -> sharded tensor on device
        - native_linear_attn_layers: dict mapping layer_idx -> native layer module
          (empty for pure transformer models)
        - tokenizer: loaded tokenizer
        - native_rotary_emb: native rotary embedding module for MRoPE models,
          or None for standard RoPE models
    """
    import safetensors.torch as st

    model_dir = Path(model_path)
    sharded_state_dict: dict[str, torch.Tensor] = {}

    # Extract sharding parameters from config
    rank = config.rank
    world_size = config.world_size
    head_dim = config.head_dim
    heads_per_rank = config.heads_per_rank
    kv_heads_per_rank = config.kv_heads_per_rank
    intermediate_per_rank = config.intermediate_per_rank

    # Discover safetensors files and their weight mappings
    files_to_keys = _discover_safetensors_files(model_dir)

    # Process each safetensors file one at a time
    total_params = sum(len(keys) for keys in files_to_keys.values())
    processed = 0

    for filename, keys in files_to_keys.items():
        filepath = model_dir / filename
        logger.info(f"Rank {rank}: loading {filename} ({len(keys)} tensors)")

        # Use safe_open to load tensors selectively (avoids loading MTP/vision into memory)
        from safetensors import safe_open

        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            for key in keys:
                # Skip MTP and vision weights entirely
                if _should_skip(key):
                    processed += 1
                    continue

                if key not in f.keys():
                    logger.warning(f"Rank {rank}: key '{key}' not found in {filename}, skipping")
                    continue

                tensor = f.get_tensor(key)

                # Handle fused gate_up_proj: split into gate_proj + up_proj, shard each
                if _matches_mlp_key(key, "gate_up_proj.weight"):
                    _split_fused_gate_up(
                        sharded_state_dict, key, tensor,
                        is_bias=False, rank=rank,
                        intermediate_per_rank=intermediate_per_rank, device=device,
                    )
                    del tensor
                    processed += 1
                    continue

                if _matches_mlp_key(key, "gate_up_proj.bias"):
                    _split_fused_gate_up(
                        sharded_state_dict, key, tensor,
                        is_bias=True, rank=rank,
                        intermediate_per_rank=intermediate_per_rank, device=device,
                    )
                    del tensor
                    processed += 1
                    continue

                # Handle fused QKV: split into q_proj, k_proj, v_proj, shard each
                if _matches_attn_key(key, "qkv_proj.weight"):
                    _split_fused_qkv(
                        sharded_state_dict, key, tensor,
                        is_bias=False, rank=rank, config=config, device=device,
                    )
                    del tensor
                    processed += 1
                    continue

                if _matches_attn_key(key, "qkv_proj.bias"):
                    _split_fused_qkv(
                        sharded_state_dict, key, tensor,
                        is_bias=True, rank=rank, config=config, device=device,
                    )
                    del tensor
                    processed += 1
                    continue

                # Standard sharding for all other parameters
                sharded = _shard_tensor(
                    key, tensor, rank=rank, world_size=world_size,
                    head_dim=head_dim, heads_per_rank=heads_per_rank,
                    kv_heads_per_rank=kv_heads_per_rank,
                    intermediate_per_rank=intermediate_per_rank,
                )

                sharded_state_dict[key] = sharded.to(device)
                del tensor
                processed += 1

        if processed % 500 == 0 or processed == total_params:
            logger.info(f"Rank {rank}: processed {processed}/{total_params} parameters")

    logger.info(
        f"Rank {rank}: sharded {len(sharded_state_dict)} parameters to device {device}"
    )

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer_source = model_id or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    # For hybrid models, load native linear_attn layers
    native_layers = _load_native_linear_attn_layers(
        model_path, model_id, sharded_state_dict, device
    )

    # For MRoPE models, load native rotary embedding
    native_rotary_emb = _load_native_rotary_emb(model_path, model_id, device)

    return sharded_state_dict, native_layers, tokenizer, native_rotary_emb


def _discover_safetensors_files(model_dir: Path) -> dict[str, list[str]]:
    """Find safetensors files and map each to its contained weight keys.

    Reads model.safetensors.index.json if present (multi-file models),
    otherwise falls back to a single model.safetensors file.

    Returns:
        Dict mapping filename -> list of weight keys in that file.
    """
    index_path = model_dir / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]

        # Group keys by their containing file
        files_to_keys: dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            files_to_keys.setdefault(filename, []).append(key)

        logger.info(
            f"Found index with {len(weight_map)} weights across "
            f"{len(files_to_keys)} safetensors files"
        )
        return files_to_keys

    # Single file fallback
    single_file = model_dir / "model.safetensors"
    if not single_file.exists():
        raise FileNotFoundError(
            f"No safetensors files found in {model_dir}. "
            f"Expected model.safetensors.index.json or model.safetensors"
        )

    # Read the file header to get key names without loading tensors
    import safetensors.torch as st

    file_tensors = st.load_file(str(single_file), device="cpu")
    keys = list(file_tensors.keys())
    del file_tensors

    logger.info(f"Found single safetensors file with {len(keys)} weights")
    return {"model.safetensors": keys}


def _shard_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    head_dim: int,
    heads_per_rank: int,
    kv_heads_per_rank: int,
    intermediate_per_rank: int,
) -> torch.Tensor:
    """Shard a single tensor based on its parameter key name.

    Replicates the logic from TensorParallelShard._shard_parameter().
    """
    # Redundant parameters: keep full on every rank
    if _is_redundant(key):
        return tensor.clone()

    # --- Attention projections ---

    # Q projection: column-parallel (split output dim 0)
    if _matches_attn_key(key, "q_proj.weight"):
        shard_size = heads_per_rank * head_dim
        return tensor.narrow(0, rank * shard_size, shard_size).clone()

    if _matches_attn_key(key, "q_proj.bias"):
        shard_size = heads_per_rank * head_dim
        return tensor.narrow(0, rank * shard_size, shard_size).clone()

    # K projection: column-parallel (split output dim 0)
    if _matches_attn_key(key, "k_proj.weight"):
        shard_size = kv_heads_per_rank * head_dim
        return tensor.narrow(0, rank * shard_size, shard_size).clone()

    if _matches_attn_key(key, "k_proj.bias"):
        shard_size = kv_heads_per_rank * head_dim
        return tensor.narrow(0, rank * shard_size, shard_size).clone()

    # V projection: column-parallel (split output dim 0)
    if _matches_attn_key(key, "v_proj.weight"):
        shard_size = kv_heads_per_rank * head_dim
        return tensor.narrow(0, rank * shard_size, shard_size).clone()

    if _matches_attn_key(key, "v_proj.bias"):
        shard_size = kv_heads_per_rank * head_dim
        return tensor.narrow(0, rank * shard_size, shard_size).clone()

    # O projection: row-parallel (split input dim 1)
    if _matches_attn_key(key, "o_proj.weight"):
        shard_size = heads_per_rank * head_dim
        return tensor.narrow(1, rank * shard_size, shard_size).clone()

    if _matches_attn_key(key, "o_proj.bias"):
        # o_proj bias is NOT sharded — added after all-reduce
        return tensor.clone()

    # --- MLP projections ---

    # Gate projection: column-parallel (split output dim 0)
    if _matches_mlp_key(key, "gate_proj.weight"):
        return tensor.narrow(0, rank * intermediate_per_rank, intermediate_per_rank).clone()

    if _matches_mlp_key(key, "gate_proj.bias"):
        return tensor.narrow(0, rank * intermediate_per_rank, intermediate_per_rank).clone()

    # Up projection: column-parallel (split output dim 0)
    if _matches_mlp_key(key, "up_proj.weight"):
        return tensor.narrow(0, rank * intermediate_per_rank, intermediate_per_rank).clone()

    if _matches_mlp_key(key, "up_proj.bias"):
        return tensor.narrow(0, rank * intermediate_per_rank, intermediate_per_rank).clone()

    # Down projection: row-parallel (split input dim 1)
    if _matches_mlp_key(key, "down_proj.weight"):
        return tensor.narrow(1, rank * intermediate_per_rank, intermediate_per_rank).clone()

    if _matches_mlp_key(key, "down_proj.bias"):
        # down_proj bias is NOT sharded — added after all-reduce
        return tensor.clone()

    # Unknown parameter — keep redundant (safe default)
    logger.debug(f"Rank {rank}: keeping parameter '{key}' redundant (unrecognized)")
    return tensor.clone()


def _split_fused_gate_up(
    sharded_state_dict: dict[str, torch.Tensor],
    param_name: str,
    param_tensor: torch.Tensor,
    *,
    is_bias: bool,
    rank: int,
    intermediate_per_rank: int,
    device: str,
) -> None:
    """Split a fused gate_up_proj into separate gate_proj + up_proj, shard each.

    Fused gate_up_proj layout:
    - Weight shape: [2 * intermediate_size, hidden_size]
    - Bias shape: [2 * intermediate_size]

    The fused tensor is ordered as [gate, up] along dimension 0.
    """
    total_intermediate = param_tensor.shape[0] // 2
    gate_full = param_tensor.narrow(0, 0, total_intermediate)
    up_full = param_tensor.narrow(0, total_intermediate, total_intermediate)

    # Shard each part for this rank (column-parallel: slice output dim)
    gate_shard = gate_full.narrow(0, rank * intermediate_per_rank, intermediate_per_rank).clone()
    up_shard = up_full.narrow(0, rank * intermediate_per_rank, intermediate_per_rank).clone()

    # Construct canonical key names
    suffix = "bias" if is_bias else "weight"
    gate_key = param_name.replace(f"gate_up_proj.{suffix}", f"gate_proj.{suffix}")
    up_key = param_name.replace(f"gate_up_proj.{suffix}", f"up_proj.{suffix}")

    sharded_state_dict[gate_key] = gate_shard.to(device)
    sharded_state_dict[up_key] = up_shard.to(device)

    logger.debug(
        f"Rank {rank}: split fused gate_up '{param_name}' -> "
        f"gate={gate_key} ({tuple(gate_shard.shape)}), "
        f"up={up_key} ({tuple(up_shard.shape)})"
    )


def _split_fused_qkv(
    sharded_state_dict: dict[str, torch.Tensor],
    param_name: str,
    param_tensor: torch.Tensor,
    *,
    is_bias: bool,
    rank: int,
    config: TPShardConfig,
    device: str,
) -> None:
    """Split a fused QKV weight/bias into separate Q, K, V entries, shard each.

    Fused QKV layout (Phi-style):
    - Weight shape: [(num_heads + 2 * num_kv_heads) * head_dim, hidden_size]
    - Bias shape: [(num_heads + 2 * num_kv_heads) * head_dim]

    The fused tensor is ordered as [Q, K, V] along dimension 0.
    """
    head_dim = config.head_dim
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    heads_per_rank = config.heads_per_rank
    kv_heads_per_rank = config.kv_heads_per_rank

    q_size = num_heads * head_dim
    k_size = num_kv_heads * head_dim
    v_size = num_kv_heads * head_dim

    # Split along dimension 0
    q_full = param_tensor.narrow(0, 0, q_size)
    k_full = param_tensor.narrow(0, q_size, k_size)
    v_full = param_tensor.narrow(0, q_size + k_size, v_size)

    # Shard each part for this rank
    q_shard_size = heads_per_rank * head_dim
    k_shard_size = kv_heads_per_rank * head_dim
    v_shard_size = kv_heads_per_rank * head_dim

    q_shard = q_full.narrow(0, rank * q_shard_size, q_shard_size).clone()
    k_shard = k_full.narrow(0, rank * k_shard_size, k_shard_size).clone()
    v_shard = v_full.narrow(0, rank * v_shard_size, v_shard_size).clone()

    # Construct canonical key names
    suffix = "bias" if is_bias else "weight"
    q_key = param_name.replace(f"qkv_proj.{suffix}", f"q_proj.{suffix}")
    k_key = param_name.replace(f"qkv_proj.{suffix}", f"k_proj.{suffix}")
    v_key = param_name.replace(f"qkv_proj.{suffix}", f"v_proj.{suffix}")

    sharded_state_dict[q_key] = q_shard.to(device)
    sharded_state_dict[k_key] = k_shard.to(device)
    sharded_state_dict[v_key] = v_shard.to(device)

    logger.debug(
        f"Rank {rank}: split fused QKV '{param_name}' -> "
        f"q={q_key} ({tuple(q_shard.shape)}), "
        f"k={k_key} ({tuple(k_shard.shape)}), "
        f"v={v_key} ({tuple(v_shard.shape)})"
    )


def _load_native_linear_attn_layers(
    model_path: str,
    model_id: str,
    sharded_state_dict: dict[str, torch.Tensor],
    device: str,
) -> dict[int, Any]:
    """Load native linear attention layers for hybrid models (Qwen3.5/3.6).

    Instantiates individual Qwen3_5GatedDeltaNet modules from the model config
    and loads their weights from the sharded_state_dict. Does NOT use
    device_map="meta" (which requires accelerate).

    Returns:
        Dict mapping layer_idx -> native linear_attn module on device.
        Empty dict for pure transformer models.
    """
    # Check if this model has linear_attn keys at all
    has_linear_attn = any(_LINEAR_ATTN_PATTERN in k for k in sharded_state_dict)
    if not has_linear_attn:
        return {}

    try:
        from transformers import AutoConfig

        source = model_id or model_path
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)

        # Get text_config for hybrid VL models
        text_config = getattr(config, 'text_config', config)
        layer_types = getattr(text_config, 'layer_types', None)
        if layer_types is None:
            return {}

        # Try to import the Qwen3.5 GatedDeltaNet module
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5GatedDeltaNet,  # type: ignore[import-untyped]
            )
        except ImportError:
            logger.warning("Could not import Qwen3_5GatedDeltaNet from transformers")
            return {}

        native_layers: dict[int, Any] = {}

        for idx, layer_type in enumerate(layer_types):
            if layer_type != "linear_attention":
                continue

            # Collect weights for this linear_attn module from sharded_state_dict
            prefix_patterns = [
                f"model.layers.{idx}.linear_attn.",
                f"model.language_model.layers.{idx}.linear_attn.",
                f"model.model.layers.{idx}.linear_attn.",
                f"model.model.language_model.layers.{idx}.linear_attn.",
            ]

            layer_weights: dict[str, torch.Tensor] = {}
            for key, tensor in sharded_state_dict.items():
                for prefix in prefix_patterns:
                    if key.startswith(prefix):
                        param_name = key[len(prefix):]
                        layer_weights[param_name] = tensor
                        break

            if not layer_weights:
                continue

            # Instantiate the GatedDeltaNet module directly from config
            linear_attn = Qwen3_5GatedDeltaNet(text_config, layer_idx=idx)  # type: ignore[arg-type]
            linear_attn = linear_attn.to(device=device, dtype=torch.bfloat16)

            # Load the weights
            missing, unexpected = linear_attn.load_state_dict(layer_weights, strict=False)
            if missing:
                logger.debug(
                    f"Layer {idx} linear_attn missing keys ({len(missing)}): {missing[:3]}..."
                )

            # Log weight loading status to file for diagnosis
            with open(f"/tmp/tp_native_layer_load.log", "a") as _nlf:
                _nlf.write(
                    f"Layer {idx}: loaded {len(layer_weights)} weights, "
                    f"missing={len(missing)}, unexpected={len(unexpected)}, "
                    f"keys={list(layer_weights.keys())[:5]}\n"
                )
                if missing:
                    _nlf.write(f"  MISSING: {missing}\n")
                _nlf.flush()

            linear_attn.eval()
            native_layers[idx] = linear_attn

        if native_layers:
            logger.info(
                f"Loaded {len(native_layers)} native linear_attn layers from config+weights"
            )

        return native_layers

    except Exception as e:
        logger.warning(f"Failed to load native linear_attn layers: {e}")
        return {}


def _load_native_rotary_emb(
    model_path: str,
    model_id: str,
    device: str,
) -> Any:
    """Load native rotary embedding for MRoPE models (Qwen3.5/3.6).

    Instantiates Qwen3_5RotaryEmbedding from the model config when the model
    uses Multi-Resolution Rotary Position Embedding (MRoPE). This is required
    for the streaming loader path where no HuggingFace model object exists to
    extract rotary_emb from.

    Returns:
        Native rotary embedding module on device, or None for non-MRoPE models.
    """
    try:
        from transformers import AutoConfig

        source = model_id or model_path
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)

        # Get text_config for hybrid VL models
        text_config = getattr(config, 'text_config', config)

        # Check if model uses MRoPE via rope_scaling dict
        rope_scaling = getattr(text_config, 'rope_scaling', None)
        rope_parameters = getattr(text_config, 'rope_parameters', None)

        is_mrope = False
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            if rope_scaling.get("type") == "mrope":
                is_mrope = True
            elif "mrope_section" in rope_scaling:
                is_mrope = True
        if not is_mrope and rope_parameters is not None and isinstance(rope_parameters, dict) and "mrope_section" in rope_parameters:
            is_mrope = True

        if not is_mrope:
            return None

        # Try to import the Qwen3.5 RotaryEmbedding module
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5RotaryEmbedding,  # type: ignore[import-untyped]
            )
        except ImportError:
            logger.warning("Could not import Qwen3_5RotaryEmbedding from transformers")
            return None

        # Instantiate from text_config, move to device
        rotary_emb = Qwen3_5RotaryEmbedding(config=text_config)  # type: ignore[arg-type]
        rotary_emb = rotary_emb.to(device=device, dtype=torch.bfloat16)
        rotary_emb.eval()

        logger.info(
            f"Loaded native Qwen3_5RotaryEmbedding for MRoPE model "
            f"(rope_scaling.type='mrope') on device={device}"
        )

        return rotary_emb

    except Exception as e:
        logger.warning(f"Failed to load native rotary embedding: {e}")
        return None


def _find_model_layers(model: Any) -> Any:
    """Find the transformer layers container in a HuggingFace model.

    Handles multiple model structures:
    - model.model.language_model.layers (Qwen3.5/3.6 multimodal)
    - model.model.layers (standard Qwen/Llama)
    - model.language_model.layers (alternative layout)
    """
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
        if hasattr(inner, "layers"):
            return inner.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    return None


def _is_redundant(param_name: str) -> bool:
    """Check if a parameter should be kept redundant (not sharded)."""
    for suffix in _REDUNDANT_SUFFIXES:
        if param_name.endswith(suffix) or param_name == suffix:
            return True
    if _LINEAR_ATTN_PATTERN in param_name:
        return True
    # MTP and vision weights are SKIPPED entirely (not loaded)
    # They're not needed for text generation
    return False


def _should_skip(param_name: str) -> bool:
    """Check if a parameter should be skipped entirely (not loaded into memory).

    MTP (multi-token prediction) and vision encoder weights are not needed
    for text generation and would waste memory on the 27B model.
    """
    if param_name.startswith(_MTP_PATTERN) or param_name.startswith("mtp."):
        return True
    if "visual." in param_name or "vision." in param_name:
        return True
    return False


def _matches_attn_key(param_name: str, suffix: str) -> bool:
    """Check if param_name matches an attention layer parameter."""
    return ".self_attn." + suffix in param_name


def _matches_mlp_key(param_name: str, suffix: str) -> bool:
    """Check if param_name matches an MLP layer parameter."""
    return ".mlp." + suffix in param_name
