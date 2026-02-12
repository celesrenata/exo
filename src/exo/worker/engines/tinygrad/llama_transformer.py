"""Llama transformer implementation in tinygrad.

This module implements the Llama transformer architecture using tinygrad,
enabling proper text generation on Intel Arc GPUs and other hardware.

Architecture:
- Token embeddings
- N transformer layers with:
  - RMSNorm
  - Multi-head attention with RoPE
  - Feed-forward network with SwiGLU
- Final RMSNorm and LM head

Reference: https://arxiv.org/abs/2307.09288
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class LlamaConfig:
    """Configuration for Llama model architecture.

    This dataclass holds all hyperparameters needed to construct a Llama model.
    Values are typically loaded from a HuggingFace config.json file.

    Attributes:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of hidden states
        intermediate_size: Dimension of feed-forward network
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for grouped-query attention)
        max_position_embeddings: Maximum sequence length
        rms_norm_eps: Epsilon for RMSNorm numerical stability
        rope_theta: Base frequency for rotary position embeddings
        rope_scaling: Optional rope scaling configuration

    Example:
        >>> config = LlamaConfig(
        ...     vocab_size=128256,
        ...     hidden_size=3072,
        ...     num_hidden_layers=28,
        ...     num_attention_heads=24,
        ...     num_key_value_heads=8
        ... )
    """

    vocab_size: int = 128256
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: dict[str, Any] | None = None

    @property
    def head_dim(self) -> int:
        """Compute dimension of each attention head.

        Returns:
            Dimension per head (hidden_size / num_attention_heads)
        """
        return self.hidden_size // self.num_attention_heads

    @property
    def n_layers(self) -> int:
        """Alias for num_hidden_layers for compatibility.

        Returns:
            Number of transformer layers
        """
        return self.num_hidden_layers


# Default configurations for common Llama model sizes
DEFAULT_CONFIGS: Final[dict[str, dict[str, Any]]] = {
    "0.5B": {
        "vocab_size": 128256,
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
    "1B": {
        "vocab_size": 128256,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
    "3B": {
        "vocab_size": 128256,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
    "8B": {
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
    "70B": {
        "vocab_size": 128256,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
}


def parse_config_from_file(config_path: Path) -> LlamaConfig:
    """Parse Llama configuration from HuggingFace config.json file.

    This function reads a config.json file from a HuggingFace model checkpoint
    and extracts the necessary parameters to construct a LlamaConfig.

    Args:
        config_path: Path to config.json file

    Returns:
        LlamaConfig instance with parsed parameters

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config file is invalid or missing required fields
        RuntimeError: If config parsing fails

    Example:
        >>> config = parse_config_from_file(Path("/path/to/model/config.json"))
        >>> print(f"Model has {config.num_hidden_layers} layers")
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config_dict = json.load(f)

        logger.debug(
            "Loaded config from file",
            config_path=str(config_path),
            model_type=config_dict.get("model_type", "unknown"),
        )

        return parse_config_from_dict(config_dict)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to parse config file: {e}") from e


def parse_config_from_dict(config_dict: dict[str, Any]) -> LlamaConfig:
    """Parse Llama configuration from dictionary.

    This function extracts configuration parameters from a dictionary
    (typically loaded from config.json) and creates a LlamaConfig instance.

    The function handles various naming conventions and provides sensible
    defaults for optional parameters.

    Args:
        config_dict: Dictionary containing model configuration

    Returns:
        LlamaConfig instance with parsed parameters

    Raises:
        ValueError: If required fields are missing or invalid

    Example:
        >>> config_dict = {"hidden_size": 3072, "num_hidden_layers": 28, ...}
        >>> config = parse_config_from_dict(config_dict)
    """
    # Validate model type if present
    model_type = config_dict.get("model_type", "").lower()
    if model_type and "llama" not in model_type:
        logger.warning(
            f"Config model_type is '{model_type}', expected 'llama'. "
            "Proceeding anyway, but model may not work correctly."
        )

    # Extract required parameters with validation
    try:
        vocab_size = config_dict.get("vocab_size")
        if vocab_size is None:
            raise ValueError("Missing required field: vocab_size")

        hidden_size = config_dict.get("hidden_size")
        if hidden_size is None:
            raise ValueError("Missing required field: hidden_size")

        num_hidden_layers = config_dict.get("num_hidden_layers")
        if num_hidden_layers is None:
            raise ValueError("Missing required field: num_hidden_layers")

        num_attention_heads = config_dict.get("num_attention_heads")
        if num_attention_heads is None:
            raise ValueError("Missing required field: num_attention_heads")

        # Optional parameters with defaults
        intermediate_size = config_dict.get(
            "intermediate_size", hidden_size * 4  # Common default
        )

        num_key_value_heads = config_dict.get(
            "num_key_value_heads", num_attention_heads  # Default to MHA
        )

        max_position_embeddings = config_dict.get("max_position_embeddings", 8192)

        rms_norm_eps = config_dict.get("rms_norm_eps", 1e-5)

        # RoPE theta - handle different naming conventions
        rope_theta = config_dict.get(
            "rope_theta", config_dict.get("rotary_emb_base", 500000.0)
        )

        # RoPE scaling configuration (optional)
        rope_scaling = config_dict.get("rope_scaling")

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )

        # Validate configuration
        validate_config(config)

        logger.info(
            "Parsed Llama config",
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )

        return config

    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid config parameters: {e}") from e


def validate_config(config: LlamaConfig) -> None:
    """Validate configuration parameter ranges and constraints.

    This function performs comprehensive validation of configuration parameters
    to ensure they are within reasonable ranges and satisfy architectural
    constraints. It raises ValueError with helpful error messages if validation fails.

    Args:
        config: LlamaConfig instance to validate

    Raises:
        ValueError: If any parameter is invalid or violates constraints

    Example:
        >>> config = LlamaConfig(vocab_size=128256, hidden_size=3072, ...)
        >>> validate_config(config)  # Passes if config is valid
        >>> bad_config = LlamaConfig(vocab_size=0, ...)
        >>> validate_config(bad_config)  # Raises ValueError
    """
    errors: list[str] = []

    # Validate vocab_size
    if config.vocab_size <= 0:
        errors.append(f"vocab_size must be positive, got {config.vocab_size}")
    if config.vocab_size > 1_000_000:
        errors.append(
            f"vocab_size seems unreasonably large ({config.vocab_size}), "
            "expected < 1,000,000"
        )

    # Validate hidden_size
    if config.hidden_size <= 0:
        errors.append(f"hidden_size must be positive, got {config.hidden_size}")
    if config.hidden_size % 128 != 0:
        logger.warning(
            f"hidden_size ({config.hidden_size}) is not a multiple of 128, "
            "which may impact performance"
        )
    if config.hidden_size > 32768:
        errors.append(
            f"hidden_size seems unreasonably large ({config.hidden_size}), "
            "expected <= 32768"
        )

    # Validate intermediate_size
    if config.intermediate_size <= 0:
        errors.append(
            f"intermediate_size must be positive, got {config.intermediate_size}"
        )
    if config.intermediate_size < config.hidden_size:
        logger.warning(
            f"intermediate_size ({config.intermediate_size}) is smaller than "
            f"hidden_size ({config.hidden_size}), which is unusual"
        )

    # Validate num_hidden_layers
    if config.num_hidden_layers <= 0:
        errors.append(
            f"num_hidden_layers must be positive, got {config.num_hidden_layers}"
        )
    if config.num_hidden_layers > 200:
        errors.append(
            f"num_hidden_layers seems unreasonably large ({config.num_hidden_layers}), "
            "expected <= 200"
        )

    # Validate num_attention_heads
    if config.num_attention_heads <= 0:
        errors.append(
            f"num_attention_heads must be positive, got {config.num_attention_heads}"
        )
    if config.hidden_size % config.num_attention_heads != 0:
        errors.append(
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"num_attention_heads ({config.num_attention_heads})"
        )

    # Validate num_key_value_heads
    if config.num_key_value_heads <= 0:
        errors.append(
            f"num_key_value_heads must be positive, got {config.num_key_value_heads}"
        )
    if config.num_key_value_heads > config.num_attention_heads:
        errors.append(
            f"num_key_value_heads ({config.num_key_value_heads}) cannot be greater than "
            f"num_attention_heads ({config.num_attention_heads})"
        )
    if config.num_attention_heads % config.num_key_value_heads != 0:
        errors.append(
            f"num_attention_heads ({config.num_attention_heads}) must be divisible by "
            f"num_key_value_heads ({config.num_key_value_heads}) for grouped-query attention"
        )

    # Validate head_dim
    head_dim = config.head_dim
    if head_dim <= 0:
        errors.append(f"head_dim must be positive, got {head_dim}")
    if head_dim % 2 != 0:
        errors.append(
            f"head_dim must be even for rotary embeddings, got {head_dim}"
        )
    if head_dim > 256:
        logger.warning(
            f"head_dim ({head_dim}) is unusually large, expected <= 256"
        )

    # Validate max_position_embeddings
    if config.max_position_embeddings <= 0:
        errors.append(
            f"max_position_embeddings must be positive, got {config.max_position_embeddings}"
        )
    if config.max_position_embeddings > 1_000_000:
        errors.append(
            f"max_position_embeddings seems unreasonably large ({config.max_position_embeddings}), "
            "expected <= 1,000,000"
        )

    # Validate rms_norm_eps
    if config.rms_norm_eps <= 0:
        errors.append(f"rms_norm_eps must be positive, got {config.rms_norm_eps}")
    if config.rms_norm_eps > 1e-3:
        logger.warning(
            f"rms_norm_eps ({config.rms_norm_eps}) is unusually large, "
            "expected <= 1e-3"
        )

    # Validate rope_theta
    if config.rope_theta <= 0:
        errors.append(f"rope_theta must be positive, got {config.rope_theta}")
    if config.rope_theta < 1000 or config.rope_theta > 10_000_000:
        logger.warning(
            f"rope_theta ({config.rope_theta}) is outside typical range [1000, 10000000]"
        )

    # If there are errors, raise ValueError with all error messages
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ValueError(error_msg)

    logger.debug(
        "Configuration validation passed",
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
    )


def get_default_config(model_size: str) -> LlamaConfig:
    """Get default configuration for a specific model size.

    This function returns a pre-configured LlamaConfig for common model sizes.
    Useful when config.json is not available or as a fallback.

    Args:
        model_size: Model size string (e.g., "3B", "8B", "70B")

    Returns:
        LlamaConfig instance with default parameters for the size

    Raises:
        ValueError: If model_size is not recognized

    Example:
        >>> config = get_default_config("3B")
        >>> print(f"3B model has {config.num_hidden_layers} layers")
    """
    # Normalize model size string
    model_size_upper = model_size.upper()

    if model_size_upper not in DEFAULT_CONFIGS:
        available = ", ".join(DEFAULT_CONFIGS.keys())
        raise ValueError(
            f"Unknown model size: {model_size}. Available sizes: {available}"
        )

    config_dict = DEFAULT_CONFIGS[model_size_upper]
    config = LlamaConfig(**config_dict)

    # Validate configuration (should always pass for default configs)
    validate_config(config)

    logger.info(
        f"Using default config for {model_size} model",
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
    )

    return config


def infer_model_size_from_config(config: LlamaConfig) -> str:
    """Infer model size string from configuration.

    This function attempts to determine the model size (e.g., "3B", "8B")
    based on the configuration parameters, primarily hidden_size.

    Args:
        config: LlamaConfig instance

    Returns:
        Model size string (e.g., "3B", "8B", "unknown")

    Example:
        >>> config = LlamaConfig(hidden_size=3072, ...)
        >>> size = infer_model_size_from_config(config)
        >>> print(size)  # "3B"
    """
    # Match based on hidden_size (most distinctive parameter)
    for size, default_config in DEFAULT_CONFIGS.items():
        if config.hidden_size == default_config["hidden_size"]:
            return size

    # If no exact match, estimate based on hidden_size
    if config.hidden_size < 1536:
        return "0.5B"
    elif config.hidden_size < 2560:
        return "1B"
    elif config.hidden_size < 3584:
        return "3B"
    elif config.hidden_size < 6144:
        return "8B"
    else:
        return "70B+"

    return "unknown"


# ============================================================================
# Core Transformer Components
# ============================================================================


class RMSNorm:
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes the input using the root mean square statistic,
    which is simpler and more efficient than LayerNorm while maintaining
    similar performance.

    The normalization is computed as:
        output = input * rsqrt(mean(input^2) + eps) * weight

    Reference: https://arxiv.org/abs/1910.07467

    Attributes:
        weight: Learnable scale parameter of shape (hidden_size,)
        eps: Small constant for numerical stability

    Example:
        >>> from tinygrad import Tensor
        >>> norm = RMSNorm(hidden_size=3072, eps=1e-5)
        >>> x = Tensor.randn(1, 10, 3072)
        >>> output = norm(x)
        >>> output.shape
        (1, 10, 3072)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """Initialize RMSNorm layer.

        Args:
            hidden_size: Dimension of the input features
            eps: Small constant added to denominator for numerical stability
        """
        from tinygrad import Tensor

        self.eps = eps
        # Initialize weight to ones (will be loaded from checkpoint)
        self.weight = Tensor.ones(hidden_size)

    def __call__(self, hidden_states: "Tensor") -> "Tensor":
        """Apply RMSNorm to input tensor.

        Args:
            hidden_states: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape as input

        Example:
            >>> x = Tensor.randn(2, 10, 3072)
            >>> output = norm(x)
        """
        from tinygrad import Tensor

        # Compute variance (mean of squares)
        # hidden_states: (..., hidden_size)
        variance = (hidden_states * hidden_states).mean(axis=-1, keepdim=True)

        # Normalize: x / sqrt(variance + eps)
        hidden_states = hidden_states * (variance + self.eps).rsqrt()

        # Scale by learnable weight
        return self.weight * hidden_states



class Embedding:
    """Token embedding layer.

    This layer maps token IDs to dense vector representations. It's essentially
    a lookup table where each token ID indexes into a learned embedding matrix.

    Attributes:
        weight: Embedding matrix of shape (vocab_size, hidden_size)

    Example:
        >>> from tinygrad import Tensor
        >>> embed = Embedding(vocab_size=128256, hidden_size=3072)
        >>> token_ids = Tensor([1, 2, 3, 4])
        >>> embeddings = embed(token_ids)
        >>> embeddings.shape
        (4, 3072)
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        """Initialize embedding layer.

        Args:
            vocab_size: Size of the vocabulary (number of possible tokens)
            hidden_size: Dimension of the embedding vectors
        """
        from tinygrad import Tensor

        # Initialize weight matrix (will be loaded from checkpoint)
        # Shape: (vocab_size, hidden_size)
        self.weight = Tensor.zeros(vocab_size, hidden_size)

    def __call__(self, input_ids: "Tensor") -> "Tensor":
        """Look up embeddings for input token IDs.

        Args:
            input_ids: Tensor of token IDs with shape (batch_size, seq_len)
                      or any shape (...,) containing integer token IDs

        Returns:
            Embedding tensor of shape (..., hidden_size)

        Example:
            >>> token_ids = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
            >>> embeddings = embed(token_ids)  # (2, 3, hidden_size)
        """
        # Tinygrad's indexing: weight[input_ids] performs the lookup
        return self.weight[input_ids]



class Linear:
    """Linear transformation layer (fully connected layer).

    Applies a linear transformation to the input: y = xW^T + b
    where W is the weight matrix and b is an optional bias vector.

    Most Llama layers use bias=False for efficiency.

    Attributes:
        weight: Weight matrix of shape (out_features, in_features)
        bias: Optional bias vector of shape (out_features,)

    Example:
        >>> from tinygrad import Tensor
        >>> linear = Linear(in_features=3072, out_features=8192, bias=False)
        >>> x = Tensor.randn(2, 10, 3072)
        >>> output = linear(x)
        >>> output.shape
        (2, 10, 8192)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term (default: True)
        """
        from tinygrad import Tensor

        # Initialize weight matrix (will be loaded from checkpoint)
        # Shape: (out_features, in_features) for efficient matmul
        self.weight = Tensor.zeros(out_features, in_features)

        # Initialize bias if requested
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x: "Tensor") -> "Tensor":
        """Apply linear transformation to input.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)

        Example:
            >>> x = Tensor.randn(2, 10, 3072)
            >>> output = linear(x)  # (2, 10, 8192)
        """
        # Matrix multiplication: x @ W^T
        # x: (..., in_features)
        # weight: (out_features, in_features)
        # output: (..., out_features)
        output = x @ self.weight.T

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output


class RotaryEmbedding:
    """Rotary Position Embeddings (RoPE).

    RoPE encodes positional information by rotating query and key vectors
    in the complex plane. This allows the model to naturally handle relative
    positions and extrapolate to longer sequences.

    The rotation is applied in pairs of dimensions, treating each pair as
    a complex number. For position m and dimension pair (2i, 2i+1), the
    rotation angle is m * theta^(-2i/d) where d is the head dimension.

    Reference: https://arxiv.org/abs/2104.09864

    Attributes:
        dim: Dimension of each attention head
        theta: Base frequency for computing rotation angles
        max_seq_len: Maximum sequence length for caching
        cos_cached: Cached cosine values for efficiency
        sin_cached: Cached sine values for efficiency

    Example:
        >>> from tinygrad import Tensor
        >>> rope = RotaryEmbedding(dim=128, theta=10000.0)
        >>> q = Tensor.randn(1, 10, 8, 128)  # (batch, seq, heads, dim)
        >>> k = Tensor.randn(1, 10, 8, 128)
        >>> q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        max_seq_len: int = 8192,
    ):
        """Initialize rotary embedding layer.

        Args:
            dim: Dimension of each attention head (must be even)
            theta: Base frequency for rotation angles (default: 10000.0)
            max_seq_len: Maximum sequence length to pre-compute (default: 8192)

        Raises:
            ValueError: If dim is not even
        """
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")

        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Pre-compute and cache cos/sin values for efficiency
        self._compute_cos_sin_cache(max_seq_len)

        logger.debug(
            "Initialized RotaryEmbedding",
            dim=dim,
            theta=theta,
            max_seq_len=max_seq_len,
        )

    def _compute_cos_sin_cache(self, seq_len: int) -> None:
        """Compute and cache cosine and sine values for all positions.

        This method pre-computes the rotation angles for all positions up to
        seq_len, storing cos and sin values for efficient lookup during forward pass.

        The frequency for dimension pair i is: theta^(-2i/dim)
        The rotation angle for position m and dimension pair i is: m * freq_i

        Args:
            seq_len: Sequence length to compute cache for
        """
        from tinygrad import Tensor
        import numpy as np

        # Compute frequency bands: theta^(-2i/dim) for i in [0, dim/2)
        # Shape: (dim/2,)
        inv_freq = 1.0 / (
            self.theta ** (np.arange(0, self.dim, 2, dtype=np.float32) / self.dim)
        )

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        positions = np.arange(seq_len, dtype=np.float32)

        # Compute rotation angles: outer product of positions and frequencies
        # Shape: (seq_len, dim/2)
        angles = np.outer(positions, inv_freq)

        # Compute cos and sin for all angles
        # We need to repeat each value twice to match the dimension pairs
        # angles: (seq_len, dim/2) -> (seq_len, dim/2, 1) -> (seq_len, dim)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Repeat each frequency twice: [cos(θ₀), cos(θ₀), cos(θ₁), cos(θ₁), ...]
        # This matches the dimension pairs in the input tensor
        cos_repeated = np.repeat(cos_angles, 2, axis=1)  # (seq_len, dim)
        sin_repeated = np.repeat(sin_angles, 2, axis=1)  # (seq_len, dim)

        # Convert to tinygrad tensors and cache
        self.cos_cached = Tensor(cos_repeated, requires_grad=False)
        self.sin_cached = Tensor(sin_repeated, requires_grad=False)

        logger.debug(
            "Computed RoPE cache",
            seq_len=seq_len,
            cos_shape=self.cos_cached.shape,
            sin_shape=self.sin_cached.shape,
        )

    def __call__(
        self,
        query: "Tensor",
        key: "Tensor",
        position_ids: "Tensor | None" = None,
    ) -> tuple["Tensor", "Tensor"]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            query: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            key: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
            position_ids: Optional position IDs of shape (batch, seq_len).
                         If None, uses sequential positions [0, 1, 2, ...]

        Returns:
            Tuple of (rotated_query, rotated_key) with same shapes as inputs

        Example:
            >>> q = Tensor.randn(2, 10, 8, 128)
            >>> k = Tensor.randn(2, 10, 8, 128)
            >>> q_rot, k_rot = rope(q, k)
        """
        from tinygrad import Tensor

        batch_size, seq_len, num_heads, head_dim = query.shape

        # Extend cache if needed
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds cached length {self.max_seq_len}. "
                "Recomputing cache (this may be slow)."
            )
            self.max_seq_len = seq_len
            self._compute_cos_sin_cache(seq_len)

        # Get cos/sin values for the current sequence
        if position_ids is None:
            # Use sequential positions: [0, 1, 2, ..., seq_len-1]
            cos = self.cos_cached[:seq_len]  # (seq_len, dim)
            sin = self.sin_cached[:seq_len]  # (seq_len, dim)
        else:
            # Use custom position IDs (for KV cache scenarios)
            # position_ids: (batch, seq_len)
            cos = self.cos_cached[position_ids]  # (batch, seq_len, dim)
            sin = self.sin_cached[position_ids]  # (batch, seq_len, dim)

        # Reshape cos/sin to match query/key dimensions
        # cos/sin: (seq_len, dim) or (batch, seq_len, dim)
        # Need: (batch, seq_len, 1, dim) for broadcasting
        if position_ids is None:
            cos = cos.reshape(1, seq_len, 1, head_dim)
            sin = sin.reshape(1, seq_len, 1, head_dim)
        else:
            cos = cos.reshape(batch_size, seq_len, 1, head_dim)
            sin = sin.reshape(batch_size, seq_len, 1, head_dim)

        # Apply rotation to query and key
        query_rotated = apply_rotary_emb(query, cos, sin)
        key_rotated = apply_rotary_emb(key, cos, sin)

        return query_rotated, key_rotated


def apply_rotary_emb(
    x: "Tensor",
    cos: "Tensor",
    sin: "Tensor",
) -> "Tensor":
    """Apply rotary embeddings to input tensor.

    This function performs the rotation in real space by treating pairs of
    dimensions as complex numbers. For a complex number (a, b), rotation by
    angle θ is computed as:
        real: a * cos(θ) - b * sin(θ)
        imag: a * sin(θ) + b * cos(θ)

    The input tensor is split into pairs of dimensions, rotated, and concatenated.

    Args:
        x: Input tensor of shape (..., seq_len, num_heads, head_dim)
        cos: Cosine values of shape (batch, seq_len, 1, head_dim) or (1, seq_len, 1, head_dim)
        sin: Sine values of shape (batch, seq_len, 1, head_dim) or (1, seq_len, 1, head_dim)

    Returns:
        Rotated tensor of same shape as input

    Example:
        >>> x = Tensor.randn(2, 10, 8, 128)
        >>> cos = Tensor.randn(1, 10, 1, 128)
        >>> sin = Tensor.randn(1, 10, 1, 128)
        >>> x_rot = apply_rotary_emb(x, cos, sin)
    """
    from tinygrad import Tensor

    # Split x into two halves for complex number representation
    # x: (..., head_dim) where head_dim is even
    # x1: (..., head_dim/2) - "real" parts (even indices: 0, 2, 4, ...)
    # x2: (..., head_dim/2) - "imag" parts (odd indices: 1, 3, 5, ...)

    *batch_dims, seq_len, num_heads, head_dim = x.shape

    # Reshape to separate even and odd dimensions
    # x: (..., seq_len, num_heads, head_dim)
    # -> (..., seq_len, num_heads, head_dim/2, 2)
    x_reshaped = x.reshape(*batch_dims, seq_len, num_heads, head_dim // 2, 2)

    # Extract even and odd dimensions
    x1 = x_reshaped[..., 0]  # (..., seq_len, num_heads, head_dim/2) - even indices
    x2 = x_reshaped[..., 1]  # (..., seq_len, num_heads, head_dim/2) - odd indices

    # Similarly reshape cos and sin
    # cos/sin: (batch, seq_len, 1, head_dim) or (1, seq_len, 1, head_dim)
    cos_reshaped = cos.reshape(*cos.shape[:-1], head_dim // 2, 2)
    sin_reshaped = sin.reshape(*sin.shape[:-1], head_dim // 2, 2)

    cos1 = cos_reshaped[..., 0]  # even indices
    cos2 = cos_reshaped[..., 1]  # odd indices (should be same as cos1)
    sin1 = sin_reshaped[..., 0]  # even indices
    sin2 = sin_reshaped[..., 1]  # odd indices (should be same as sin1)

    # Apply rotation formula:
    # real_rotated = real * cos - imag * sin
    # imag_rotated = real * sin + imag * cos
    x1_rotated = x1 * cos1 - x2 * sin1
    x2_rotated = x1 * sin2 + x2 * cos2

    # Stack and reshape back to original shape
    # Stack: (..., seq_len, num_heads, head_dim/2, 2)
    x_rotated = Tensor.stack([x1_rotated, x2_rotated], dim=-1)

    # Reshape back: (..., seq_len, num_heads, head_dim)
    x_rotated = x_rotated.reshape(*batch_dims, seq_len, num_heads, head_dim)

    return x_rotated


class Attention:
    """Multi-head attention with rotary position embeddings.

    This class implements the multi-head attention mechanism used in Llama models,
    with support for grouped-query attention (GQA) where the number of key-value
    heads can be less than the number of query heads.

    The attention mechanism computes:
        1. Project input to queries, keys, and values
        2. Apply rotary position embeddings to queries and keys
        3. Compute scaled dot-product attention: softmax(QK^T / sqrt(d)) V
        4. Project output back to hidden dimension

    For grouped-query attention, key and value heads are repeated to match
    the number of query heads before computing attention.

    Reference: https://arxiv.org/abs/2307.09288 (Llama 2)

    Attributes:
        num_heads: Number of query attention heads
        num_kv_heads: Number of key-value heads (for GQA)
        head_dim: Dimension of each attention head
        q_proj: Linear projection for queries
        k_proj: Linear projection for keys
        v_proj: Linear projection for values
        o_proj: Linear projection for output
        rope: Rotary position embedding layer

    Example:
        >>> from tinygrad import Tensor
        >>> config = LlamaConfig(hidden_size=3072, num_attention_heads=24, num_key_value_heads=8)
        >>> attn = Attention(config)
        >>> hidden_states = Tensor.randn(2, 10, 3072)
        >>> output = attn(hidden_states)
        >>> output.shape
        (2, 10, 3072)
    """

    def __init__(self, config: LlamaConfig):
        """Initialize attention layer.

        Args:
            config: Model configuration containing attention parameters
        """
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Validate configuration
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_kv_heads})"
            )

        # Initialize projection layers
        # Q projection: hidden_size -> num_heads * head_dim
        self.q_proj = Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )

        # K and V projections: hidden_size -> num_kv_heads * head_dim
        self.k_proj = Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )

        # Output projection: num_heads * head_dim -> hidden_size
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # Rotary position embeddings
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
        )

        logger.debug(
            "Initialized Attention layer",
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
        )

    def __call__(
        self,
        hidden_states: "Tensor",
        cache: "LayerCache | None" = None,
        position_ids: "Tensor | None" = None,
        attention_mask: "Tensor | None" = None,
    ) -> tuple["Tensor", "LayerCache | None"]:
        """Compute multi-head attention with optional KV caching.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            cache: Optional LayerCache for storing/retrieving cached keys and values
            position_ids: Optional position IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, 1, seq_len, kv_seq_len)
                           or (batch_size, seq_len, kv_seq_len). Values should be 0 for positions
                           to attend to and large negative values for positions to mask out.

        Returns:
            Tuple of (output, cache):
            - output: Attention output of shape (batch_size, seq_len, hidden_size)
            - cache: Updated LayerCache (or None if no cache was provided)

        Example:
            >>> hidden_states = Tensor.randn(2, 10, 3072)
            >>> cache = LayerCache()
            >>> output, updated_cache = attn(hidden_states, cache=cache)
            >>> output.shape
            (2, 10, 3072)
        """
        from tinygrad import Tensor

        batch_size, seq_len, _ = hidden_states.shape

        # Project to queries, keys, and values
        # Q: (batch_size, seq_len, num_heads * head_dim)
        query = self.q_proj(hidden_states)
        # K, V: (batch_size, seq_len, num_kv_heads * head_dim)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # Q: (batch_size, seq_len, num_heads, head_dim)
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # K, V: (batch_size, seq_len, num_kv_heads, head_dim)
        key = key.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary position embeddings to Q and K
        query, key = self.rope(query, key, position_ids)

        # Update KV cache if provided
        if cache is not None:
            # Append new keys and values to cache
            key = cache.update_key(key)
            value = cache.update_value(value)

        # Handle grouped-query attention: repeat K and V heads if needed
        if self.num_heads != self.num_kv_heads:
            # Repeat each KV head (num_heads // num_kv_heads) times
            n_rep = self.num_heads // self.num_kv_heads
            key = repeat_kv(key, n_rep)
            value = repeat_kv(value, n_rep)

        # Compute scaled dot-product attention
        # query: (batch_size, seq_len, num_heads, head_dim)
        # key, value: (batch_size, kv_seq_len, num_heads, head_dim) where kv_seq_len >= seq_len
        attn_output = scaled_dot_product_attention(
            query, key, value, attention_mask, self.head_dim
        )

        # Reshape output: (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, seq_len, num_heads * head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Project output
        output = self.o_proj(attn_output)

        return output, cache


def repeat_kv(hidden_states: "Tensor", n_rep: int) -> "Tensor":
    """Repeat key or value heads for grouped-query attention.

    In grouped-query attention (GQA), the number of key-value heads is less than
    the number of query heads. This function repeats each KV head n_rep times to
    match the number of query heads.

    For example, if we have 24 query heads and 8 KV heads, each KV head is
    repeated 3 times (24 // 8 = 3).

    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
        n_rep: Number of times to repeat each head (num_heads // num_kv_heads)

    Returns:
        Tensor of shape (batch_size, seq_len, num_kv_heads * n_rep, head_dim)

    Example:
        >>> kv = Tensor.randn(2, 10, 8, 128)  # 8 KV heads
        >>> kv_repeated = repeat_kv(kv, 3)     # Repeat 3 times
        >>> kv_repeated.shape
        (2, 10, 24, 128)  # Now 24 heads
    """
    from tinygrad import Tensor

    if n_rep == 1:
        return hidden_states

    batch_size, seq_len, num_kv_heads, head_dim = hidden_states.shape

    # Expand and reshape to repeat each head n_rep times
    # Method: Add a new dimension and repeat, then reshape
    # (batch_size, seq_len, num_kv_heads, head_dim)
    # -> (batch_size, seq_len, num_kv_heads, 1, head_dim)
    # -> (batch_size, seq_len, num_kv_heads, n_rep, head_dim)
    # -> (batch_size, seq_len, num_kv_heads * n_rep, head_dim)

    hidden_states = hidden_states.unsqueeze(3)  # Add dimension for repetition
    hidden_states = hidden_states.expand(
        batch_size, seq_len, num_kv_heads, n_rep, head_dim
    )
    hidden_states = hidden_states.reshape(
        batch_size, seq_len, num_kv_heads * n_rep, head_dim
    )

    return hidden_states


def scaled_dot_product_attention(
    query: "Tensor",
    key: "Tensor",
    value: "Tensor",
    attention_mask: "Tensor | None" = None,
    head_dim: int = 128,
) -> "Tensor":
    """Compute scaled dot-product attention.

    This function implements the core attention mechanism:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        query: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        key: Key tensor of shape (batch_size, kv_seq_len, num_heads, head_dim)
        value: Value tensor of shape (batch_size, kv_seq_len, num_heads, head_dim)
        attention_mask: Optional mask of shape (batch_size, 1, seq_len, kv_seq_len)
                       or (batch_size, seq_len, kv_seq_len). Values should be 0 for
                       positions to attend to and large negative values to mask out.
        head_dim: Dimension of each attention head (for scaling)

    Returns:
        Attention output of shape (batch_size, seq_len, num_heads, head_dim)

    Example:
        >>> q = Tensor.randn(2, 10, 8, 128)
        >>> k = Tensor.randn(2, 10, 8, 128)
        >>> v = Tensor.randn(2, 10, 8, 128)
        >>> output = scaled_dot_product_attention(q, k, v)
        >>> output.shape
        (2, 10, 8, 128)
    """
    from tinygrad import Tensor
    import math

    batch_size, seq_len, num_heads, head_dim_actual = query.shape
    _, kv_seq_len, _, _ = key.shape

    # Transpose for batch matrix multiplication
    # query: (batch_size, seq_len, num_heads, head_dim)
    # -> (batch_size, num_heads, seq_len, head_dim)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Compute attention scores: Q @ K^T
    # query: (batch_size, num_heads, seq_len, head_dim)
    # key^T: (batch_size, num_heads, head_dim, kv_seq_len)
    # scores: (batch_size, num_heads, seq_len, kv_seq_len)
    attn_scores = query @ key.transpose(-2, -1)

    # Scale by sqrt(head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = attn_scores * scale

    # Apply attention mask if provided
    if attention_mask is not None:
        # Ensure mask has correct shape for broadcasting
        # Expected: (batch_size, 1, seq_len, kv_seq_len) or (batch_size, num_heads, seq_len, kv_seq_len)
        if attention_mask.ndim == 3:
            # (batch_size, seq_len, kv_seq_len) -> (batch_size, 1, seq_len, kv_seq_len)
            attention_mask = attention_mask.unsqueeze(1)

        # Add mask to scores (mask should contain large negative values for positions to ignore)
        attn_scores = attn_scores + attention_mask

    # Apply softmax to get attention weights
    # attn_weights: (batch_size, num_heads, seq_len, kv_seq_len)
    attn_weights = attn_scores.softmax(axis=-1)

    # Apply attention weights to values
    # attn_weights: (batch_size, num_heads, seq_len, kv_seq_len)
    # value: (batch_size, num_heads, kv_seq_len, head_dim)
    # output: (batch_size, num_heads, seq_len, head_dim)
    attn_output = attn_weights @ value

    # Transpose back to original format
    # (batch_size, num_heads, seq_len, head_dim)
    # -> (batch_size, seq_len, num_heads, head_dim)
    attn_output = attn_output.transpose(1, 2)

    return attn_output


class MLP:
    """Feed-forward network with SwiGLU activation.

    This class implements the feed-forward network (FFN) used in Llama models,
    which uses the SwiGLU activation function. SwiGLU is a gated linear unit
    variant that has been shown to improve model performance.

    The computation is:
        FFN(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))

    where ⊙ denotes element-wise multiplication and SiLU (Swish) is defined as:
        SiLU(x) = x * sigmoid(x)

    Reference: https://arxiv.org/abs/2002.05202 (GLU Variants)

    Attributes:
        gate_proj: Linear projection for gating (hidden_size -> intermediate_size)
        up_proj: Linear projection for values (hidden_size -> intermediate_size)
        down_proj: Linear projection back to hidden size (intermediate_size -> hidden_size)

    Example:
        >>> from tinygrad import Tensor
        >>> config = LlamaConfig(hidden_size=3072, intermediate_size=8192)
        >>> mlp = MLP(config)
        >>> x = Tensor.randn(2, 10, 3072)
        >>> output = mlp(x)
        >>> output.shape
        (2, 10, 3072)
    """

    def __init__(self, config: LlamaConfig):
        """Initialize MLP layer.

        Args:
            config: Model configuration containing MLP parameters
        """
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Initialize projection layers (all without bias)
        # Gate projection: used for gating mechanism
        self.gate_proj = Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )

        # Up projection: used for values
        self.up_proj = Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )

        # Down projection: projects back to hidden size
        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )

        logger.debug(
            "Initialized MLP layer",
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
        )

    def __call__(self, hidden_states: "Tensor") -> "Tensor":
        """Apply feed-forward network with SwiGLU activation.

        This method implements the SwiGLU activation:
            output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)

        Example:
            >>> x = Tensor.randn(2, 10, 3072)
            >>> output = mlp(x)
            >>> output.shape
            (2, 10, 3072)
        """
        # Apply gate projection and SiLU activation
        # gate: (batch_size, seq_len, intermediate_size)
        gate = self.gate_proj(hidden_states)
        gate = silu(gate)

        # Apply up projection
        # up: (batch_size, seq_len, intermediate_size)
        up = self.up_proj(hidden_states)

        # Element-wise multiplication (gating)
        # gated: (batch_size, seq_len, intermediate_size)
        gated = gate * up

        # Project back to hidden size
        # output: (batch_size, seq_len, hidden_size)
        output = self.down_proj(gated)

        return output


def silu(x: "Tensor") -> "Tensor":
    """SiLU (Swish) activation function.

    SiLU is defined as: SiLU(x) = x * sigmoid(x)

    This activation function is smooth and non-monotonic, which has been
    shown to improve model performance compared to ReLU in many cases.

    Reference: https://arxiv.org/abs/1710.05941

    Args:
        x: Input tensor of any shape

    Returns:
        Output tensor of same shape as input

    Example:
        >>> x = Tensor.randn(2, 10, 8192)
        >>> output = silu(x)
        >>> output.shape
        (2, 10, 8192)
    """
    # SiLU(x) = x * sigmoid(x)
    return x * x.sigmoid()



class TransformerLayer:
    """Single Llama transformer layer.

    This class implements one layer of the Llama transformer, consisting of:
    1. Pre-attention RMSNorm
    2. Multi-head self-attention with residual connection
    3. Pre-MLP RMSNorm
    4. Feed-forward network (MLP) with residual connection

    The layer uses pre-normalization (norm before attention/MLP) rather than
    post-normalization, which has been shown to improve training stability.

    Reference: https://arxiv.org/abs/2307.09288 (Llama 2)

    Attributes:
        input_layernorm: RMSNorm applied before attention
        self_attn: Multi-head attention layer
        post_attention_layernorm: RMSNorm applied before MLP
        mlp: Feed-forward network

    Example:
        >>> from tinygrad import Tensor
        >>> config = LlamaConfig(hidden_size=3072, num_hidden_layers=28)
        >>> layer = TransformerLayer(config)
        >>> hidden_states = Tensor.randn(2, 10, 3072)
        >>> output = layer(hidden_states)
        >>> output.shape
        (2, 10, 3072)
    """

    def __init__(self, config: LlamaConfig):
        """Initialize transformer layer.

        Args:
            config: Model configuration containing layer parameters
        """
        self.hidden_size = config.hidden_size

        # Pre-attention layer normalization
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Multi-head self-attention
        self.self_attn = Attention(config)

        # Pre-MLP layer normalization
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Feed-forward network
        self.mlp = MLP(config)

        logger.debug(
            "Initialized TransformerLayer",
            hidden_size=self.hidden_size,
        )

    def __call__(
        self,
        hidden_states: "Tensor",
        cache: "LayerCache | None" = None,
        position_ids: "Tensor | None" = None,
        attention_mask: "Tensor | None" = None,
    ) -> tuple["Tensor", "LayerCache | None"]:
        """Forward pass through transformer layer with residual connections.

        This method implements the standard transformer layer computation:
        1. Apply pre-attention norm
        2. Compute self-attention with optional KV caching
        3. Add residual connection
        4. Apply pre-MLP norm
        5. Compute MLP
        6. Add residual connection

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            cache: Optional LayerCache for storing/retrieving cached keys and values
            position_ids: Optional position IDs of shape (batch_size, seq_len)
                         for rotary position embeddings
            attention_mask: Optional attention mask of shape (batch_size, 1, seq_len, kv_seq_len)
                           or (batch_size, seq_len, kv_seq_len)

        Returns:
            Tuple of (output, cache):
            - output: Output tensor of shape (batch_size, seq_len, hidden_size)
            - cache: Updated LayerCache (or None if no cache was provided)

        Example:
            >>> hidden_states = Tensor.randn(2, 10, 3072)
            >>> cache = LayerCache()
            >>> output, updated_cache = layer(hidden_states, cache=cache)
            >>> output.shape
            (2, 10, 3072)
        """
        # Self-attention block with residual connection
        # Save residual before normalization
        residual = hidden_states

        # Apply pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Compute self-attention with cache
        hidden_states, cache = self.self_attn(
            hidden_states,
            cache=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        # Add residual connection
        hidden_states = residual + hidden_states

        # MLP block with residual connection
        # Save residual before normalization
        residual = hidden_states

        # Apply pre-MLP normalization
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Compute MLP
        hidden_states = self.mlp(hidden_states)

        # Add residual connection
        hidden_states = residual + hidden_states

        return hidden_states, cache


# ============================================================================
# KV Cache System
# ============================================================================


class LayerCache:
    """Cache for key-value tensors in a single transformer layer.

    This class stores the key and value tensors from previous forward passes
    to enable efficient autoregressive generation. Instead of recomputing
    attention for all previous tokens, we cache and reuse them.

    Attributes:
        key_cache: Cached key tensor of shape (batch_size, cached_seq_len, num_kv_heads, head_dim)
        value_cache: Cached value tensor of shape (batch_size, cached_seq_len, num_kv_heads, head_dim)

    Example:
        >>> from tinygrad import Tensor
        >>> cache = LayerCache()
        >>> # First token
        >>> k1 = Tensor.randn(1, 1, 8, 128)
        >>> v1 = Tensor.randn(1, 1, 8, 128)
        >>> k_cached = cache.update_key(k1)
        >>> v_cached = cache.update_value(v1)
        >>> # Second token - cache is concatenated
        >>> k2 = Tensor.randn(1, 1, 8, 128)
        >>> k_cached = cache.update_key(k2)  # Now has shape (1, 2, 8, 128)
    """

    def __init__(self) -> None:
        """Initialize empty layer cache."""
        self.key_cache: "Tensor | None" = None
        self.value_cache: "Tensor | None" = None

    def update_key(self, new_key: "Tensor") -> "Tensor":
        """Append new keys to cache and return full cached keys.

        Args:
            new_key: New key tensor of shape (batch_size, new_seq_len, num_kv_heads, head_dim)

        Returns:
            Full cached key tensor of shape (batch_size, total_seq_len, num_kv_heads, head_dim)
            where total_seq_len = cached_seq_len + new_seq_len

        Example:
            >>> k1 = Tensor.randn(1, 1, 8, 128)
            >>> k_full = cache.update_key(k1)  # (1, 1, 8, 128)
            >>> k2 = Tensor.randn(1, 1, 8, 128)
            >>> k_full = cache.update_key(k2)  # (1, 2, 8, 128)
        """
        from tinygrad import Tensor

        if self.key_cache is None:
            # First token: initialize cache
            self.key_cache = new_key
        else:
            # Subsequent tokens: concatenate along sequence dimension
            self.key_cache = Tensor.cat([self.key_cache, new_key], dim=1)

        return self.key_cache

    def update_value(self, new_value: "Tensor") -> "Tensor":
        """Append new values to cache and return full cached values.

        Args:
            new_value: New value tensor of shape (batch_size, new_seq_len, num_kv_heads, head_dim)

        Returns:
            Full cached value tensor of shape (batch_size, total_seq_len, num_kv_heads, head_dim)
            where total_seq_len = cached_seq_len + new_seq_len

        Example:
            >>> v1 = Tensor.randn(1, 1, 8, 128)
            >>> v_full = cache.update_value(v1)  # (1, 1, 8, 128)
            >>> v2 = Tensor.randn(1, 1, 8, 128)
            >>> v_full = cache.update_value(v2)  # (1, 2, 8, 128)
        """
        from tinygrad import Tensor

        if self.value_cache is None:
            # First token: initialize cache
            self.value_cache = new_value
        else:
            # Subsequent tokens: concatenate along sequence dimension
            self.value_cache = Tensor.cat([self.value_cache, new_value], dim=1)

        return self.value_cache

    def get_seq_length(self) -> int:
        """Get the current sequence length in the cache.

        Returns:
            Number of cached tokens (0 if cache is empty)

        Example:
            >>> cache = LayerCache()
            >>> cache.get_seq_length()
            0
            >>> cache.update_key(Tensor.randn(1, 5, 8, 128))
            >>> cache.get_seq_length()
            5
        """
        if self.key_cache is None:
            return 0
        return self.key_cache.shape[1]

    def clear(self) -> None:
        """Clear the cache, freeing memory.

        Example:
            >>> cache.clear()
            >>> cache.get_seq_length()
            0
        """
        self.key_cache = None
        self.value_cache = None


class KVCache:
    """Manages key-value cache for all transformer layers.

    This class maintains separate LayerCache instances for each transformer layer
    and tracks the overall cache state. It supports per-request cache management
    for handling multiple concurrent generation requests.

    Attributes:
        num_layers: Number of transformer layers
        layer_caches: List of LayerCache instances, one per layer
        request_id: Optional request ID for tracking cache ownership

    Example:
        >>> config = LlamaConfig(num_hidden_layers=28)
        >>> cache = KVCache(num_layers=config.num_hidden_layers)
        >>> # Get cache for layer 0
        >>> layer_0_cache = cache.get_layer_cache(0)
        >>> # Update cache for layer 0
        >>> k = Tensor.randn(1, 1, 8, 128)
        >>> layer_0_cache.update_key(k)
    """

    def __init__(
        self,
        num_layers: int,
        request_id: str | None = None,
    ) -> None:
        """Initialize KV cache for all layers.

        Args:
            num_layers: Number of transformer layers
            request_id: Optional request ID for tracking cache ownership
        """
        self.num_layers = num_layers
        self.request_id = request_id
        self.layer_caches: list[LayerCache] = [
            LayerCache() for _ in range(num_layers)
        ]

        logger.debug(
            "Initialized KVCache",
            num_layers=num_layers,
            request_id=request_id,
        )

    def get_layer_cache(self, layer_idx: int) -> LayerCache:
        """Get cache for a specific layer.

        Args:
            layer_idx: Index of the layer (0 to num_layers-1)

        Returns:
            LayerCache instance for the specified layer

        Raises:
            IndexError: If layer_idx is out of range

        Example:
            >>> cache = KVCache(num_layers=28)
            >>> layer_cache = cache.get_layer_cache(0)
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"Layer index {layer_idx} out of range [0, {self.num_layers})"
            )
        return self.layer_caches[layer_idx]

    def update_layer_cache(self, layer_idx: int, layer_cache: LayerCache) -> None:
        """Update cache for a specific layer.

        This method allows replacing the entire LayerCache for a layer,
        which can be useful for advanced cache management scenarios.

        Args:
            layer_idx: Index of the layer (0 to num_layers-1)
            layer_cache: New LayerCache instance for the layer

        Raises:
            IndexError: If layer_idx is out of range

        Example:
            >>> cache = KVCache(num_layers=28)
            >>> new_layer_cache = LayerCache()
            >>> cache.update_layer_cache(0, new_layer_cache)
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"Layer index {layer_idx} out of range [0, {self.num_layers})"
            )
        self.layer_caches[layer_idx] = layer_cache

    def get_seq_length(self) -> int:
        """Get the current sequence length in the cache.

        Returns the sequence length from the first layer's cache.
        All layers should have the same sequence length.

        Returns:
            Number of cached tokens (0 if cache is empty)

        Example:
            >>> cache = KVCache(num_layers=28)
            >>> cache.get_seq_length()
            0
        """
        if not self.layer_caches:
            return 0
        return self.layer_caches[0].get_seq_length()

    def clear(self) -> None:
        """Clear all layer caches, freeing memory.

        Example:
            >>> cache.clear()
            >>> cache.get_seq_length()
            0
        """
        for layer_cache in self.layer_caches:
            layer_cache.clear()

        logger.debug(
            "Cleared KVCache",
            num_layers=self.num_layers,
            request_id=self.request_id,
        )


class KVCacheManager:
    """Manages KV caches for multiple concurrent requests.

    This class maintains a mapping from request IDs to KVCache instances,
    enabling efficient cache management for multiple concurrent generation
    requests. It handles cache creation, retrieval, and eviction.

    Attributes:
        num_layers: Number of transformer layers
        caches: Dictionary mapping request IDs to KVCache instances

    Example:
        >>> manager = KVCacheManager(num_layers=28)
        >>> # Get or create cache for a request
        >>> cache = manager.get_cache("request-123")
        >>> # Evict cache when request is complete
        >>> manager.evict_cache("request-123")
    """

    def __init__(self, num_layers: int) -> None:
        """Initialize cache manager.

        Args:
            num_layers: Number of transformer layers
        """
        self.num_layers = num_layers
        self.caches: dict[str, KVCache] = {}

        logger.debug(
            "Initialized KVCacheManager",
            num_layers=num_layers,
        )

    def get_cache(self, request_id: str) -> KVCache:
        """Get or create cache for a request.

        If a cache already exists for the request ID, it is returned.
        Otherwise, a new cache is created and stored.

        Args:
            request_id: Unique identifier for the request

        Returns:
            KVCache instance for the request

        Example:
            >>> manager = KVCacheManager(num_layers=28)
            >>> cache = manager.get_cache("request-123")
            >>> # Same cache is returned for same request ID
            >>> cache2 = manager.get_cache("request-123")
            >>> assert cache is cache2
        """
        if request_id not in self.caches:
            self.caches[request_id] = KVCache(
                num_layers=self.num_layers,
                request_id=request_id,
            )
            logger.debug(
                "Created new cache for request",
                request_id=request_id,
                total_caches=len(self.caches),
            )

        return self.caches[request_id]

    def evict_cache(self, request_id: str) -> bool:
        """Evict cache for a completed request.

        This method removes the cache for a request, freeing memory.
        It should be called when a generation request is complete.

        Args:
            request_id: Unique identifier for the request

        Returns:
            True if cache was evicted, False if no cache existed

        Example:
            >>> manager = KVCacheManager(num_layers=28)
            >>> cache = manager.get_cache("request-123")
            >>> manager.evict_cache("request-123")
            True
            >>> manager.evict_cache("request-123")
            False
        """
        if request_id in self.caches:
            cache = self.caches.pop(request_id)
            cache.clear()
            logger.debug(
                "Evicted cache for request",
                request_id=request_id,
                remaining_caches=len(self.caches),
            )
            return True

        return False

    def clear_all(self) -> None:
        """Clear all caches, freeing memory.

        This method evicts all caches for all requests.

        Example:
            >>> manager.clear_all()
        """
        for cache in self.caches.values():
            cache.clear()

        self.caches.clear()

        logger.info(
            "Cleared all caches",
            num_layers=self.num_layers,
        )

    def get_num_active_caches(self) -> int:
        """Get the number of active caches.

        Returns:
            Number of active request caches

        Example:
            >>> manager = KVCacheManager(num_layers=28)
            >>> manager.get_num_active_caches()
            0
            >>> manager.get_cache("request-123")
            >>> manager.get_num_active_caches()
            1
        """
        return len(self.caches)


# ============================================================================
# Complete Llama Transformer
# ============================================================================


class LlamaTransformer:
    """Complete Llama transformer model for text generation.

    This class implements the full Llama transformer architecture, consisting of:
    1. Token embedding layer
    2. N transformer layers (attention + MLP)
    3. Final RMSNorm
    4. Language modeling head (projection to vocabulary)

    The model supports efficient autoregressive generation using KV caching
    and can handle various Llama model sizes (1B, 3B, 8B, 70B, etc.).

    Reference: https://arxiv.org/abs/2307.09288 (Llama 2)

    Attributes:
        config: Model configuration
        embed_tokens: Token embedding layer
        layers: List of transformer layers
        norm: Final RMSNorm layer
        lm_head: Language modeling head (projects to vocabulary logits)

    Example:
        >>> from tinygrad import Tensor
        >>> config = LlamaConfig(vocab_size=128256, hidden_size=3072, num_hidden_layers=28)
        >>> model = LlamaTransformer(config)
        >>> input_ids = Tensor([[1, 2, 3, 4]])
        >>> logits, cache = model(input_ids)
        >>> logits.shape
        (1, 4, 128256)
    """

    def __init__(self, config: LlamaConfig):
        """Initialize Llama transformer model.

        Args:
            config: Model configuration containing all hyperparameters

        Example:
            >>> config = LlamaConfig(
            ...     vocab_size=128256,
            ...     hidden_size=3072,
            ...     num_hidden_layers=28,
            ...     num_attention_heads=24,
            ...     num_key_value_heads=8
            ... )
            >>> model = LlamaTransformer(config)
        """
        self.config = config

        # Token embedding layer
        self.embed_tokens = Embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        # Transformer layers
        self.layers = [
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ]

        # Final layer normalization
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Language modeling head (projects to vocabulary)
        # Note: Often shares weights with embed_tokens.weight
        self.lm_head = Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )

        logger.info(
            "Initialized LlamaTransformer",
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        input_ids: "Tensor",
        cache: KVCache | None = None,
        position_ids: "Tensor | None" = None,
        attention_mask: "Tensor | None" = None,
    ) -> tuple["Tensor", KVCache]:
        """Forward pass through the transformer.

        This method processes input token IDs through the full transformer stack:
        1. Embed tokens to dense vectors
        2. Process through all transformer layers with optional KV caching
        3. Apply final layer normalization
        4. Project to vocabulary logits

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            cache: Optional KVCache for efficient autoregressive generation.
                  If None, a new cache is created.
            position_ids: Optional position IDs of shape (batch_size, seq_len).
                         If None, sequential positions [0, 1, 2, ...] are used.
            attention_mask: Optional attention mask of shape (batch_size, 1, seq_len, kv_seq_len)
                           or (batch_size, seq_len, kv_seq_len). Values should be 0 for
                           positions to attend to and large negative values to mask out.

        Returns:
            Tuple of (logits, cache):
            - logits: Vocabulary logits of shape (batch_size, seq_len, vocab_size)
            - cache: Updated KVCache for next forward pass

        Example:
            >>> # First forward pass (prefill)
            >>> input_ids = Tensor([[1, 2, 3, 4]])  # (1, 4)
            >>> logits, cache = model(input_ids)
            >>> logits.shape
            (1, 4, 128256)
            >>> 
            >>> # Second forward pass (generation with cache)
            >>> next_token = Tensor([[5]])  # (1, 1)
            >>> logits, cache = model(next_token, cache=cache)
            >>> logits.shape
            (1, 1, 128256)
        """
        from tinygrad import Tensor

        batch_size, seq_len = input_ids.shape

        # Initialize cache if not provided
        if cache is None:
            cache = KVCache(num_layers=self.config.num_hidden_layers)

        # Generate position IDs if not provided
        if position_ids is None:
            # For first forward pass or when no cache exists
            cache_seq_len = cache.get_seq_length()
            position_ids = self._generate_position_ids(
                batch_size=batch_size,
                seq_len=seq_len,
                cache_seq_len=cache_seq_len,
            )

        # Embed input tokens
        # input_ids: (batch_size, seq_len)
        # hidden_states: (batch_size, seq_len, hidden_size)
        hidden_states = self.embed_tokens(input_ids)

        logger.debug(
            "Forward pass",
            batch_size=batch_size,
            seq_len=seq_len,
            cache_seq_len=cache.get_seq_length(),
            hidden_states_shape=hidden_states.shape,
        )

        # Process through all transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Get cache for this layer
            layer_cache = cache.get_layer_cache(layer_idx)

            # Forward through layer
            hidden_states, updated_layer_cache = layer(
                hidden_states=hidden_states,
                cache=layer_cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            # Update cache for this layer
            # Note: updated_layer_cache is the same object as layer_cache,
            # but we keep this pattern for clarity
            cache.update_layer_cache(layer_idx, updated_layer_cache)

        # Apply final layer normalization
        # hidden_states: (batch_size, seq_len, hidden_size)
        hidden_states = self.norm(hidden_states)

        # Project to vocabulary logits
        # logits: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(hidden_states)

        logger.debug(
            "Forward pass complete",
            logits_shape=logits.shape,
            final_cache_seq_len=cache.get_seq_length(),
        )

        return logits, cache

    def _generate_position_ids(
        self,
        batch_size: int,
        seq_len: int,
        cache_seq_len: int,
    ) -> "Tensor":
        """Generate position IDs based on cache state.

        This method computes position IDs for the current forward pass,
        taking into account any cached tokens from previous passes.

        For the first forward pass (prefill), position IDs are [0, 1, 2, ..., seq_len-1].
        For subsequent passes (generation), position IDs start from cache_seq_len.

        Args:
            batch_size: Batch size
            seq_len: Current sequence length
            cache_seq_len: Number of tokens already in cache

        Returns:
            Position IDs tensor of shape (batch_size, seq_len)

        Example:
            >>> # First pass: no cache
            >>> pos_ids = model._generate_position_ids(1, 4, 0)
            >>> pos_ids
            [[0, 1, 2, 3]]
            >>> 
            >>> # Second pass: 4 tokens in cache
            >>> pos_ids = model._generate_position_ids(1, 1, 4)
            >>> pos_ids
            [[4]]
        """
        from tinygrad import Tensor
        import numpy as np

        # Compute starting position based on cache
        start_pos = cache_seq_len

        # Generate sequential position IDs: [start_pos, start_pos+1, ..., start_pos+seq_len-1]
        position_ids = np.arange(start_pos, start_pos + seq_len, dtype=np.int32)

        # Expand to batch dimension: (seq_len,) -> (batch_size, seq_len)
        position_ids = np.tile(position_ids, (batch_size, 1))

        # Convert to tinygrad tensor
        position_ids_tensor = Tensor(position_ids, requires_grad=False)

        logger.debug(
            "Generated position IDs",
            batch_size=batch_size,
            seq_len=seq_len,
            cache_seq_len=cache_seq_len,
            start_pos=start_pos,
            position_ids_shape=position_ids_tensor.shape,
        )

        return position_ids_tensor

    def generate_next_token(
        self,
        input_ids: "Tensor",
        cache: KVCache | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple["Tensor", KVCache]:
        """Generate the next token given input tokens.

        This is a convenience method for single-token generation that handles
        the forward pass and sampling in one call.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            cache: Optional KVCache from previous generation steps
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold (0.0 to 1.0)

        Returns:
            Tuple of (next_token_id, cache):
            - next_token_id: Sampled token ID of shape (batch_size, 1)
            - cache: Updated KVCache

        Example:
            >>> input_ids = Tensor([[1, 2, 3]])
            >>> next_token, cache = model.generate_next_token(input_ids)
            >>> next_token.shape
            (1, 1)
        """
        # Forward pass
        logits, cache = self(input_ids, cache=cache)

        # Get logits for last token: (batch_size, vocab_size)
        last_logits = logits[:, -1, :]

        # Apply temperature
        if temperature != 1.0:
            last_logits = last_logits / temperature

        # Sample next token
        # For now, use greedy sampling (argmax)
        # TODO: Implement top-p (nucleus) sampling
        next_token_id = last_logits.argmax(axis=-1, keepdim=True)

        return next_token_id, cache


# ============================================================================
# Weight Loading Utilities
# ============================================================================


def create_weight_name_mapping(model: LlamaTransformer) -> dict[str, Any]:
    """Create mapping from HuggingFace weight names to model parameters.

    This function builds a dictionary that maps HuggingFace checkpoint weight names
    to the corresponding tinygrad model parameters. This enables loading weights
    from HuggingFace checkpoints into our custom transformer implementation.

    HuggingFace Llama weight naming convention:
    - model.embed_tokens.weight
    - model.layers.{i}.input_layernorm.weight
    - model.layers.{i}.self_attn.q_proj.weight
    - model.layers.{i}.self_attn.k_proj.weight
    - model.layers.{i}.self_attn.v_proj.weight
    - model.layers.{i}.self_attn.o_proj.weight
    - model.layers.{i}.post_attention_layernorm.weight
    - model.layers.{i}.mlp.gate_proj.weight
    - model.layers.{i}.mlp.up_proj.weight
    - model.layers.{i}.mlp.down_proj.weight
    - model.norm.weight
    - lm_head.weight

    Args:
        model: LlamaTransformer instance to create mapping for

    Returns:
        Dictionary mapping HuggingFace weight names to model parameter objects

    Example:
        >>> config = LlamaConfig(num_hidden_layers=28)
        >>> model = LlamaTransformer(config)
        >>> weight_map = create_weight_name_mapping(model)
        >>> # Access embedding weight
        >>> embed_weight = weight_map["model.embed_tokens.weight"]
        >>> # Access layer 0 attention query projection
        >>> q_proj = weight_map["model.layers.0.self_attn.q_proj.weight"]
    """
    weight_map: dict[str, Any] = {}

    # Embedding layer
    weight_map["model.embed_tokens.weight"] = model.embed_tokens.weight

    # Transformer layers
    for layer_idx, layer in enumerate(model.layers):
        prefix = f"model.layers.{layer_idx}"

        # Input layer norm
        weight_map[f"{prefix}.input_layernorm.weight"] = layer.input_layernorm.weight

        # Self-attention projections
        weight_map[f"{prefix}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight
        weight_map[f"{prefix}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight
        weight_map[f"{prefix}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight
        weight_map[f"{prefix}.self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight

        # Post-attention layer norm
        weight_map[f"{prefix}.post_attention_layernorm.weight"] = (
            layer.post_attention_layernorm.weight
        )

        # MLP projections
        weight_map[f"{prefix}.mlp.gate_proj.weight"] = layer.mlp.gate_proj.weight
        weight_map[f"{prefix}.mlp.up_proj.weight"] = layer.mlp.up_proj.weight
        weight_map[f"{prefix}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight

    # Final layer norm
    weight_map["model.norm.weight"] = model.norm.weight

    # Language modeling head
    weight_map["lm_head.weight"] = model.lm_head.weight

    logger.debug(
        "Created weight name mapping",
        num_weights=len(weight_map),
        num_layers=len(model.layers),
    )

    return weight_map


def get_parameter_from_name(
    model: LlamaTransformer,
    weight_name: str,
) -> Any | None:
    """Get model parameter by HuggingFace weight name.

    This is a convenience function that extracts the parameter object
    from the model given a HuggingFace-style weight name.

    Args:
        model: LlamaTransformer instance
        weight_name: HuggingFace weight name (e.g., "model.layers.0.self_attn.q_proj.weight")

    Returns:
        Parameter object if found, None otherwise

    Example:
        >>> model = LlamaTransformer(config)
        >>> param = get_parameter_from_name(model, "model.embed_tokens.weight")
        >>> param.shape
        (128256, 3072)
    """
    # Parse weight name to extract components
    parts = weight_name.split(".")

    try:
        if weight_name == "model.embed_tokens.weight":
            return model.embed_tokens.weight

        elif weight_name == "model.norm.weight":
            return model.norm.weight

        elif weight_name == "lm_head.weight":
            return model.lm_head.weight

        elif weight_name.startswith("model.layers."):
            # Extract layer index
            layer_idx = int(parts[2])
            layer = model.layers[layer_idx]

            # Extract component path
            component_path = ".".join(parts[3:])

            if component_path == "input_layernorm.weight":
                return layer.input_layernorm.weight

            elif component_path == "post_attention_layernorm.weight":
                return layer.post_attention_layernorm.weight

            elif component_path == "self_attn.q_proj.weight":
                return layer.self_attn.q_proj.weight

            elif component_path == "self_attn.k_proj.weight":
                return layer.self_attn.k_proj.weight

            elif component_path == "self_attn.v_proj.weight":
                return layer.self_attn.v_proj.weight

            elif component_path == "self_attn.o_proj.weight":
                return layer.self_attn.o_proj.weight

            elif component_path == "mlp.gate_proj.weight":
                return layer.mlp.gate_proj.weight

            elif component_path == "mlp.up_proj.weight":
                return layer.mlp.up_proj.weight

            elif component_path == "mlp.down_proj.weight":
                return layer.mlp.down_proj.weight

    except (IndexError, ValueError, AttributeError) as e:
        logger.warning(
            f"Failed to parse weight name: {weight_name}",
            error=str(e),
        )
        return None

    logger.warning(f"Unknown weight name: {weight_name}")
    return None



def assign_weights_to_model(
    model: LlamaTransformer,
    weights: dict[str, "np.ndarray[Any, Any]"],
    device: str,
) -> tuple[int, int]:
    """Assign loaded weights to model parameters.

    This function converts numpy weight arrays to tinygrad Tensors and assigns
    them to the corresponding model parameters. It validates weight shapes and
    provides detailed logging of the loading process.

    Args:
        model: LlamaTransformer instance to load weights into
        weights: Dictionary mapping HuggingFace weight names to numpy arrays
        device: Target device for tensors (CPU, GPU, METAL)

    Returns:
        Tuple of (num_loaded, num_expected):
        - num_loaded: Number of weights successfully loaded
        - num_expected: Total number of expected weights

    Raises:
        ValueError: If weight shape doesn't match expected parameter shape
        RuntimeError: If weight assignment fails

    Example:
        >>> model = LlamaTransformer(config)
        >>> weights = load_safetensors("model.safetensors")
        >>> num_loaded, num_expected = assign_weights_to_model(model, weights, "GPU")
        >>> print(f"Loaded {num_loaded}/{num_expected} weights")
    """
    from tinygrad import Tensor

    # Create weight name mapping
    weight_map = create_weight_name_mapping(model)

    num_loaded = 0
    num_expected = len(weight_map)
    num_shape_mismatches = 0

    logger.info(
        "Starting weight assignment",
        num_weights_available=len(weights),
        num_weights_expected=num_expected,
        device=device,
    )

    # Iterate through expected weights and load them
    for hf_name, param in weight_map.items():
        # Handle weight tying: lm_head.weight can be tied to embed_tokens.weight
        if hf_name == "lm_head.weight" and hf_name not in weights:
            if "model.embed_tokens.weight" in weights:
                logger.info("Using model.embed_tokens.weight for lm_head.weight (weight tying)")
                np_weight = weights["model.embed_tokens.weight"]
            else:
                logger.warning(f"Weight not found in checkpoint: {hf_name}")
                continue
        elif hf_name not in weights:
            logger.warning(f"Weight not found in checkpoint: {hf_name}")
            continue
        else:
            # Get numpy weight array
            np_weight = weights[hf_name]

        # Get expected shape from parameter
        expected_shape = param.shape

        # Verify shape matches
        if np_weight.shape != expected_shape:
            logger.error(
                f"Shape mismatch for {hf_name}",
                expected_shape=expected_shape,
                actual_shape=np_weight.shape,
            )
            num_shape_mismatches += 1
            continue

        try:
            # Convert numpy array to tinygrad Tensor
            # Note: tinygrad will handle device placement
            tensor = Tensor(np_weight, device=device, requires_grad=False)

            # Assign to parameter
            # In tinygrad, we replace the parameter tensor directly
            param.assign(tensor)

            num_loaded += 1

            logger.debug(
                f"Loaded weight: {hf_name}",
                shape=np_weight.shape,
                dtype=np_weight.dtype,
            )

        except Exception as e:
            logger.error(
                f"Failed to assign weight {hf_name}: {e}",
                error=str(e),
            )
            raise RuntimeError(f"Failed to assign weight {hf_name}: {e}") from e

    # Log summary
    logger.info(
        "Weight assignment complete",
        num_loaded=num_loaded,
        num_expected=num_expected,
        num_missing=num_expected - num_loaded,
        num_shape_mismatches=num_shape_mismatches,
        device=device,
    )

    if num_shape_mismatches > 0:
        raise ValueError(
            f"Found {num_shape_mismatches} weight shape mismatches. "
            "Model architecture may not match checkpoint."
        )

    return num_loaded, num_expected


def verify_weight_shapes(
    model: LlamaTransformer,
    weights: dict[str, "np.ndarray[Any, Any]"],
) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Verify that weight shapes match model parameter shapes.

    This function checks all weights against expected parameter shapes
    and returns a list of mismatches for debugging.

    Args:
        model: LlamaTransformer instance
        weights: Dictionary mapping weight names to numpy arrays

    Returns:
        List of (weight_name, expected_shape, actual_shape) tuples for mismatches

    Example:
        >>> model = LlamaTransformer(config)
        >>> weights = load_safetensors("model.safetensors")
        >>> mismatches = verify_weight_shapes(model, weights)
        >>> if mismatches:
        ...     for name, expected, actual in mismatches:
        ...         print(f"{name}: expected {expected}, got {actual}")
    """
    weight_map = create_weight_name_mapping(model)
    mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for hf_name, param in weight_map.items():
        if hf_name not in weights:
            continue

        expected_shape = param.shape
        actual_shape = weights[hf_name].shape

        if expected_shape != actual_shape:
            mismatches.append((hf_name, expected_shape, actual_shape))

    if mismatches:
        logger.warning(
            f"Found {len(mismatches)} shape mismatches",
            num_mismatches=len(mismatches),
        )
        for name, expected, actual in mismatches:
            logger.warning(
                f"Shape mismatch: {name}",
                expected=expected,
                actual=actual,
            )

    return mismatches



def load_sharded_weights(
    checkpoint_dir: Path,
    index_path: Path,
) -> dict[str, "np.ndarray[Any, Any]"]:
    """Load weights from sharded safetensors files.

    This function handles loading weights from multiple safetensors shard files
    as specified by a model.safetensors.index.json file. It combines weights
    from all shards into a single dictionary.

    The index file format:
    {
        "metadata": {...},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.input_layernorm.weight": "model-00001-of-00003.safetensors",
            ...
        }
    }

    Args:
        checkpoint_dir: Directory containing the checkpoint files
        index_path: Path to model.safetensors.index.json

    Returns:
        Dictionary mapping weight names to numpy arrays

    Raises:
        FileNotFoundError: If index file or shard files are not found
        RuntimeError: If loading fails

    Example:
        >>> checkpoint_dir = Path("/path/to/checkpoint")
        >>> index_path = checkpoint_dir / "model.safetensors.index.json"
        >>> weights = load_sharded_weights(checkpoint_dir, index_path)
        >>> print(f"Loaded {len(weights)} weights from sharded model")
    """
    import json

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    logger.info(
        "Loading sharded model",
        checkpoint_dir=str(checkpoint_dir),
        index_path=str(index_path),
    )

    # Load index file
    try:
        with open(index_path) as f:
            index = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in index file: {e}") from e

    # Get weight map: weight_name -> shard_file
    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise RuntimeError("Index file missing 'weight_map' field")

    # Get unique shard files
    shard_files = sorted(set(weight_map.values()))

    logger.info(
        f"Found {len(shard_files)} shard files",
        num_shards=len(shard_files),
        num_weights=len(weight_map),
    )

    # Load weights from all shards
    all_weights: dict[str, "np.ndarray[Any, Any]"] = {}
    
    for shard_idx, shard_file in enumerate(shard_files):
        shard_path = checkpoint_dir / shard_file

        if not shard_path.exists():
            logger.warning(f"Shard file not found: {shard_path}")
            continue

        logger.debug(
            f"Loading shard {shard_idx + 1}/{len(shard_files)}",
            shard_file=shard_file,
        )

        try:
            # Load this shard using the existing single-file loader
            # Import here to avoid circular dependency
            from exo.worker.engines.tinygrad.model_loader import _load_single_safetensors

            shard_weights = _load_single_safetensors(shard_path)

            # Only keep weights that belong to this shard according to index
            for weight_name, weight_array in shard_weights.items():
                if weight_name in weight_map and weight_map[weight_name] == shard_file:
                    all_weights[weight_name] = weight_array

            logger.debug(
                f"Loaded shard {shard_idx + 1}/{len(shard_files)}",
                num_weights_in_shard=len(shard_weights),
                num_weights_kept=sum(
                    1 for name in shard_weights if weight_map.get(name) == shard_file
                ),
            )

        except Exception as e:
            logger.error(f"Failed to load shard {shard_file}: {e}")
            raise RuntimeError(f"Failed to load shard {shard_file}: {e}") from e

    logger.info(
        "Sharded model loading complete",
        num_shards=len(shard_files),
        num_weights=len(all_weights),
    )

    return all_weights


def combine_sharded_weights(
    shard_weights_list: list[dict[str, "np.ndarray[Any, Any]"]],
) -> dict[str, "np.ndarray[Any, Any]"]:
    """Combine weights from multiple shards into a single dictionary.

    This is a utility function for combining weights that have been loaded
    from multiple shard files. It handles potential duplicates by keeping
    the first occurrence.

    Args:
        shard_weights_list: List of weight dictionaries, one per shard

    Returns:
        Combined dictionary of all weights

    Example:
        >>> shard1 = {"weight1": np.array([1, 2, 3])}
        >>> shard2 = {"weight2": np.array([4, 5, 6])}
        >>> combined = combine_sharded_weights([shard1, shard2])
        >>> len(combined)
        2
    """
    combined: dict[str, "np.ndarray[Any, Any]"] = {}
    duplicates = 0

    for shard_idx, shard_weights in enumerate(shard_weights_list):
        for weight_name, weight_array in shard_weights.items():
            if weight_name in combined:
                logger.warning(
                    f"Duplicate weight found: {weight_name}",
                    shard_idx=shard_idx,
                )
                duplicates += 1
                continue

            combined[weight_name] = weight_array

    if duplicates > 0:
        logger.warning(
            f"Found {duplicates} duplicate weights across shards",
            num_duplicates=duplicates,
        )

    logger.debug(
        "Combined sharded weights",
        num_shards=len(shard_weights_list),
        num_weights=len(combined),
    )

    return combined



def validate_weights(
    model: LlamaTransformer,
    weights: dict[str, "np.ndarray[Any, Any]"],
) -> tuple[list[str], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    """Validate loaded weights against model requirements.

    This function performs comprehensive validation of weights:
    1. Checks for missing required weights
    2. Validates weight shapes match expected parameter shapes
    3. Checks weight dtypes are compatible

    Args:
        model: LlamaTransformer instance
        weights: Dictionary mapping weight names to numpy arrays

    Returns:
        Tuple of (missing_weights, shape_mismatches):
        - missing_weights: List of required weight names that are missing
        - shape_mismatches: List of (name, expected_shape, actual_shape) tuples

    Example:
        >>> model = LlamaTransformer(config)
        >>> weights = load_safetensors("model.safetensors")
        >>> missing, mismatches = validate_weights(model, weights)
        >>> if missing:
        ...     print(f"Missing {len(missing)} required weights")
        >>> if mismatches:
        ...     print(f"Found {len(mismatches)} shape mismatches")
    """
    logger.info(
        "Validating weights",
        num_weights_available=len(weights),
    )

    # Create weight name mapping to get expected weights
    weight_map = create_weight_name_mapping(model)

    # Check for missing weights
    missing_weights: list[str] = []
    for expected_name in weight_map.keys():
        if expected_name not in weights:
            # Handle weight tying: lm_head.weight can be tied to embed_tokens.weight
            if expected_name == "lm_head.weight" and "model.embed_tokens.weight" in weights:
                logger.info("lm_head.weight is tied to model.embed_tokens.weight (weight tying)")
                continue
            missing_weights.append(expected_name)

    # Check for shape mismatches
    shape_mismatches = verify_weight_shapes(model, weights)

    # Log validation results
    if missing_weights:
        logger.warning(
            f"Found {len(missing_weights)} missing weights",
            num_missing=len(missing_weights),
        )
        for name in missing_weights[:10]:  # Log first 10
            logger.warning(f"Missing weight: {name}")
        if len(missing_weights) > 10:
            logger.warning(f"... and {len(missing_weights) - 10} more")

    if shape_mismatches:
        logger.error(
            f"Found {len(shape_mismatches)} shape mismatches",
            num_mismatches=len(shape_mismatches),
        )

    # Check dtypes
    unsupported_dtypes: list[tuple[str, str]] = []
    for name, weight_array in weights.items():
        dtype_str = str(weight_array.dtype)
        # tinygrad supports float32, float16, int32, int64, etc.
        # We mainly care about float types for model weights
        if dtype_str not in ["float32", "float16", "bfloat16", "int32", "int64"]:
            unsupported_dtypes.append((name, dtype_str))

    if unsupported_dtypes:
        logger.warning(
            f"Found {len(unsupported_dtypes)} weights with unsupported dtypes",
            num_unsupported=len(unsupported_dtypes),
        )
        for name, dtype in unsupported_dtypes[:5]:
            logger.warning(f"Unsupported dtype: {name} ({dtype})")

    # Summary
    logger.info(
        "Weight validation complete",
        num_expected=len(weight_map),
        num_available=len(weights),
        num_missing=len(missing_weights),
        num_shape_mismatches=len(shape_mismatches),
        num_unsupported_dtypes=len(unsupported_dtypes),
    )

    return missing_weights, shape_mismatches


def check_required_weights(
    model: LlamaTransformer,
    weights: dict[str, "np.ndarray[Any, Any]"],
) -> None:
    """Check that all required weights are present.

    This function validates that all critical weights needed for model
    operation are present in the loaded weights. Raises an exception
    if any required weights are missing.

    Args:
        model: LlamaTransformer instance
        weights: Dictionary mapping weight names to numpy arrays

    Raises:
        ValueError: If required weights are missing

    Example:
        >>> model = LlamaTransformer(config)
        >>> weights = load_safetensors("model.safetensors")
        >>> check_required_weights(model, weights)  # Raises if weights missing
    """
    missing_weights, shape_mismatches = validate_weights(model, weights)

    if missing_weights:
        error_msg = (
            f"Missing {len(missing_weights)} required weights. "
            f"Model cannot be loaded. Missing weights:\n"
        )
        # Include first 20 missing weights in error message
        for name in missing_weights[:20]:
            error_msg += f"  - {name}\n"
        if len(missing_weights) > 20:
            error_msg += f"  ... and {len(missing_weights) - 20} more\n"

        raise ValueError(error_msg)

    if shape_mismatches:
        error_msg = (
            f"Found {len(shape_mismatches)} weight shape mismatches. "
            f"Model architecture may not match checkpoint. Mismatches:\n"
        )
        for name, expected, actual in shape_mismatches[:10]:
            error_msg += f"  - {name}: expected {expected}, got {actual}\n"
        if len(shape_mismatches) > 10:
            error_msg += f"  ... and {len(shape_mismatches) - 10} more\n"

        raise ValueError(error_msg)

    logger.info("All required weights present and valid")


def get_weight_statistics(
    weights: dict[str, "np.ndarray[Any, Any]"],
) -> dict[str, Any]:
    """Compute statistics about loaded weights.

    This function provides useful debugging information about the weights,
    including total size, dtype distribution, and shape statistics.

    Args:
        weights: Dictionary mapping weight names to numpy arrays

    Returns:
        Dictionary containing weight statistics

    Example:
        >>> weights = load_safetensors("model.safetensors")
        >>> stats = get_weight_statistics(weights)
        >>> print(f"Total parameters: {stats['total_parameters']:,}")
        >>> print(f"Total size: {stats['total_size_mb']:.2f} MB")
    """
    total_parameters = 0
    total_bytes = 0
    dtype_counts: dict[str, int] = {}
    shape_info: list[tuple[str, tuple[int, ...]]] = []

    for name, weight_array in weights.items():
        # Count parameters
        num_params = int(np.prod(weight_array.shape))
        total_parameters += num_params

        # Count bytes
        total_bytes += weight_array.nbytes

        # Track dtypes
        dtype_str = str(weight_array.dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

        # Track shapes
        shape_info.append((name, weight_array.shape))

    # Compute statistics
    stats = {
        "num_weights": len(weights),
        "total_parameters": total_parameters,
        "total_size_bytes": total_bytes,
        "total_size_mb": total_bytes / (1024 * 1024),
        "total_size_gb": total_bytes / (1024 * 1024 * 1024),
        "dtype_distribution": dtype_counts,
        "largest_weights": sorted(
            shape_info,
            key=lambda x: int(np.prod(x[1])),
            reverse=True,
        )[:10],
    }

    logger.info(
        "Weight statistics",
        num_weights=stats["num_weights"],
        total_parameters=f"{stats['total_parameters']:,}",
        total_size_gb=f"{stats['total_size_gb']:.2f} GB",
        dtypes=dtype_counts,
    )

    return stats
