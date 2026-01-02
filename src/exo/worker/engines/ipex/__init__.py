from typing import Any, Protocol

import torch
from transformers import PreTrainedTokenizer


# IPEX-specific error classes for comprehensive error handling
class IPEXEngineError(Exception):
    """Base exception for IPEX engine errors."""

    def __init__(
        self, message: str, device_id: int | None = None, error_code: str | None = None
    ):
        self.device_id = device_id
        self.error_code = error_code
        super().__init__(message)


class IPEXDriverError(IPEXEngineError):
    """Intel GPU driver not available or incompatible."""

    def __init__(
        self,
        message: str = "Intel GPU driver not available or incompatible",
        driver_version: str | None = None,
        required_version: str | None = None,
    ):
        self.driver_version = driver_version
        self.required_version = required_version
        super().__init__(message, error_code="DRIVER_ERROR")


class IPEXMemoryError(IPEXEngineError):
    """Intel GPU memory allocation failed."""

    def __init__(
        self,
        message: str = "Intel GPU memory allocation failed",
        device_id: int | None = None,
        requested_memory: int | None = None,
        available_memory: int | None = None,
    ):
        self.requested_memory = requested_memory
        self.available_memory = available_memory
        super().__init__(message, device_id=device_id, error_code="MEMORY_ERROR")


class IPEXInitializationError(IPEXEngineError):
    """IPEX engine initialization failed."""

    def __init__(
        self,
        message: str = "IPEX engine initialization failed",
        component: str | None = None,
    ):
        self.component = component
        super().__init__(message, error_code="INIT_ERROR")


class IPEXModelLoadError(IPEXEngineError):
    """IPEX model loading failed."""

    def __init__(
        self,
        message: str = "IPEX model loading failed",
        model_path: str | None = None,
        device_id: int | None = None,
    ):
        self.model_path = model_path
        super().__init__(message, device_id=device_id, error_code="MODEL_LOAD_ERROR")


class IPEXInferenceError(IPEXEngineError):
    """IPEX inference execution failed."""

    def __init__(
        self,
        message: str = "IPEX inference execution failed",
        device_id: int | None = None,
        step: str | None = None,
    ):
        self.step = step
        super().__init__(message, device_id=device_id, error_code="INFERENCE_ERROR")


class IPEXDistributedError(IPEXEngineError):
    """IPEX distributed inference failed."""

    def __init__(
        self,
        message: str = "IPEX distributed inference failed",
        rank: int | None = None,
        world_size: int | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        super().__init__(message, error_code="DISTRIBUTED_ERROR")


class IPEXModel(Protocol):
    """Protocol for IPEX-optimized models used in EXO."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Any: ...

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor: ...


class IPEXTokenizerWrapper:
    """Wrapper for HuggingFace tokenizers to match MLX interface with IPEX optimizations."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token_ids = (
            [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages_dicts: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class IPEXDetokenizer:
    """Simple detokenizer for streaming generation with IPEX optimizations."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.tokens: list[int] = []
        self._last_segment = ""

    def reset(self) -> None:
        self.tokens = []
        self._last_segment = ""

    def add_token(self, token: int) -> None:
        self.tokens.append(token)
        # Decode the last few tokens to get the new text segment
        try:
            full_text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
            if len(full_text) > len(self._last_segment):
                self._last_segment = full_text[len(self._last_segment) :]
            else:
                self._last_segment = ""
        except Exception:
            self._last_segment = ""

    def finalize(self) -> None:
        pass

    @property
    def last_segment(self) -> str:
        return self._last_segment
