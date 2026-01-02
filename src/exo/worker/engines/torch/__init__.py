from typing import Any, Protocol

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class Model(Protocol):
    """Protocol for PyTorch models used in EXO."""
    
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


class TokenizerWrapper:
    """Wrapper for HuggingFace tokenizers to match MLX interface."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        
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


class Detokenizer:
    """Simple detokenizer for streaming generation."""
    
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
                self._last_segment = full_text[len(self._last_segment):]
            else:
                self._last_segment = ""
        except Exception:
            self._last_segment = ""
            
    def finalize(self) -> None:
        pass
        
    @property
    def last_segment(self) -> str:
        return self._last_segment