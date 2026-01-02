"""
Minimal stub for openai_harmony to allow testing CPU engine.
This is a temporary workaround until the full dependency is available.
"""

from enum import Enum
from typing import Any


class HarmonyEncodingName(Enum):
    HARMONY_GPT_OSS = "harmony_gpt_oss"


class Role(Enum):
    ASSISTANT = "assistant"
    USER = "user"


class StreamableParser:
    """Minimal stub for StreamableParser."""

    def __init__(self, encoding: Any, role: Role = Role.ASSISTANT):
        self.encoding = encoding
        self.role = role
        self._buffer = ""

    def parse(self, text: str) -> dict[str, Any]:
        """Minimal parsing - just return the text as content."""
        self._buffer += text
        return {
            "role": self.role.value,
            "content": text,
            "thinking": None,
        }

    def finalize(self) -> dict[str, Any]:
        """Finalize parsing."""
        return {
            "role": self.role.value,
            "content": self._buffer,
            "thinking": None,
        }


def load_harmony_encoding(name: HarmonyEncodingName) -> dict[str, Any]:
    """Minimal stub for loading harmony encoding."""
    return {
        "name": name.value,
        "version": "stub",
    }
