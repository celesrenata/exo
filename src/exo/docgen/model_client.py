"""HTTP client for local model server chat completions."""

import os

from httpx import AsyncClient, HTTPError, HTTPStatusError, TimeoutException
from loguru import logger

from exo.docgen.models import ModelAlias

DEFAULT_MODEL_URL: str = "http://localhost:11434"

DEFAULT_MODEL_MAP: dict[ModelAlias, str] = {
    "deepseek": "deepseek-coder-v2",
    "condense": "qwen3-30b-a3b",
    "fast": "deepseek-coder-v2",
}


def _resolve_model_name(alias: ModelAlias) -> str:
    """Resolve a model alias to an actual model name using env var or defaults."""
    env_map = os.environ.get("DOCGEN_MODEL_MAP")
    if env_map is not None and env_map.strip():
        parsed: dict[str, str] = {}
        for entry in env_map.split(","):
            entry = entry.strip()
            if "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            parsed[key.strip()] = value.strip()
        if alias in parsed:
            return parsed[alias]
    return DEFAULT_MODEL_MAP[alias]


def _get_base_url() -> str:
    """Get the model server base URL from env var or default."""
    return os.environ.get("DOCGEN_MODEL_URL", DEFAULT_MODEL_URL)


async def chat_completion(
    alias: ModelAlias,
    system_prompt: str,
    user_prompt: str,
) -> str | None:
    """Send a chat completion request to the local model server.

    Returns the model's response content, or None on timeout/HTTP error.
    """
    model_name = _resolve_model_name(alias)
    base_url = _get_base_url()
    url = f"{base_url}/v1/chat/completions"
    payload: dict[str, object] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    async with AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()  # pyright: ignore[reportAny]
            choices: list[dict[str, object]] = data["choices"]  # pyright: ignore[reportAny]
            first_choice = choices[0]
            message = first_choice["message"]
            assert isinstance(message, dict)
            raw_content: object = message.get("content", "")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            assert isinstance(raw_content, str)
            return raw_content
        except TimeoutException:
            logger.warning("Model '{}' timed out for request", alias)
            return None
        except HTTPStatusError as exc:
            logger.warning(
                "Model '{}' returned HTTP error: {}", alias, exc.response.status_code
            )
            return None
        except HTTPError as exc:
            logger.warning("Model '{}' HTTP error: {}", alias, exc)
            return None
