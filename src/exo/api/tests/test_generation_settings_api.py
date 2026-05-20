# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.generation_settings import register_generation_settings_routes
from exo.api.main import API
from exo.shared.types.state import State


def _make_api() -> Any:
    """Create a minimal API instance with generation settings routes and error handler."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api.state = State()
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    register_generation_settings_routes(api)
    return api


def test_get_returns_default_settings() -> None:
    """GET /api/generation/settings returns default settings."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/api/generation/settings")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data == {
        "thinkingMode": True,
        "thinkingTokenBudget": None,
        "outputTokenBudget": None,
    }


def test_patch_partial_update_thinking_mode() -> None:
    """PATCH with thinking_mode=False returns updated settings with other fields unchanged."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.patch(
        "/api/generation/settings",
        json={"thinking_mode": False},
    )
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data == {
        "thinkingMode": False,
        "thinkingTokenBudget": None,
        "outputTokenBudget": None,
    }
    api._send.assert_called_once()  # pyright: ignore[reportPrivateUsage]


def test_patch_invalid_thinking_token_budget_returns_422() -> None:
    """PATCH with thinking_token_budget=0 returns 422 (must be >= 1)."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.patch(
        "/api/generation/settings",
        json={"thinking_token_budget": 0},
    )
    assert response.status_code == 422


def test_patch_thinking_mode_false_accepts_any_budget() -> None:
    """PATCH with thinking_mode=false is accepted regardless of thinking_token_budget value."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.patch(
        "/api/generation/settings",
        json={"thinking_mode": False, "thinking_token_budget": 1024},
    )
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data == {
        "thinkingMode": False,
        "thinkingTokenBudget": 1024,
        "outputTokenBudget": None,
    }
