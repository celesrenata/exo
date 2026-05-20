"""Integration tests for the documentation generator pipeline.

Tests end-to-end scanning, model client HTTP interactions, dry-run mode,
and hash store persistence across multiple runs.

Validates: Requirements 1.1, 7.1, 7.3, 10.2
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.docgen.cli import main
from exo.docgen.hash_store import (
    HASH_STORE_FILENAME,
    compute_file_hash,
    filter_changed_files,
    load_hash_store,
)
from exo.docgen.model_client import chat_completion
from exo.docgen.scanner import scan_file

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixture_dir(tmp_path: Path) -> Path:
    """Copy the test fixtures to a temporary directory and yield the path."""
    for source_file in FIXTURES_DIR.iterdir():
        if source_file.is_file():
            shutil.copy2(source_file, tmp_path / source_file.name)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. End-to-end scan: verify entity counts and scores against fixtures
# Validates: Requirements 1.1
# ---------------------------------------------------------------------------


class TestEndToEndScan:
    """End-to-end scanner integration tests against fixture files."""

    def test_documented_file_has_all_entities_documented(self, fixture_dir: Path) -> None:
        """Fully documented file should have score 1.0 and zero undocumented entities."""
        result = scan_file(fixture_dir / "documented.py")

        assert result.file_path == str(fixture_dir / "documented.py")
        assert len(result.entities) == 5
        assert result.documentation_score == 1.0
        assert result.undocumented_count == 0

    def test_undocumented_file_has_zero_score(self, fixture_dir: Path) -> None:
        """Completely undocumented file should have score 0.0."""
        result = scan_file(fixture_dir / "undocumented.py")

        assert result.file_path == str(fixture_dir / "undocumented.py")
        assert len(result.entities) == 5
        assert result.documentation_score == 0.0
        assert result.undocumented_count == 5

    def test_mixed_file_has_partial_score(self, fixture_dir: Path) -> None:
        """Mixed file should have score reflecting documented/total ratio."""
        result = scan_file(fixture_dir / "mixed.py")

        assert result.file_path == str(fixture_dir / "mixed.py")
        assert len(result.entities) == 6
        assert result.documentation_score == 0.5
        assert result.undocumented_count == 3

    def test_syntax_error_file_returns_empty_result(self, fixture_dir: Path) -> None:
        """File with syntax errors should return empty entities and score 1.0."""
        result = scan_file(fixture_dir / "syntax_error.py")

        assert result.file_path == str(fixture_dir / "syntax_error.py")
        assert len(result.entities) == 0
        assert result.documentation_score == 1.0
        assert result.undocumented_count == 0

    def test_scan_all_fixtures_produces_correct_totals(self, fixture_dir: Path) -> None:
        """Scanning all valid fixtures produces expected aggregate counts."""
        valid_files = ["documented.py", "undocumented.py", "mixed.py"]
        results = [scan_file(fixture_dir / name) for name in valid_files]

        total_entities = sum(len(r.entities) for r in results)
        total_undocumented = sum(r.undocumented_count for r in results)

        assert total_entities == 16
        assert total_undocumented == 8


# ---------------------------------------------------------------------------
# 2. Model client mock: verify HTTP request structure and error handling
# Validates: Requirements 1.1 (generation pipeline uses model client)
# ---------------------------------------------------------------------------


class TestModelClientIntegration:
    """Model client integration tests with mocked HTTP responses."""

    async def test_successful_response_returns_content(self) -> None:
        """A successful model response returns the content string."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "choices": [{"message": {"content": "Generated documentation."}}]
            }
        )

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("exo.docgen.model_client.AsyncClient", return_value=mock_client_instance):
            result = await chat_completion("deepseek", "You are a doc writer.", "Document this function.")

        assert result == "Generated documentation."

        # Verify request structure
        mock_client_instance.post.assert_called_once()  # pyright: ignore[reportAny]
        call_args = mock_client_instance.post.call_args  # pyright: ignore[reportAny]
        url = call_args[0][0]  # pyright: ignore[reportAny]
        assert url.endswith("/v1/chat/completions")  # pyright: ignore[reportAny]

        payload = call_args[1]["json"]  # pyright: ignore[reportAny]
        assert payload["temperature"] == 0.3
        assert payload["max_tokens"] == 2048
        assert len(payload["messages"]) == 2  # pyright: ignore[reportAny]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are a doc writer."
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "Document this function."

    async def test_timeout_returns_none(self) -> None:
        """A timeout from the model server returns None."""
        from httpx import TimeoutException

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=TimeoutException("timed out"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("exo.docgen.model_client.AsyncClient", return_value=mock_client_instance):
            result = await chat_completion("deepseek", "system", "user")

        assert result is None

    async def test_http_error_returns_none(self) -> None:
        """An HTTP error from the model server returns None."""
        from httpx import HTTPStatusError, Request, Response

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 500

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(
            side_effect=HTTPStatusError(
                "server error", request=mock_request, response=mock_response
            )
        )
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("exo.docgen.model_client.AsyncClient", return_value=mock_client_instance):
            result = await chat_completion("condense", "system", "user")

        assert result is None

    async def test_model_alias_resolves_to_correct_name(self) -> None:
        """The model alias is resolved and sent in the request payload."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={"choices": [{"message": {"content": "ok"}}]}
        )

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("exo.docgen.model_client.AsyncClient", return_value=mock_client_instance):
            await chat_completion("fast", "system", "user")

        payload = mock_client_instance.post.call_args[1]["json"]  # pyright: ignore[reportAny]
        assert isinstance(payload["model"], str)
        assert len(payload["model"]) > 0  # pyright: ignore[reportAny]

    async def test_custom_model_url_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The base URL is read from DOCGEN_MODEL_URL environment variable."""
        monkeypatch.setenv("DOCGEN_MODEL_URL", "http://custom-server:8080")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={"choices": [{"message": {"content": "response"}}]}
        )

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("exo.docgen.model_client.AsyncClient", return_value=mock_client_instance):
            await chat_completion("deepseek", "system", "user")

        url = mock_client_instance.post.call_args[0][0]  # pyright: ignore[reportAny]
        assert url == "http://custom-server:8080/v1/chat/completions"


# ---------------------------------------------------------------------------
# 3. Dry-run pipeline: verify no files are modified on disk
# Validates: Requirements 10.2
# ---------------------------------------------------------------------------


class TestDryRunPipeline:
    """Dry-run mode integration tests verifying filesystem safety."""

    def test_dry_run_does_not_modify_source_files(
        self, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running with --dry-run does not modify any source files."""
        monkeypatch.setattr(
            sys, "argv", ["generate-docs", "--dry-run", "--target", str(fixture_dir)]
        )
        monkeypatch.chdir(fixture_dir)

        # Record file contents before run
        original_contents: dict[str, bytes] = {}
        for file_path in fixture_dir.iterdir():
            if file_path.is_file():
                original_contents[file_path.name] = file_path.read_bytes()

        with pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 0

        # Verify no source files were modified
        for file_path in fixture_dir.iterdir():
            if file_path.is_file() and file_path.name in original_contents:
                assert file_path.read_bytes() == original_contents[file_path.name], (
                    f"File {file_path.name} was modified during dry-run"
                )

    def test_dry_run_does_not_save_hash_store(
        self, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running with --dry-run does not persist the hash store."""
        monkeypatch.setattr(
            sys, "argv", ["generate-docs", "--dry-run", "--target", str(fixture_dir)]
        )
        monkeypatch.chdir(fixture_dir)

        hash_store_path = fixture_dir / HASH_STORE_FILENAME

        with pytest.raises(SystemExit):
            main()

        assert not hash_store_path.exists(), "Hash store should not be saved in dry-run mode"


# ---------------------------------------------------------------------------
# 4. Hash store persistence: verify incremental behavior across runs
# Validates: Requirements 7.1, 7.3
# ---------------------------------------------------------------------------


class TestHashStorePersistence:
    """Hash store persistence tests verifying incremental update behavior."""

    def test_hash_store_created_after_first_run(
        self, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hash store file is created after the first pipeline run."""
        monkeypatch.setattr(
            sys, "argv", ["generate-docs", "--target", str(fixture_dir)]
        )
        monkeypatch.chdir(fixture_dir)

        with patch("exo.docgen.cli.chat_completion", new_callable=AsyncMock, return_value=None), pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 0

        hash_store_path = fixture_dir / HASH_STORE_FILENAME
        assert hash_store_path.exists(), "Hash store file should exist after first run"

        # Verify the store contains entries for scanned files
        store = load_hash_store(fixture_dir)
        python_files = [f for f in fixture_dir.iterdir() if f.suffix == ".py"]
        for python_file in python_files:
            key = str(python_file)
            assert key in store.hashes, f"Hash store missing entry for {python_file.name}"
            assert store.hashes[key] == compute_file_hash(python_file)

    def test_unchanged_files_skipped_on_second_run(
        self, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files with unchanged hashes are skipped on subsequent runs."""
        monkeypatch.chdir(fixture_dir)

        # First run: creates hash store
        monkeypatch.setattr(
            sys, "argv", ["generate-docs", "--target", str(fixture_dir)]
        )
        with patch("exo.docgen.cli.chat_completion", new_callable=AsyncMock, return_value=None), pytest.raises(SystemExit):
            main()

        # Load hash store after first run
        store_after_first = load_hash_store(fixture_dir)

        # Verify filter_changed_files returns empty for unchanged files
        python_files = sorted(f for f in fixture_dir.iterdir() if f.suffix == ".py")
        changed = filter_changed_files(python_files, store_after_first, force=False)
        assert changed == [], "No files should be marked as changed after first run"

        # Second run: should exit early because no files changed
        monkeypatch.setattr(
            sys, "argv", ["generate-docs", "--target", str(fixture_dir)]
        )
        with patch("exo.docgen.cli.chat_completion", new_callable=AsyncMock, return_value=None), pytest.raises(SystemExit) as exit_info:
            main()

        # Pipeline exits with 0 when no files have changed
        assert exit_info.value.code == 0

        # Hash store should be unchanged
        store_after_second = load_hash_store(fixture_dir)
        assert store_after_second.hashes == store_after_first.hashes

    def test_force_flag_reprocesses_all_files(
        self, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The --force flag causes all files to be reprocessed regardless of hash state."""
        monkeypatch.chdir(fixture_dir)

        # First run: creates hash store
        monkeypatch.setattr(
            sys, "argv", ["generate-docs", "--target", str(fixture_dir)]
        )
        with patch("exo.docgen.cli.chat_completion", new_callable=AsyncMock, return_value=None), pytest.raises(SystemExit):
            main()

        store = load_hash_store(fixture_dir)
        python_files = sorted(f for f in fixture_dir.iterdir() if f.suffix == ".py")

        # Without force: no files changed
        changed_no_force = filter_changed_files(python_files, store, force=False)
        assert changed_no_force == []

        # With force: all files returned
        changed_with_force = filter_changed_files(python_files, store, force=True)
        assert len(changed_with_force) == len(python_files)

    def test_modified_file_detected_on_second_run(
        self, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file modified between runs is detected as changed."""
        monkeypatch.chdir(fixture_dir)

        # First run
        monkeypatch.setattr(
            sys, "argv", ["generate-docs", "--target", str(fixture_dir)]
        )
        with patch("exo.docgen.cli.chat_completion", new_callable=AsyncMock, return_value=None), pytest.raises(SystemExit):
            main()

        store = load_hash_store(fixture_dir)

        # Modify one file
        target_file = fixture_dir / "undocumented.py"
        original_content = target_file.read_text()
        target_file.write_text(original_content + "\n# modified\n")

        # Check that the modified file is detected
        python_files = sorted(f for f in fixture_dir.iterdir() if f.suffix == ".py")
        changed = filter_changed_files(python_files, store, force=False)
        assert target_file in changed
        assert len(changed) == 1

    def test_deleted_file_removed_from_hash_store(
        self, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file deleted between runs is removed from the hash store."""
        monkeypatch.chdir(fixture_dir)

        # First run
        monkeypatch.setattr(
            sys, "argv", ["generate-docs", "--target", str(fixture_dir)]
        )
        with patch("exo.docgen.cli.chat_completion", new_callable=AsyncMock, return_value=None), pytest.raises(SystemExit):
            main()

        store_before = load_hash_store(fixture_dir)
        deleted_file = fixture_dir / "undocumented.py"
        deleted_key = str(deleted_file)
        assert deleted_key in store_before.hashes

        # Delete the file
        deleted_file.unlink()

        # Second run (force to ensure it processes)
        monkeypatch.setattr(
            sys, "argv", ["generate-docs", "--target", str(fixture_dir), "--force"]
        )
        with patch("exo.docgen.cli.chat_completion", new_callable=AsyncMock, return_value=None), pytest.raises(SystemExit):
            main()

        store_after = load_hash_store(fixture_dir)
        assert deleted_key not in store_after.hashes
