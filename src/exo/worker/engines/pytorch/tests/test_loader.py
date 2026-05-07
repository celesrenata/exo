"""Unit tests for model/loader.py — HuggingFace safetensors model loading.

Tests use temporary directories with mock safetensors files to verify
loading logic without requiring actual model weights or GPU hardware.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exo.worker.engines.pytorch.model.loader import (
    LoadProgress,
    ModelLoadError,
    _check_memory_capacity,
    _discover_safetensors_files,
    _estimate_model_size,
    load_model_from_bound_instance,
    load_safetensors_to_device,
)


class TestModelLoadError:
    """Tests for the ModelLoadError exception."""

    def test_is_runtime_error(self) -> None:
        err = ModelLoadError("test error")
        assert isinstance(err, RuntimeError)

    def test_message_preserved(self) -> None:
        msg = "Model requires 16 GiB but only 8 GiB available"
        err = ModelLoadError(msg)
        assert str(err) == msg


class TestLoadProgress:
    """Tests for the LoadProgress dataclass."""

    def test_frozen_dataclass(self) -> None:
        progress = LoadProgress(
            loaded_bytes=1024,
            total_bytes=4096,
            percent=25.0,
            current_file="model-00001-of-00002.safetensors",
            files_loaded=0,
            total_files=2,
        )
        with pytest.raises(Exception):
            progress.percent = 50.0  # type: ignore[misc]

    def test_fields(self) -> None:
        progress = LoadProgress(
            loaded_bytes=2048,
            total_bytes=4096,
            percent=50.0,
            current_file="model-00002-of-00002.safetensors",
            files_loaded=1,
            total_files=2,
        )
        assert progress.loaded_bytes == 2048
        assert progress.total_bytes == 4096
        assert progress.percent == 50.0
        assert progress.current_file == "model-00002-of-00002.safetensors"
        assert progress.files_loaded == 1
        assert progress.total_files == 2


class TestDiscoverSafetensorsFiles:
    """Tests for _discover_safetensors_files."""

    def test_finds_safetensors_files(self, tmp_path: Path) -> None:
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 100)
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\x00" * 100)
        (tmp_path / "config.json").write_text("{}")

        files = _discover_safetensors_files(tmp_path)
        assert len(files) == 2
        assert all(f.suffix == ".safetensors" for f in files)

    def test_returns_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\x00" * 100)
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 100)

        files = _discover_safetensors_files(tmp_path)
        assert files[0].name == "model-00001-of-00002.safetensors"
        assert files[1].name == "model-00002-of-00002.safetensors"

    def test_raises_when_no_files(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}")

        with pytest.raises(ModelLoadError, match="No .safetensors files found"):
            _discover_safetensors_files(tmp_path)

    def test_raises_on_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="No .safetensors files found"):
            _discover_safetensors_files(tmp_path)


class TestEstimateModelSize:
    """Tests for _estimate_model_size."""

    def test_sums_file_sizes(self, tmp_path: Path) -> None:
        f1 = tmp_path / "model-00001.safetensors"
        f2 = tmp_path / "model-00002.safetensors"
        f1.write_bytes(b"\x00" * 1000)
        f2.write_bytes(b"\x00" * 2000)

        total = _estimate_model_size([f1, f2])
        assert total == 3000

    def test_single_file(self, tmp_path: Path) -> None:
        f1 = tmp_path / "model.safetensors"
        f1.write_bytes(b"\x00" * 5000)

        total = _estimate_model_size([f1])
        assert total == 5000


class TestCheckMemoryCapacity:
    """Tests for _check_memory_capacity."""

    @patch("exo.worker.engines.pytorch.model.loader._get_available_memory")
    def test_passes_when_sufficient(self, mock_mem: MagicMock) -> None:
        mock_mem.return_value = 16 * 1024**3  # 16 GiB available
        # Should not raise
        _check_memory_capacity(8 * 1024**3, "cuda", 0)

    @patch("exo.worker.engines.pytorch.model.loader._get_available_memory")
    def test_raises_when_insufficient(self, mock_mem: MagicMock) -> None:
        mock_mem.return_value = 4 * 1024**3  # 4 GiB available

        with pytest.raises(ModelLoadError, match="requires approximately"):
            _check_memory_capacity(8 * 1024**3, "cuda", 0)

    @patch("exo.worker.engines.pytorch.model.loader._get_available_memory")
    def test_error_includes_memory_details(self, mock_mem: MagicMock) -> None:
        mock_mem.return_value = 2 * 1024**3  # 2 GiB available

        with pytest.raises(ModelLoadError) as exc_info:
            _check_memory_capacity(10 * 1024**3, "xpu", 0)

        error_msg = str(exc_info.value)
        assert "xpu:0" in error_msg
        assert "GiB" in error_msg

    @patch("exo.worker.engines.pytorch.model.loader._get_available_memory")
    def test_proceeds_when_memory_unknown(self, mock_mem: MagicMock) -> None:
        mock_mem.return_value = 0  # Cannot determine memory
        # Should not raise — proceeds optimistically
        _check_memory_capacity(8 * 1024**3, "xpu", 0)


class TestLoadSafetensorsToDevice:
    """Tests for load_safetensors_to_device."""

    @patch("exo.worker.engines.pytorch.model.loader._check_memory_capacity")
    @patch("safetensors.torch.load_file")
    def test_yields_progress_reports(
        self, mock_load_file: MagicMock, mock_check_mem: MagicMock, tmp_path: Path
    ) -> None:
        import torch

        # Create fake safetensors files
        f1 = tmp_path / "model-00001-of-00002.safetensors"
        f2 = tmp_path / "model-00002-of-00002.safetensors"
        f1.write_bytes(b"\x00" * 1000)
        f2.write_bytes(b"\x00" * 1000)

        # Mock load_file to return tensors
        mock_load_file.side_effect = [
            {"layer.0.weight": torch.zeros(10, 10)},
            {"layer.1.weight": torch.zeros(10, 10)},
        ]

        gen = load_safetensors_to_device(tmp_path, "cuda", 0)
        results = list(gen)

        # Should have progress reports + final state dict
        progress_reports = [r for r in results if isinstance(r, LoadProgress)]
        state_dicts = [r for r in results if isinstance(r, dict)]

        # 2 files → 2 pre-load progress + 1 final progress = 3 progress reports
        assert len(progress_reports) == 3
        assert len(state_dicts) == 1

        # First progress: 0%
        assert progress_reports[0].percent == 0.0
        assert progress_reports[0].files_loaded == 0

        # Last progress: 100%
        assert progress_reports[-1].percent == 100.0
        assert progress_reports[-1].files_loaded == 2

        # State dict has all tensors
        assert "layer.0.weight" in state_dicts[0]
        assert "layer.1.weight" in state_dicts[0]

    @patch("exo.worker.engines.pytorch.model.loader._check_memory_capacity")
    @patch("safetensors.torch.load_file")
    def test_raises_on_load_failure(
        self, mock_load_file: MagicMock, mock_check_mem: MagicMock, tmp_path: Path
    ) -> None:
        f1 = tmp_path / "model.safetensors"
        f1.write_bytes(b"\x00" * 100)

        mock_load_file.side_effect = RuntimeError("corrupted file")

        gen = load_safetensors_to_device(tmp_path, "cuda", 0)

        # First yield is progress, then it should raise
        with pytest.raises(ModelLoadError, match="Failed to load"):
            list(gen)


class TestLoadModelFromBoundInstance:
    """Tests for load_model_from_bound_instance."""

    def test_raises_when_dir_missing(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "nonexistent"

        with pytest.raises(ModelLoadError, match="does not exist"):
            list(load_model_from_bound_instance(missing_dir, "cuda", 0))

    def test_raises_when_path_is_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "model.safetensors"
        file_path.write_bytes(b"\x00" * 100)

        with pytest.raises(ModelLoadError, match="not a directory"):
            list(load_model_from_bound_instance(file_path, "cuda", 0))

    @patch("exo.worker.engines.pytorch.model.loader._check_memory_capacity")
    @patch("safetensors.torch.load_file")
    def test_delegates_to_load_safetensors(
        self, mock_load_file: MagicMock, mock_check_mem: MagicMock, tmp_path: Path
    ) -> None:
        import torch

        f1 = tmp_path / "model.safetensors"
        f1.write_bytes(b"\x00" * 500)

        mock_load_file.return_value = {"weight": torch.zeros(5, 5)}

        results = list(load_model_from_bound_instance(tmp_path, "xpu", 0))

        progress_reports = [r for r in results if isinstance(r, LoadProgress)]
        state_dicts = [r for r in results if isinstance(r, dict)]

        assert len(progress_reports) >= 1
        assert len(state_dicts) == 1
        assert "weight" in state_dicts[0]
