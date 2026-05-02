"""
Test: Verify the ipex → xpu rename is complete across the entire codebase.

This test complements test_no_ipex_imports.py (which lives in the pytorch_xpu
engine tests) by checking the broader codebase — Python source, Nix configs,
instance types, runner variables, and bootstrap env vars.

Requirements: 6.4, 8.4, 11.3, 11.4
"""

from pathlib import Path

import pytest


# Workspace root (two levels above src/)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]  # src/exo/shared/tests -> workspace root
_SRC_ROOT = _WORKSPACE_ROOT / "src"
_FLAKE_NIX = _WORKSPACE_ROOT / "flake.nix"

# Files excluded from stale-reference scanning (they legitimately mention the old names)
_EXCLUDED_FILENAMES = {
    Path(__file__).resolve().name,  # this test itself
    "test_no_ipex" + "_imports.py",  # the engine-level stale-reference test
}

# Search strings constructed dynamically to avoid self-matching
_STALE_MODULE_PATH = "pytorch" + "_ipex"
_STALE_CLASS_PREFIX = "PyTorch" + "IPEX"
_STALE_NIX_MODULE = "pytorch" + "_ipex"
_STALE_NIX_ENV_VAR = "IPEX_TILE" + "_AS_DEVICE"


def _python_files_excluding_self(root: Path) -> list[Path]:
    """Collect all .py files under root, excluding known stale-reference test files."""
    return sorted(
        p
        for p in root.rglob("*.py")
        if p.resolve().name not in _EXCLUDED_FILENAMES
    )


def _scan_for_pattern(
    files: list[Path], pattern: str, *, skip_comments: bool = True
) -> list[tuple[Path, int, str]]:
    """Scan files for a pattern, returning (path, line_num, line) tuples."""
    violations: list[tuple[Path, int, str]] = []
    for py_file in files:
        try:
            content = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        for line_num, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if skip_comments and stripped.startswith("#"):
                continue
            if pattern in stripped:
                violations.append((py_file, line_num, stripped))
    return violations


class TestXpuRenameCompleteness:
    """Verify the ipex → xpu rename is complete across the codebase."""

    # ── Python source: no stale module path ──────────────────────────

    def test_no_stale_pytorch_ipex_in_src(self) -> None:
        """Assert zero occurrences of 'pytorch_ipex' in src/ Python files.

        Requirements: 6.4, 11.4
        """
        files = _python_files_excluding_self(_SRC_ROOT)
        violations = _scan_for_pattern(files, _STALE_MODULE_PATH)

        if violations:
            lines = [f"Found stale '{_STALE_MODULE_PATH}' references:"]
            for path, num, line in violations:
                rel = path.relative_to(_SRC_ROOT)
                lines.append(f"  {rel}:{num}: {line}")
            pytest.fail("\n".join(lines))

    # ── Python source: no stale class names ──────────────────────────

    def test_no_stale_pytorch_ipex_class_names_in_src(self) -> None:
        """Assert zero occurrences of 'PyTorchIPEX' in src/ Python files.

        Requirements: 8.4, 11.3
        """
        files = _python_files_excluding_self(_SRC_ROOT)
        violations = _scan_for_pattern(files, _STALE_CLASS_PREFIX)

        if violations:
            lines = [f"Found stale '{_STALE_CLASS_PREFIX}' references:"]
            for path, num, line in violations:
                rel = path.relative_to(_SRC_ROOT)
                lines.append(f"  {rel}:{num}: {line}")
            pytest.fail("\n".join(lines))

    # ── Instance types are importable ────────────────────────────────

    def test_instance_meta_pytorch_xpu_ring_exists(self) -> None:
        """Verify InstanceMeta.PyTorchXPURing exists and is importable."""
        from exo.shared.types.worker.instances import InstanceMeta

        assert hasattr(InstanceMeta, "PyTorchXPURing"), (
            "InstanceMeta.PyTorchXPURing not found — rename may be incomplete"
        )
        assert InstanceMeta.PyTorchXPURing.value == "PyTorchXPURing"

    def test_pytorch_xpu_ring_instance_importable(self) -> None:
        """Verify PyTorchXPURingInstance is importable from instances module."""
        from exo.shared.types.worker.instances import PyTorchXPURingInstance

        assert PyTorchXPURingInstance is not None

    # ── Runner uses is_pytorch_xpu variable ──────────────────────────

    def test_runner_uses_is_pytorch_xpu(self) -> None:
        """Verify runner.py uses the 'is_pytorch_xpu' variable name.

        Requirements: 8.4
        """
        runner_path = _SRC_ROOT / "exo" / "worker" / "runner" / "runner.py"
        assert runner_path.exists(), f"runner.py not found at {runner_path}"

        content = runner_path.read_text(encoding="utf-8")
        target = "is_pytorch" + "_xpu"
        assert target in content, (
            f"Expected '{target}' variable in runner.py but it was not found"
        )

    # ── Bootstrap sets correct env var ───────────────────────────────

    def test_bootstrap_sets_xpu_enabled_env_var(self) -> None:
        """Verify bootstrap.py sets EXO_PYTORCH_XPU_ENABLED (not the old IPEX one).

        Requirements: 8.4
        """
        bootstrap_path = _SRC_ROOT / "exo" / "worker" / "runner" / "bootstrap.py"
        assert bootstrap_path.exists(), f"bootstrap.py not found at {bootstrap_path}"

        content = bootstrap_path.read_text(encoding="utf-8")

        expected = "EXO_PYTORCH" + "_XPU_ENABLED"
        assert expected in content, (
            f"Expected '{expected}' in bootstrap.py but it was not found"
        )

        stale = "EXO_PYTORCH" + "_IPEX_ENABLED"
        for line_num, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert stale not in stripped, (
                f"Found stale '{stale}' in bootstrap.py:{line_num}: {stripped}"
            )

    # ── Nix config: no stale references ──────────────────────────────

    def test_flake_nix_no_stale_pytorch_ipex(self) -> None:
        """Assert zero occurrences of 'pytorch_ipex' in flake.nix.

        Requirements: 11.4
        """
        assert _FLAKE_NIX.exists(), f"flake.nix not found at {_FLAKE_NIX}"

        content = _FLAKE_NIX.read_text(encoding="utf-8")
        violations: list[tuple[int, str]] = []

        for line_num, line in enumerate(content.splitlines(), start=1):
            if _STALE_NIX_MODULE in line:
                violations.append((line_num, line.strip()))

        if violations:
            lines = [f"Found stale '{_STALE_NIX_MODULE}' in flake.nix:"]
            for num, line in violations:
                lines.append(f"  flake.nix:{num}: {line}")
            pytest.fail("\n".join(lines))

    def test_flake_nix_no_ipex_tile_as_device(self) -> None:
        """Assert zero occurrences of 'IPEX_TILE_AS_DEVICE' in flake.nix.

        Requirements: 11.4
        """
        assert _FLAKE_NIX.exists(), f"flake.nix not found at {_FLAKE_NIX}"

        content = _FLAKE_NIX.read_text(encoding="utf-8")
        violations: list[tuple[int, str]] = []

        for line_num, line in enumerate(content.splitlines(), start=1):
            if _STALE_NIX_ENV_VAR in line:
                violations.append((line_num, line.strip()))

        if violations:
            lines = [f"Found stale '{_STALE_NIX_ENV_VAR}' in flake.nix:"]
            for num, line in violations:
                lines.append(f"  flake.nix:{num}: {line}")
            pytest.fail("\n".join(lines))
