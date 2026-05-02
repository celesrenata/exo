"""
Test: Verify no IPEX imports or stale pytorch_ipex references remain in the codebase.

This test ensures that all references to the deprecated Intel Extension for
PyTorch have been removed from the src/ directory, that native torch.xpu
is used instead, and that the ipex → xpu rename is complete.

Requirements: 6.4, 8.4, 11.1, 11.3, 11.4, 11.5
"""

from pathlib import Path

import pytest

# Root of the src/ directory
SRC_ROOT = Path(__file__).resolve().parents[5]  # src/exo/worker/engines/pytorch_xpu/tests -> src/

# This test file itself is excluded from scanning to avoid self-referential matches
_SELF = Path(__file__).resolve()

# The search strings are constructed dynamically to avoid matching this file
_IPEX_MODULE = "intel_extension" + "_for_pytorch"
_IMPORT_IPEX = "import" + " ipex"
_AS_IPEX = "as" + " ipex"
_IPEX_OPTIMIZE = "ipex" + ".optimize"

# Stale reference patterns from the ipex → xpu rename
_STALE_MODULE_PATH = "pytorch" + "_ipex"
_STALE_CLASS_PREFIX = "PyTorch" + "IPEX"


def _collect_python_files(root: Path) -> list[Path]:
    """Collect all .py files under the given root directory, excluding this test."""
    return sorted(p for p in root.rglob("*.py") if p.resolve() != _SELF)


class TestNoIpexImports:
    """Verify that IPEX has been fully removed from the src/ directory."""

    def test_no_intel_extension_for_pytorch_imports(self) -> None:
        """Assert zero occurrences of the IPEX module name in src/."""
        violations: list[tuple[Path, int, str]] = []

        for py_file in _collect_python_files(SRC_ROOT):
            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            for line_num, line in enumerate(content.splitlines(), start=1):
                if _IPEX_MODULE in line:
                    violations.append((py_file, line_num, line.strip()))

        if violations:
            msg_lines = [f"Found '{_IPEX_MODULE}' in the following files:"]
            for path, line_num, line in violations:
                rel = path.relative_to(SRC_ROOT)
                msg_lines.append(f"  {rel}:{line_num}: {line}")
            pytest.fail("\n".join(msg_lines))

    def test_no_import_ipex(self) -> None:
        """Assert zero occurrences of IPEX import statements in src/."""
        violations: list[tuple[Path, int, str]] = []

        for py_file in _collect_python_files(SRC_ROOT):
            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            for line_num, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                # Skip comments
                if stripped.startswith("#"):
                    continue
                if _IMPORT_IPEX in stripped or _AS_IPEX in stripped:
                    violations.append((py_file, line_num, stripped))

        if violations:
            msg_lines = [f"Found '{_IMPORT_IPEX}' in the following files:"]
            for path, line_num, line in violations:
                rel = path.relative_to(SRC_ROOT)
                msg_lines.append(f"  {rel}:{line_num}: {line}")
            pytest.fail("\n".join(msg_lines))

    def test_torch_xpu_is_available_used(self) -> None:
        """Verify that torch.xpu.is_available() is used in the codebase."""
        found = False

        for py_file in _collect_python_files(SRC_ROOT):
            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            if "xpu.is_available()" in content:
                found = True
                break

        assert found, (
            "Expected to find 'xpu.is_available()' in at least one file under src/, "
            "but it was not found. Native PyTorch XPU detection should be used."
        )

    def test_no_ipex_optimize_calls(self) -> None:
        """Assert zero occurrences of ipex.optimize in src/."""
        violations: list[tuple[Path, int, str]] = []

        for py_file in _collect_python_files(SRC_ROOT):
            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            for line_num, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if _IPEX_OPTIMIZE in stripped:
                    violations.append((py_file, line_num, stripped))

        if violations:
            msg_lines = [f"Found '{_IPEX_OPTIMIZE}' calls in the following files:"]
            for path, line_num, line in violations:
                rel = path.relative_to(SRC_ROOT)
                msg_lines.append(f"  {rel}:{line_num}: {line}")
            pytest.fail("\n".join(msg_lines))

    def test_no_stale_pytorch_ipex_module_path(self) -> None:
        """Assert zero occurrences of the old 'pytorch_ipex' module path in src/.

        After the ipex → xpu rename, no Python file should reference the old
        module path. This catches incomplete renames.
        Requirements: 6.4, 11.4
        """
        violations: list[tuple[Path, int, str]] = []

        for py_file in _collect_python_files(SRC_ROOT):
            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            for line_num, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                # Skip comments — they may legitimately mention the old name
                # in historical context
                if stripped.startswith("#"):
                    continue
                if _STALE_MODULE_PATH in stripped:
                    violations.append((py_file, line_num, stripped))

        if violations:
            msg_lines = [
                f"Found stale '{_STALE_MODULE_PATH}' references in the following files "
                f"(should be 'pytorch_xpu' after rename):"
            ]
            for path, line_num, line in violations:
                rel = path.relative_to(SRC_ROOT)
                msg_lines.append(f"  {rel}:{line_num}: {line}")
            pytest.fail("\n".join(msg_lines))

    def test_no_stale_pytorch_ipex_class_names(self) -> None:
        """Assert zero occurrences of old 'PyTorchIPEX' class/enum names in src/.

        After the ipex → xpu rename, no Python file should reference the old
        class names like PyTorchIPEXRing or PyTorchIPEXRingInstance.
        Requirements: 8.4, 11.3
        """
        violations: list[tuple[Path, int, str]] = []

        for py_file in _collect_python_files(SRC_ROOT):
            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            for line_num, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if _STALE_CLASS_PREFIX in stripped:
                    violations.append((py_file, line_num, stripped))

        if violations:
            msg_lines = [
                f"Found stale '{_STALE_CLASS_PREFIX}' references in the following files "
                f"(should be 'PyTorchXPU' after rename):"
            ]
            for path, line_num, line in violations:
                rel = path.relative_to(SRC_ROOT)
                msg_lines.append(f"  {rel}:{line_num}: {line}")
            pytest.fail("\n".join(msg_lines))
