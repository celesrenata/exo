"""
NixOS configuration smoke tests for distributed inference.

Since we cannot evaluate Nix expressions in the test environment, these tests
verify:
  1. The Nix module file exists and has the expected option definitions
  2. The GPU verification Python script is syntactically valid and importable
  3. Firewall port range configuration is present in the module
  4. No IPEX imports remain (delegates to existing test_no_ipex_imports.py)

Requirements: 9.1, 9.2, 9.3, 9.4, 11.1
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# Paths relative to the workspace root
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_NIX_DIR = _WORKSPACE_ROOT / "nix"
_DISTRIBUTED_MODULE = _NIX_DIR / "distributed-inference.nix"
_GPU_VERIFY_SCRIPT = _NIX_DIR / "verify-gpu-on-startup.py"


class TestNixModuleStructure:
    """Verify the NixOS distributed inference module has correct structure."""

    def test_distributed_module_exists(self) -> None:
        """The nix/distributed-inference.nix file must exist."""
        assert _DISTRIBUTED_MODULE.is_file(), (
            f"Expected NixOS module at {_DISTRIBUTED_MODULE.relative_to(_WORKSPACE_ROOT)}"
        )

    def test_module_defines_enable_option(self) -> None:
        """The module must define services.exo.distributed.enable option."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "services.exo.distributed" in content, (
            "Module must define options under services.exo.distributed"
        )
        assert "enable" in content, (
            "Module must define an 'enable' option"
        )

    def test_module_defines_master_addr_option(self) -> None:
        """The module must define a masterAddr option."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "masterAddr" in content, (
            "Module must define a 'masterAddr' option for MASTER_ADDR"
        )

    def test_module_defines_master_port_option(self) -> None:
        """The module must define a masterPort option with default 29500."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "masterPort" in content, (
            "Module must define a 'masterPort' option for MASTER_PORT"
        )
        assert "29500" in content, (
            "masterPort should default to 29500"
        )


class TestEnvironmentVariables:
    """Verify MASTER_ADDR and MASTER_PORT are set on the systemd service."""

    def test_master_addr_env_configured(self) -> None:
        """The module must set MASTER_ADDR environment variable on the exo service."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "MASTER_ADDR" in content, (
            "Module must configure MASTER_ADDR environment variable"
        )
        # Verify it's set in the systemd service environment block
        # The Nix module uses nested attribute syntax:
        #   systemd.services.exo = { ... environment = { MASTER_ADDR = ...; }; ... };
        assert "environment" in content and "systemd.services.exo" in content, (
            "MASTER_ADDR must be set via the systemd service environment block"
        )

    def test_master_port_env_configured(self) -> None:
        """The module must set MASTER_PORT environment variable on the exo service."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "MASTER_PORT" in content, (
            "Module must configure MASTER_PORT environment variable"
        )


class TestFirewallConfiguration:
    """Verify firewall rules include ephemeral port range for Gloo TCP."""

    def test_ephemeral_port_range_option_exists(self) -> None:
        """The module must define an ephemeralPortRange option."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "ephemeralPortRange" in content, (
            "Module must define ephemeralPortRange option"
        )

    def test_default_ephemeral_range(self) -> None:
        """Default ephemeral port range should be 49152-65535."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "49152" in content, "Default ephemeral range start should be 49152"
        assert "65535" in content, "Default ephemeral range end should be 65535"

    def test_firewall_port_range_configured(self) -> None:
        """The module must open the ephemeral port range in the firewall."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "networking.firewall.allowedTCPPortRanges" in content, (
            "Module must configure firewall TCP port ranges for Gloo"
        )

    def test_master_port_opened_in_firewall(self) -> None:
        """The master port must also be opened in the firewall."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "networking.firewall.allowedTCPPorts" in content, (
            "Module must open the master port in the firewall"
        )


class TestGpuVerificationScript:
    """Verify the GPU verification script is valid and functional."""

    def test_gpu_verify_script_exists(self) -> None:
        """The nix/verify-gpu-on-startup.py script must exist."""
        assert _GPU_VERIFY_SCRIPT.is_file(), (
            f"Expected GPU verification script at "
            f"{_GPU_VERIFY_SCRIPT.relative_to(_WORKSPACE_ROOT)}"
        )

    def test_gpu_verify_script_is_valid_python(self) -> None:
        """The GPU verification script must be syntactically valid Python."""
        source = _GPU_VERIFY_SCRIPT.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(_GPU_VERIFY_SCRIPT))
        except SyntaxError as exc:
            pytest.fail(f"GPU verification script has syntax error: {exc}")

    def test_gpu_verify_script_imports_detect_gpus(self) -> None:
        """The script must import detect_gpus from the gpu_detector module."""
        source = _GPU_VERIFY_SCRIPT.read_text(encoding="utf-8")
        assert "detect_gpus" in source, (
            "GPU verification script must use detect_gpus() from gpu_detector"
        )
        assert "gpu_detector" in source, (
            "GPU verification script must import from gpu_detector module"
        )

    def test_gpu_verify_script_logs_memory_architecture(self) -> None:
        """The script must log memory architecture (shared vs discrete)."""
        source = _GPU_VERIFY_SCRIPT.read_text(encoding="utf-8")
        assert "memory_architecture" in source, (
            "GPU verification script must log memory architecture"
        )

    def test_gpu_verify_script_has_main_entry(self) -> None:
        """The script must have a __main__ entry point."""
        source = _GPU_VERIFY_SCRIPT.read_text(encoding="utf-8")
        assert '__name__' in source and '__main__' in source, (
            "GPU verification script must have if __name__ == '__main__' block"
        )

    def test_gpu_verify_referenced_in_nix_module(self) -> None:
        """The NixOS module must reference the GPU verification script."""
        content = _DISTRIBUTED_MODULE.read_text(encoding="utf-8")
        assert "verify-gpu-on-startup" in content, (
            "NixOS module must reference the GPU verification script "
            "as an ExecStartPre command"
        )
        assert "ExecStartPre" in content, (
            "NixOS module must use ExecStartPre for GPU verification"
        )


class TestNoIpexImportsReference:
    """Cross-reference: verify no IPEX imports remain in the codebase.

    The comprehensive IPEX removal tests live in:
        src/exo/worker/engines/pytorch_xpu/tests/test_no_ipex_imports.py

    This test provides a lightweight check from the nix/tests location
    to confirm the same invariant holds.

    Requirements: 11.1
    """

    def test_no_ipex_imports_in_src(self) -> None:
        """Quick scan: no 'intel_extension_for_pytorch' in src/ Python files."""
        src_root = _WORKSPACE_ROOT / "src"
        # Construct search string dynamically to avoid self-match
        ipex_module = "intel_extension" + "_for_pytorch"
        # Exclude the dedicated IPEX removal test file (it references the
        # string in test names and search patterns, not as actual imports)
        excluded = {
            (src_root / "exo" / "worker" / "engines" / "pytorch_xpu"
             / "tests" / "test_no_ipex_imports.py").resolve(),
        }
        violations: list[str] = []

        for py_file in sorted(src_root.rglob("*.py")):
            if py_file.resolve() in excluded:
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
            for line_num, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if ipex_module in stripped:
                    rel = py_file.relative_to(_WORKSPACE_ROOT)
                    violations.append(f"  {rel}:{line_num}: {stripped}")

        if violations:
            pytest.fail(
                "Found IPEX imports in src/ (should be zero):\n"
                + "\n".join(violations)
            )

    def test_dedicated_no_ipex_test_exists(self) -> None:
        """The dedicated no-IPEX test file must exist."""
        test_file = (
            _WORKSPACE_ROOT
            / "src"
            / "exo"
            / "worker"
            / "engines"
            / "pytorch_xpu"
            / "tests"
            / "test_no_ipex_imports.py"
        )
        assert test_file.is_file(), (
            "Expected dedicated IPEX removal test at "
            "src/exo/worker/engines/pytorch_xpu/tests/test_no_ipex_imports.py"
        )
