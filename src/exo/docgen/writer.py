"""File writer module for the documentation generator.

Provides `write_planned` which processes a list of planned file writes.
Supports both normal execution (actual filesystem writes) and dry-run mode
(no filesystem modifications, only diagnostics printed to stdout).
"""

import difflib
import sys

from loguru import logger

from exo.docgen.models import PlannedWrite, WriteResult


def write_planned(writes: list[PlannedWrite], dry_run: bool) -> list[WriteResult]:
    """Execute or simulate a batch of planned file writes.

    Args:
        writes: List of PlannedWrite instances describing intended file operations.
        dry_run: If True, no files are created or modified. Diagnostics are
            printed to stdout (diffs for updates, full content for creates).

    Returns:
        List of WriteResult instances, one per input write, indicating the
        action performed (or simulated) and whether it succeeded.
    """
    results: list[WriteResult] = []

    for write in writes:
        result = _handle_dry_run(write) if dry_run else _handle_write(write)
        results.append(result)

    _print_summary(writes)

    return results


def _handle_dry_run(write: PlannedWrite) -> WriteResult:
    """Handle a single planned write in dry-run mode without touching the filesystem."""
    if write.action == "update":
        existing = write.existing_content or ""
        diff_lines = difflib.unified_diff(
            existing.splitlines(keepends=True),
            write.content.splitlines(keepends=True),
            fromfile=str(write.destination),
            tofile=str(write.destination),
        )
        sys.stdout.writelines(diff_lines)
    else:
        print(f"[create] {write.destination}")
        print(write.content)

    return WriteResult(destination=write.destination, action=write.action, success=True)


def _handle_write(write: PlannedWrite) -> WriteResult:
    """Handle a single planned write in normal mode, creating directories and writing content."""
    try:
        write.destination.parent.mkdir(parents=True, exist_ok=True)
        write.destination.write_text(write.content, encoding="utf-8")
        return WriteResult(destination=write.destination, action=write.action, success=True)
    except Exception as exception:
        logger.warning(f"Failed to write {write.destination}: {exception}")
        return WriteResult(destination=write.destination, action=write.action, success=False)


def _print_summary(writes: list[PlannedWrite]) -> None:
    """Print a summary list of all planned operations."""
    print("\nPlanned operations:")
    for write in writes:
        print(f"  [{write.action}] {write.destination}")
