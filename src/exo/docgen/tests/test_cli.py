"""Property-based tests for CLI target flag scope restriction."""

# Feature: local-model-documentation-generator, Property 14: Target flag restricts scanning scope

from __future__ import annotations

import shutil
import string
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.docgen.cli import _collect_python_files  # pyright: ignore[reportPrivateUsage]


@st.composite
def file_tree_with_target(draw: st.DrawFn) -> tuple[list[str], str]:
    """Generate a file tree description and a target subdirectory.

    Returns a tuple of (list of relative .py file paths, target relative directory).
    Ensures there are files both inside and outside the target subtree.
    """
    dir_name = st.text(min_size=1, max_size=8, alphabet=string.ascii_lowercase)
    file_name = st.text(min_size=1, max_size=8, alphabet=string.ascii_lowercase).map(
        lambda s: s + ".py"
    )

    target_dir = draw(dir_name)

    inside_files: list[str] = []
    n_inside = draw(st.integers(min_value=1, max_value=5))
    for _ in range(n_inside):
        depth = draw(st.integers(min_value=0, max_value=2))
        parts = [target_dir] + draw(st.lists(dir_name, min_size=depth, max_size=depth))
        name = draw(file_name)
        inside_files.append("/".join(parts + [name]))

    outside_files: list[str] = []
    n_outside = draw(st.integers(min_value=1, max_value=5))
    for _ in range(n_outside):
        other_dir = draw(dir_name.filter(lambda d: d != target_dir))
        depth = draw(st.integers(min_value=0, max_value=2))
        parts = [other_dir] + draw(st.lists(dir_name, min_size=depth, max_size=depth))
        name = draw(file_name)
        outside_files.append("/".join(parts + [name]))

    # Deduplicate to avoid filesystem collisions
    all_files = list(dict.fromkeys(inside_files + outside_files))
    return all_files, target_dir


# Validates: Requirements 8.3
@given(data=file_tree_with_target())
@settings(max_examples=100)
def test_target_flag_restricts_scanning_scope(
    data: tuple[list[str], str],
) -> None:
    """Property 14: Target flag restricts scanning scope.

    For any target path and file tree, the scanner SHALL process only files
    that are descendants of the target path, and no files outside that subtree.
    """
    file_paths_rel, target_rel = data

    root = Path(tempfile.mkdtemp())
    try:
        for rel_path in file_paths_rel:
            file_path = root / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("# placeholder\n")

        target_path = root / target_rel
        target_path.mkdir(parents=True, exist_ok=True)

        result = _collect_python_files(target_path)

        # All returned files are descendants of target
        for returned_file in result:
            assert returned_file.is_relative_to(target_path), (
                f"{returned_file} is not a descendant of {target_path}"
            )

        # No files outside target subtree are returned
        all_created = [root / rel for rel in file_paths_rel]
        outside_files = [f for f in all_created if not f.is_relative_to(target_path)]
        for outside_file in outside_files:
            assert outside_file not in result, (
                f"{outside_file} is outside target but was returned"
            )

        # All .py files under target ARE returned (completeness)
        expected_under_target = sorted(
            f for f in all_created if f.is_relative_to(target_path) and f.is_file()
        )
        assert result == expected_under_target, (
            f"Expected {expected_under_target}, got {result}"
        )
    finally:
        shutil.rmtree(root)
