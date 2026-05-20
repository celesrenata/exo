"""Property-based tests for the hash store module."""

import hashlib
import uuid
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from exo.docgen.hash_store import update_hash_store
from exo.docgen.models import HashStore

# Feature: local-model-documentation-generator, Property 12: Hash store reflects current filesystem state after run
# Validates: Requirements 7.3, 7.5


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_hash_store_reflects_current_filesystem_state_after_run(
    tmp_path: Path,
    data: st.DataObject,
) -> None:
    """For any completed generation run, the updated hash store SHALL contain an entry
    for every source file that exists on disk with its current SHA-256 hash, and SHALL
    contain no entries for files that no longer exist on disk.
    """
    # Use a unique subdirectory per example to avoid file collisions across hypothesis examples
    example_dir = tmp_path / uuid.uuid4().hex
    example_dir.mkdir(parents=True, exist_ok=True)

    # Draw how many files to create (at least 1 scanned file)
    num_files = data.draw(st.integers(min_value=1, max_value=10), label="num_files")
    num_deleted_entries = data.draw(st.integers(min_value=0, max_value=5), label="num_deleted_entries")

    # Create real files on disk with random content
    created_files: list[Path] = []
    for i in range(num_files):
        content = data.draw(st.binary(min_size=1, max_size=200), label=f"content_{i}")
        file_path = example_dir / f"file_{i}.py"
        file_path.write_bytes(content)
        created_files.append(file_path)

    existing_files = set(created_files)

    # Decide which files were scanned (non-empty subset of existing files)
    num_scanned = data.draw(
        st.integers(min_value=1, max_value=num_files),
        label="num_scanned",
    )
    scanned_files = created_files[:num_scanned]
    unscanned_files = created_files[num_scanned:]

    # Build old hash store with:
    # - Some entries for existing (unscanned) files with their "old" hashes
    # - Some entries for deleted files (paths that don't exist on disk)
    old_hashes: dict[str, str] = {}

    # Add entries for unscanned existing files (these should be preserved)
    for file_path in unscanned_files:
        old_hashes[str(file_path)] = hashlib.sha256(b"old_content_placeholder").hexdigest()

    # Add entries for deleted files (these should be removed)
    for i in range(num_deleted_entries):
        deleted_path = example_dir / f"deleted_{i}.py"
        old_hashes[str(deleted_path)] = hashlib.sha256(f"deleted_{i}".encode()).hexdigest()

    old_store = HashStore(hashes=old_hashes)

    # Call the function under test
    updated_store = update_hash_store(old_store, scanned_files, existing_files)

    # Property assertion 1: Every scanned file has an entry with its current SHA-256 hash
    for file_path in scanned_files:
        key = str(file_path)
        assert key in updated_store.hashes, f"Scanned file {key} missing from updated store"
        expected_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert updated_store.hashes[key] == expected_hash, (
            f"Hash mismatch for scanned file {key}"
        )

    # Property assertion 2: No entries exist for files not in existing_files
    for key in updated_store.hashes:
        assert Path(key) in existing_files, (
            f"Store contains entry for non-existing file: {key}"
        )

    # Property assertion 3: Entries for existing but unscanned files are preserved from old store
    for file_path in unscanned_files:
        key = str(file_path)
        if key in old_hashes:
            assert key in updated_store.hashes, (
                f"Existing unscanned file {key} was dropped from store"
            )
            assert updated_store.hashes[key] == old_hashes[key], (
                f"Hash for unscanned file {key} was modified"
            )
