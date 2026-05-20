"""Incremental hash store for documentation generator change detection."""

import hashlib
import json
from pathlib import Path

from loguru import logger

from exo.docgen.models import HashStore

HASH_STORE_FILENAME = ".docgen_hashes.json"


def load_hash_store(path: Path) -> HashStore:
    """Read the hash store from the given directory.

    Args:
        path: Directory containing the `.docgen_hashes.json` file.

    Returns:
        The loaded HashStore, or an empty HashStore if the file is missing or corrupt.
    """
    hash_file = path / HASH_STORE_FILENAME
    try:
        content = hash_file.read_text(encoding="utf-8")
        data = json.loads(content)  # pyright: ignore[reportAny]
        return HashStore.model_validate(data)
    except FileNotFoundError:
        logger.warning(f"Hash store not found at {hash_file}, treating all files as changed")
        return HashStore()
    except (json.JSONDecodeError, ValueError) as error:
        logger.warning(f"Corrupt hash store at {hash_file}: {error}")
        return HashStore()


def save_hash_store(store: HashStore, path: Path) -> None:
    """Write the HashStore to the given directory as JSON.

    Args:
        store: The HashStore to persist.
        path: Directory where `.docgen_hashes.json` will be written.
    """
    hash_file = path / HASH_STORE_FILENAME
    serialized = json.dumps(store.model_dump(by_alias=True), indent=2)
    hash_file.write_text(serialized, encoding="utf-8")


def compute_file_hash(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents.

    Args:
        path: Path to the file to hash.

    Returns:
        The SHA-256 hex digest string.
    """
    file_bytes = path.read_bytes()
    return hashlib.sha256(file_bytes).hexdigest()


def filter_changed_files(files: list[Path], store: HashStore, force: bool) -> list[Path]:
    """Filter files to only those that have changed since the last run.

    Args:
        files: All candidate source files.
        store: The hash store from the previous run.
        force: If True, treat all files as changed regardless of hash state.

    Returns:
        Files whose content hash differs from the store or that have no store entry.
    """
    if force:
        return list(files)

    changed: list[Path] = []
    for file_path in files:
        key = str(file_path)
        stored_hash = store.hashes.get(key)
        if stored_hash is None:
            changed.append(file_path)
        else:
            current_hash = compute_file_hash(file_path)
            if current_hash != stored_hash:
                changed.append(file_path)
    return changed


def update_hash_store(
    store: HashStore,
    scanned_files: list[Path],
    existing_files: set[Path],
) -> HashStore:
    """Create an updated HashStore reflecting the current filesystem state.

    Args:
        store: The previous hash store.
        scanned_files: Files that were scanned in this run (hashes will be recomputed).
        existing_files: All files that currently exist on disk.

    Returns:
        A new HashStore with updated hashes for scanned files, preserved entries
        for existing but unscanned files, and removed entries for deleted files.
    """
    new_hashes: dict[str, str] = {}

    # Add/update entries for all scanned files with their current hashes
    for file_path in scanned_files:
        key = str(file_path)
        new_hashes[key] = compute_file_hash(file_path)

    # Preserve entries from old store for files that exist but were not scanned
    for key, stored_hash in store.hashes.items():
        file_path = Path(key)
        if file_path in existing_files and key not in new_hashes:
            new_hashes[key] = stored_hash

    # Entries for files not in existing_files are dropped (removed from store)

    return HashStore(hashes=new_hashes)
