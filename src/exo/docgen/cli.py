"""CLI entry point for the documentation generator.

Provides the `main()` function registered as the `generate-docs` console script.
Orchestrates the pipeline: scan → filter → generate → validate → write → update hash store.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from exo.docgen.generators.api_doc import (
    extract_routes_from_file,
    generate_api_documentation,
)
from exo.docgen.generators.architecture import generate_architecture_documentation
from exo.docgen.generators.docstring import (
    build_prompt,
    format_docstring,
    insert_docstring,
    parse_docstring_response,
)
from exo.docgen.generators.module_doc import generate_module_documentation
from exo.docgen.hash_store import (
    filter_changed_files,
    load_hash_store,
    save_hash_store,
    update_hash_store,
)
from exo.docgen.model_client import chat_completion
from exo.docgen.model_selector import select_model
from exo.docgen.models import (
    GenerateDocsArgs,
    PlannedWrite,
    RouteInfo,
    ScanResult,
    ValidationFailure,
)
from exo.docgen.scanner import is_target_file, scan_file
from exo.docgen.validator import (
    validate_docstring as validate_docstring_output,
)
from exo.docgen.validator import (
    validate_module_documentation,
)
from exo.docgen.writer import write_planned


def _parse_args() -> GenerateDocsArgs:
    """Parse command-line arguments into a GenerateDocsArgs model."""
    parser = argparse.ArgumentParser(
        prog="generate-docs",
        description="Scan source files and generate documentation using local models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview planned changes without writing files to disk.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate documentation for all files regardless of hash state.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit with code 1 if any validation failures occur.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Recursively scan only files within this path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs"),
        help="Output directory for generated documentation (default: docs).",
    )

    parsed = parser.parse_args()
    return GenerateDocsArgs(
        dry_run=parsed.dry_run,  # pyright: ignore[reportAny]
        force=parsed.force,  # pyright: ignore[reportAny]
        strict=parsed.strict,  # pyright: ignore[reportAny]
        target=parsed.target,  # pyright: ignore[reportAny]
        output=parsed.output,  # pyright: ignore[reportAny]
    )


def _collect_python_files(target: Path | None) -> list[Path]:
    """Collect Python source files to process.

    If target is provided, recursively collects all .py files under that path.
    Otherwise, collects .py files from the current working directory that pass
    the is_target_file filter (within src/exo/, rust/, dashboard/).
    """
    files: list[Path] = []

    if target is not None:
        for path in sorted(target.rglob("*.py")):
            if path.is_file():
                files.append(path)
    else:
        root = Path.cwd()
        for path in sorted(root.rglob("*.py")):
            if path.is_file() and is_target_file(path):
                files.append(path)

    return files


def _print_summary_report(scan_results: list[ScanResult]) -> None:
    """Print a summary report of scan results sorted by documentation score ascending."""
    sorted_results = sorted(scan_results, key=lambda result: result.documentation_score)

    print("\n--- Documentation Summary Report ---")
    print(f"{'File':<60} {'Score':>6} {'Undocumented':>12}")
    print("-" * 80)

    for result in sorted_results:
        print(f"{result.file_path:<60} {result.documentation_score:>6.2f} {result.undocumented_count:>12}")

    total_undocumented = sum(result.undocumented_count for result in sorted_results)
    print("-" * 80)
    print(f"{'Total files: ' + str(len(sorted_results)):<60} {'':>6} {total_undocumented:>12}")


async def _run_pipeline(args: GenerateDocsArgs) -> None:
    """Execute the full documentation generation pipeline asynchronously."""
    # Validate --target path exists
    if args.target is not None and not args.target.exists():
        print(f"Error: target path does not exist: {args.target}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it does not exist
    args.output.mkdir(parents=True, exist_ok=True)

    # Collect Python source files
    files = _collect_python_files(args.target)

    if not files:
        logger.info("No Python files found to process.")
        sys.exit(0)

    # Load hash store from project root (current working directory)
    hash_store_directory = Path.cwd()
    store = load_hash_store(hash_store_directory)

    # Filter to only changed files (unless --force)
    changed_files = filter_changed_files(files, store, force=args.force)

    if not changed_files:
        logger.info("No files have changed since last run. Nothing to do.")
        sys.exit(0)

    # Scan each changed file for documentable entities
    scan_results: list[ScanResult] = []
    for file_path in changed_files:
        result = scan_file(file_path)
        scan_results.append(result)

    # --- Generation Pipeline ---
    all_writes: list[PlannedWrite] = []
    validation_failures: list[ValidationFailure] = []
    module_directories: set[Path] = set()

    # Select model alias for docstring generation
    docstring_alias = select_model("docstring")

    # Process each scanned file: generate docstrings for undocumented entities
    for scan_result in scan_results:
        file_path = Path(scan_result.file_path)

        # Track module directories (parent dirs with __init__.py)
        parent_directory = file_path.parent
        init_file = parent_directory / "__init__.py"
        if init_file.exists():
            module_directories.add(parent_directory)

        # Get undocumented entities
        undocumented_entities = [
            entity for entity in scan_result.entities if not entity.has_docstring
        ]
        if not undocumented_entities:
            continue

        # Read source file
        try:
            source_text = file_path.read_text(encoding="utf-8")
            source_lines = source_text.splitlines(keepends=True)
        except Exception as error:
            logger.warning("Could not read {}: {}", file_path, error)
            continue

        file_had_failures = False

        # Process each undocumented entity
        for entity in undocumented_entities:
            try:
                # Build prompt and call model
                system_prompt, user_prompt = build_prompt(entity)
                raw_response = await chat_completion(
                    docstring_alias, system_prompt, user_prompt
                )

                if raw_response is None:
                    logger.warning(
                        "Model '{}' returned no response for entity '{}' in {}",
                        docstring_alias,
                        entity.name,
                        file_path,
                    )
                    continue

                raw_response = raw_response.strip()
                if not raw_response:
                    logger.warning(
                        "Model '{}' returned empty response for entity '{}' in {}",
                        docstring_alias,
                        entity.name,
                        file_path,
                    )
                    continue

                # Parse response into structured docstring
                generated = parse_docstring_response(raw_response, entity)

                # Validate the generated docstring
                entity_failures = validate_docstring_output(
                    generated, entity, str(file_path)
                )
                if entity_failures:
                    validation_failures.extend(entity_failures)
                    file_had_failures = True
                    continue

                # Format and insert docstring into source lines
                formatted = format_docstring(generated, entity.indentation_level)
                source_lines = insert_docstring(source_lines, entity, formatted)

            except Exception as error:
                logger.warning(
                    "Error generating docstring for '{}' in {}: {}",
                    entity.name,
                    file_path,
                    error,
                )
                continue

        # Create PlannedWrite for the updated source file (only if no failures)
        if not file_had_failures:
            new_content = "".join(source_lines)
            if new_content != source_text:
                all_writes.append(
                    PlannedWrite(
                        destination=file_path,
                        content=new_content,
                        action="update",
                        existing_content=source_text,
                    )
                )

    # Generate module documentation for each module directory
    for module_directory in sorted(module_directories):
        try:
            readme_path = module_directory / "README.md"
            existing_readme: str | None = None
            if readme_path.exists():
                existing_readme = readme_path.read_text(encoding="utf-8")

            module_content = await generate_module_documentation(
                module_directory, existing_readme
            )
            if module_content is None:
                logger.warning(
                    "Failed to generate module documentation for {}",
                    module_directory,
                )
                continue

            # Validate module documentation
            module_failures = validate_module_documentation(
                module_content, str(readme_path)
            )
            if module_failures:
                validation_failures.extend(module_failures)
                continue

            all_writes.append(
                PlannedWrite(
                    destination=readme_path,
                    content=module_content,
                    action="update" if existing_readme is not None else "create",
                    existing_content=existing_readme,
                )
            )
        except Exception as error:
            logger.warning(
                "Error generating module documentation for {}: {}",
                module_directory,
                error,
            )

    # Generate API documentation from src/exo/api/ files
    project_root = Path.cwd()
    api_directory = project_root / "src" / "exo" / "api"
    if api_directory.is_dir():
        try:
            all_routes: list[RouteInfo] = []
            for api_file in sorted(api_directory.glob("*.py")):
                try:
                    routes = extract_routes_from_file(api_file)
                    all_routes.extend(routes)
                except Exception as error:
                    logger.warning(
                        "Failed to extract routes from {}: {}", api_file, error
                    )

            if all_routes:
                api_markdown = generate_api_documentation(all_routes)
                api_destination = args.output / "api_endpoints.md"
                existing_api: str | None = None
                if api_destination.exists():
                    existing_api = api_destination.read_text(encoding="utf-8")

                all_writes.append(
                    PlannedWrite(
                        destination=api_destination,
                        content=api_markdown,
                        action="update" if existing_api is not None else "create",
                        existing_content=existing_api,
                    )
                )
        except Exception as error:
            logger.warning("Error generating API documentation: {}", error)

    # Generate architecture documentation
    try:
        architecture_write = await generate_architecture_documentation(
            args.output, project_root
        )
        if architecture_write is not None:
            all_writes.append(architecture_write)
    except Exception as error:
        logger.warning("Error generating architecture documentation: {}", error)

    # Write all planned changes
    write_planned(all_writes, dry_run=args.dry_run)

    # Print summary report sorted by score ascending
    _print_summary_report(scan_results)

    # Update hash store with current file hashes
    existing_files = set(files)
    updated_store = update_hash_store(store, changed_files, existing_files)

    if not args.dry_run:
        save_hash_store(updated_store, hash_store_directory)

    # Exit code: 1 if strict mode and validation failures exist
    if args.strict and validation_failures:
        logger.error(
            "Strict mode: {} validation failure(s) detected.",
            len(validation_failures),
        )
        sys.exit(1)

    sys.exit(0)


def main() -> None:
    """Entry point for the generate-docs CLI command."""
    args = _parse_args()
    asyncio.run(_run_pipeline(args))
