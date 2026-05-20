"""Module documentation generator.

Aggregates public API from all Python files in a module directory and generates
a README.md with module summary, public classes/functions list, and usage example.
Preserves manually written sections between <!-- manual --> and <!-- /manual --> markers.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from exo.docgen.model_client import chat_completion
from exo.docgen.models import DocumentableEntity, ScanResult
from exo.docgen.scanner import scan_file


def collect_public_api(module_directory: Path) -> list[DocumentableEntity]:
    """Scan all .py files in the given directory and return public classes and functions.

    Entities are considered public if their kind is 'class' or 'function' and their
    name does not start with an underscore. Methods and module-level entities are excluded.

    Args:
        module_directory: Path to the module directory to scan.

    Returns:
        List of public DocumentableEntity instances found in the module.
    """
    public_entities: list[DocumentableEntity] = []
    if not module_directory.is_dir():
        logger.warning("Module path {} is not a directory; no entities collected", module_directory)
        return public_entities

    for py_file in sorted(module_directory.glob("*.py")):
        try:
            scan_result: ScanResult = scan_file(py_file)
        except Exception as exc:
            logger.warning("Failed to scan {}: {}", py_file, exc)
            continue

        for entity in scan_result.entities:
            if entity.kind in ("class", "function") and not entity.name.startswith("_"):
                public_entities.append(entity)

    return public_entities


def _find_manual_blocks(text: str) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each <!-- manual --> ... <!-- /manual --> block.

    The markers themselves are included in the returned ranges.
    """
    blocks: list[tuple[int, int]] = []
    start_marker = "<!-- manual -->"
    end_marker = "<!-- /manual -->"
    position = 0
    while True:
        start_index = text.find(start_marker, position)
        if start_index == -1:
            break
        end_index = text.find(end_marker, start_index + len(start_marker))
        if end_index == -1:
            # Malformed: no closing marker; treat rest of text as the block
            end_index = len(text)
        else:
            end_index += len(end_marker)
        blocks.append((start_index, end_index))
        position = end_index
    return blocks


def preserve_manual_sections(existing_content: str, new_content: str) -> str:
    """Merge manual sections from existing README into newly generated content.

    Finds all <!-- manual --> ... <!-- /manual --> blocks in the existing content
    and replaces corresponding blocks in the new content with the existing content
    byte-for-byte. If the new content has fewer manual blocks than the existing
    content, the extra existing blocks are appended at the end.

    Args:
        existing_content: The current README content containing manual sections.
        new_content: The newly generated README content.

    Returns:
        The new content with manual sections from the existing content preserved.
    """
    existing_blocks = _find_manual_blocks(existing_content)
    new_blocks = _find_manual_blocks(new_content)

    if not existing_blocks:
        return new_content

    result_parts: list[str] = []
    last_end = 0
    existing_index = 0

    for new_start, new_end in new_blocks:
        # Append text before the current new block
        result_parts.append(new_content[last_end:new_start])

        if existing_index < len(existing_blocks):
            # Replace with the corresponding existing block
            old_start, old_end = existing_blocks[existing_index]
            result_parts.append(existing_content[old_start:old_end])
            existing_index += 1
        else:
            # No more existing blocks; keep the new block as-is
            result_parts.append(new_content[new_start:new_end])

        last_end = new_end

    # Append any remaining text after the last new block
    result_parts.append(new_content[last_end:])

    # Append any remaining existing blocks that were not matched
    for old_start, old_end in existing_blocks[existing_index:]:
        result_parts.append("\n\n")
        result_parts.append(existing_content[old_start:old_end])

    return "".join(result_parts)


def _format_class_description(entity: DocumentableEntity) -> str:
    """Format a class entity for inclusion in the model prompt."""
    attributes = [f"`{a.name}`" for a in entity.public_attributes] if entity.public_attributes else []
    attribute_string = ", ".join(attributes) if attributes else "none"
    return (
        f"- **{entity.name}** (line {entity.line_number})\n"
        f"  - Public attributes: {attribute_string}\n"
        f"  - Has docstring: {'yes' if entity.has_docstring else 'no'}"
    )


def _format_function_description(entity: DocumentableEntity) -> str:
    """Format a function entity for inclusion in the model prompt."""
    parameters = [
        f"`{p.name}: {p.type_annotation or 'Any'}`" for p in entity.parameters
    ]
    parameter_string = ", ".join(parameters) if parameters else "none"
    return_type = entity.return_annotation or "None"
    return (
        f"- **{entity.name}** (line {entity.line_number})\n"
        f"  - Parameters: {parameter_string}\n"
        f"  - Returns: `{return_type}`\n"
        f"  - Has docstring: {'yes' if entity.has_docstring else 'no'}"
    )


async def generate_module_documentation(
    module_directory: Path,
    existing_readme: str | None = None,
) -> str | None:
    """Generate a README.md for a Python module directory.

    Scans the module directory for public classes and functions, builds a prompt
    describing the public API, calls the model client to generate documentation,
    and formats the result as markdown. If an existing README is provided, manual
    sections are preserved.

    Args:
        module_directory: Path to the module directory.
        existing_readme: Optional existing README content to preserve manual sections from.

    Returns:
        The generated README markdown string, or None if the model call fails.
    """
    public_api = collect_public_api(module_directory)
    module_name = module_directory.name

    # Handle modules with no public API
    if not public_api:
        logger.warning("No public API found in module '{}'; generating minimal README", module_name)
        minimal_readme = (
            f"# {module_name}\n\n"
            f"This module has no public classes or functions.\n"
        )
        if existing_readme is not None:
            return preserve_manual_sections(existing_readme, minimal_readme)
        return minimal_readme

    # Build the prompt
    classes = [entity for entity in public_api if entity.kind == "class"]
    functions = [entity for entity in public_api if entity.kind == "function"]

    class_lines = [_format_class_description(cls) for cls in classes]
    function_lines = [_format_function_description(func) for func in functions]

    user_prompt = (
        f"Module directory: `{module_name}`\n\n"
        "## Public API\n\n"
        "### Classes\n"
        + ("\n".join(class_lines) if class_lines else "None")
        + "\n\n### Functions\n"
        + ("\n".join(function_lines) if function_lines else "None")
        + "\n\nPlease produce:\n"
        "1. A 1-3 sentence summary of the module's purpose.\n"
        "2. A markdown list of the public API (classes and functions).\n"
        "3. A usage example in a fenced Python code block.\n"
    )

    system_prompt = (
        "You are a technical writer creating documentation for a Python module. "
        "Your output must be valid Markdown. "
        "Do not include any extra commentary outside the requested sections. "
        "Start with a module heading using the module name."
    )

    # Call the model client
    response = await chat_completion(
        alias="condense",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    if response is None:
        logger.warning("Model call failed for module '{}'", module_name)
        return None

    # Format the response as markdown
    new_readme = response.strip()
    if not new_readme.startswith("#"):
        new_readme = f"# {module_name}\n\n{new_readme}"

    # Preserve manual sections from existing README
    if existing_readme is not None:
        new_readme = preserve_manual_sections(existing_readme, new_readme)

    return new_readme
