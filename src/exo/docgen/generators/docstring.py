"""Google-style docstring generator using local model inference.

Builds prompts from function/class/module metadata, calls the model client,
parses the response into structured sections, and inserts the formatted
docstring into source code with correct indentation.
"""

from __future__ import annotations

from loguru import logger

from exo.docgen.model_client import chat_completion
from exo.docgen.models import (
    DocumentableEntity,
    GeneratedDocstring,
    ModelAlias,
)

SYSTEM_PROMPT: str = (
    "You are a Python documentation assistant. "
    "Produce a Google-style docstring for the given Python entity. "
    "Follow this format exactly:\n\n"
    "Summary line (one sentence, at most 79 characters, ending with a period).\n\n"
    "Extended description if needed (optional).\n\n"
    "Args:\n"
    "    param_name (type): Description.\n"
    "    param_name (type, optional): Description. Defaults to value.\n\n"
    "Returns:\n"
    "    type: Description of return value.\n\n"
    "Raises:\n"
    "    ExceptionType: Description of when raised.\n\n"
    "Attributes:\n"
    "    attr_name (type): Description.\n\n"
    "Rules:\n"
    "- Omit any section that has no content.\n"
    "- Do NOT include Args if there are no parameters.\n"
    "- Do NOT include Returns if there is no return type.\n"
    "- Do NOT include Raises if no exceptions are raised.\n"
    "- Do NOT include Attributes if there are no public attributes.\n"
    "- Keep the summary line under 79 characters.\n"
    "- For modules, only produce a summary and optional description.\n"
    "- Output ONLY the docstring content (no triple quotes, no code fences)."
)


def build_prompt(entity: DocumentableEntity) -> tuple[str, str]:
    """Construct system and user prompts for the model client.

    Args:
        entity: The documentable entity to generate a docstring for.

    Returns:
        A tuple of (system_prompt, user_prompt).
    """
    parts: list[str] = []

    if entity.kind == "module":
        parts.append(f"Module name: {entity.name}")
        parts.append("Generate a module-level docstring with a summary and description.")
        return SYSTEM_PROMPT, "\n".join(parts)

    kind_label = "Class" if entity.kind == "class" else "Function"
    parts.append(f"{kind_label}: {entity.name}")

    if entity.parameters:
        parts.append("\nParameters:")
        for parameter in entity.parameters:
            type_str = f": {parameter.type_annotation}" if parameter.type_annotation else ""
            default_str = ""
            if not parameter.is_required:
                default_str = f" = {parameter.default_value}" if parameter.default_value else " = None"
            parts.append(f"    {parameter.name}{type_str}{default_str}")

    if entity.return_annotation:
        parts.append(f"\nReturn type: {entity.return_annotation}")

    if entity.raises:
        parts.append("\nRaises:")
        for exception_name in entity.raises:
            parts.append(f"    {exception_name}")

    if entity.kind == "class" and entity.public_attributes:
        parts.append("\nPublic attributes:")
        for attribute in entity.public_attributes:
            type_str = f": {attribute.type_annotation}" if attribute.type_annotation else ""
            parts.append(f"    {attribute.name}{type_str}")

    return SYSTEM_PROMPT, "\n".join(parts)


def parse_docstring_response(
    raw: str,
    entity: DocumentableEntity,
) -> GeneratedDocstring:
    """Parse raw model output into a GeneratedDocstring with structured sections.

    Looks for section headers (Args:, Returns:, Raises:, Attributes:) and
    splits the text accordingly. Everything before the first section header
    is treated as the summary.

    Args:
        raw: The raw text returned by the model.
        entity: The entity being documented (used for section filtering).

    Returns:
        A GeneratedDocstring with parsed sections.
    """
    lines = raw.strip().splitlines()

    # Remove any triple-quote wrappers the model might have added
    if lines and lines[0].strip().startswith('"""'):
        lines[0] = lines[0].strip().removeprefix('"""').strip()
    if lines and lines[-1].strip().endswith('"""'):
        lines[-1] = lines[-1].strip().removesuffix('"""').strip()
    # Remove empty lines at start/end after stripping quotes
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    section_headers = ("args:", "returns:", "raises:", "attributes:")

    # Find section boundaries
    section_indices: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        stripped_lower = line.strip().lower()
        for header in section_headers:
            if stripped_lower == header or stripped_lower.startswith(header):
                section_name = header.rstrip(":")
                section_indices.append((section_name, index))
                break

    # Extract summary (everything before first section)
    first_section_line = section_indices[0][1] if section_indices else len(lines)
    summary_lines = lines[:first_section_line]
    # Strip trailing blank lines from summary
    while summary_lines and not summary_lines[-1].strip():
        summary_lines.pop()
    summary = "\n".join(summary_lines).strip()

    # Truncate summary first line to 79 chars if needed
    summary_first_line = summary.split("\n")[0] if summary else ""
    if len(summary_first_line) > 79:
        summary_first_line = summary_first_line[:76] + "..."
        remaining_summary_lines = summary.split("\n")[1:]
        remaining_summary = "\n".join(remaining_summary_lines)
        summary = summary_first_line + ("\n" + remaining_summary if remaining_summary else "")

    # Extract each section's content
    def _extract_section_content(section_start: int) -> str | None:
        """Extract content lines for a section starting at section_start."""
        section_end = len(lines)
        for _name, idx in section_indices:
            if idx > section_start:
                section_end = idx
                break
        content_lines = lines[section_start + 1 : section_end]
        # Strip trailing blank lines
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()
        if not content_lines:
            return None
        return "\n".join(content_lines)

    args_section: str | None = None
    returns_section: str | None = None
    raises_section: str | None = None
    attributes_section: str | None = None

    for section_name, section_index in section_indices:
        content = _extract_section_content(section_index)
        if section_name == "args":
            args_section = content
        elif section_name == "returns":
            returns_section = content
        elif section_name == "raises":
            raises_section = content
        elif section_name == "attributes":
            attributes_section = content

    # Enforce omission rules based on entity metadata
    if not entity.parameters:
        args_section = None
    if not entity.return_annotation:
        returns_section = None
    if not entity.raises:
        raises_section = None
    if not entity.public_attributes:
        attributes_section = None

    return GeneratedDocstring(
        summary=summary,
        args_section=args_section,
        returns_section=returns_section,
        raises_section=raises_section,
        attributes_section=attributes_section,
    )


def format_docstring(docstring: GeneratedDocstring, indentation_level: int) -> str:
    """Format a GeneratedDocstring into a properly indented Python docstring string.

    The docstring body is indented one level deeper than the entity's def/class
    statement (indentation_level + 4 spaces). For modules (indentation_level=0),
    the body has 4 spaces of indentation.

    Args:
        docstring: The parsed docstring to format.
        indentation_level: The column offset of the entity's def/class line.

    Returns:
        A multi-line string containing the triple-quoted docstring.
    """
    body_indent = " " * (indentation_level + 4)

    output_lines: list[str] = []

    # Summary line(s)
    for line in docstring.summary.splitlines():
        output_lines.append(line)

    # Sections with blank line separator
    sections: list[tuple[str, str]] = []
    if docstring.args_section:
        sections.append(("Args:", docstring.args_section))
    if docstring.returns_section:
        sections.append(("Returns:", docstring.returns_section))
    if docstring.raises_section:
        sections.append(("Raises:", docstring.raises_section))
    if docstring.attributes_section:
        sections.append(("Attributes:", docstring.attributes_section))

    for header, content in sections:
        output_lines.append("")  # blank line before section
        output_lines.append(header)
        for content_line in content.splitlines():
            output_lines.append(content_line)

    # Build the final docstring with indentation
    result_lines: list[str] = []
    result_lines.append(f'{body_indent}"""')

    for line in output_lines:
        if line.strip():
            result_lines.append(f"{body_indent}{line}")
        else:
            result_lines.append("")

    result_lines.append(f'{body_indent}"""')

    return "\n".join(result_lines)


def _find_body_start_line(source_lines: list[str], entity: DocumentableEntity) -> int:
    """Find the line index where the entity's body starts (0-indexed).

    For functions/methods/classes, this is the line after the signature ends
    (after the colon). Handles multi-line signatures by tracking parenthesis depth.

    Args:
        source_lines: The source file as a list of lines.
        entity: The entity whose body start to find.

    Returns:
        The 0-indexed line number where the body begins.
    """
    if entity.kind == "module":
        # For modules, find insertion point after __future__ imports
        last_future_line = -1
        for index, line in enumerate(source_lines):
            if line.strip().startswith("from __future__"):
                last_future_line = index
        return last_future_line + 1

    start_line = entity.line_number - 1  # Convert to 0-indexed
    paren_depth = 0

    for index in range(start_line, len(source_lines)):
        line = source_lines[index]
        for char in line:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1

        stripped = line.rstrip()
        if paren_depth <= 0 and stripped.endswith(":"):
            return index + 1

    # Fallback: insert after the def line
    logger.warning(
        "Could not find end of signature for '{}' at line {}",
        entity.name,
        entity.line_number,
    )
    return start_line + 1


def insert_docstring(
    source_lines: list[str],
    entity: DocumentableEntity,
    formatted_docstring: str,
) -> list[str]:
    """Insert a formatted docstring into source lines at the correct position.

    For functions, methods, and classes: inserts after the signature line.
    For modules: inserts after any __future__ imports at the top.

    Does not alter indentation of any existing lines.

    Args:
        source_lines: The source file as a list of lines.
        entity: The entity to insert the docstring for.
        formatted_docstring: The pre-formatted docstring string.

    Returns:
        A new list of source lines with the docstring inserted.
    """
    insert_index = _find_body_start_line(source_lines, entity)
    docstring_lines = formatted_docstring.splitlines()

    result = list(source_lines[:insert_index])
    result.extend(docstring_lines)
    result.extend(source_lines[insert_index:])

    return result


async def generate_docstring(
    entity: DocumentableEntity,
    source_lines: list[str],
    alias: ModelAlias = "deepseek",
) -> list[str] | None:
    """Orchestrate the full docstring generation pipeline for a single entity.

    Builds the prompt, calls the model, parses the response, formats the
    docstring, and inserts it into the source lines.

    Args:
        entity: The entity to generate a docstring for.
        source_lines: The current source file as a list of lines.
        alias: The model alias to use for generation.

    Returns:
        The modified source lines with the docstring inserted, or None on failure.
    """
    system_prompt, user_prompt = build_prompt(entity)

    raw_response = await chat_completion(alias, system_prompt, user_prompt)
    if raw_response is None:
        logger.warning(
            "Model '{}' returned no response for entity '{}'", alias, entity.name
        )
        return None

    raw_response = raw_response.strip()
    if not raw_response:
        logger.warning(
            "Model '{}' returned empty response for entity '{}'", alias, entity.name
        )
        return None

    generated = parse_docstring_response(raw_response, entity)
    formatted = format_docstring(generated, entity.indentation_level)
    updated_lines = insert_docstring(source_lines, entity, formatted)

    return updated_lines
