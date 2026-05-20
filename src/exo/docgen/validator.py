"""Output completeness validation for generated documentation."""

from __future__ import annotations

import re

from loguru import logger

from exo.docgen.models import (
    DocumentableEntity,
    GeneratedDocstring,
    ValidationFailure,
)

_NUMBERED_LIST_PATTERN = re.compile(r"^\d+\.\s+")


def validate_docstring(
    docstring: GeneratedDocstring,
    entity: DocumentableEntity,
    file_path: str,
) -> list[ValidationFailure]:
    """Validate a generated docstring against its entity's structural requirements.

    Checks that required sections are present based on the entity's metadata
    and that the summary line respects the 79-character limit.
    """
    failures: list[ValidationFailure] = []

    # Check required Args section when entity has parameters
    if entity.parameters and docstring.args_section is None:
        message = (
            f"Entity '{entity.name}' in {file_path} has parameters "
            f"but generated docstring lacks an Args section."
        )
        failures.append(
            ValidationFailure(
                file_path=file_path,
                missing_sections=["Args"],
                message=message,
            )
        )
        logger.warning(message)

    # Check required Returns section when entity has return annotation
    if entity.return_annotation is not None and docstring.returns_section is None:
        message = (
            f"Entity '{entity.name}' in {file_path} has a return annotation "
            f"but generated docstring lacks a Returns section."
        )
        failures.append(
            ValidationFailure(
                file_path=file_path,
                missing_sections=["Returns"],
                message=message,
            )
        )
        logger.warning(message)

    # Check summary first line is ≤ 79 characters
    first_line = docstring.summary.split("\n", maxsplit=1)[0]
    if len(first_line) > 79:
        message = (
            f"Summary line for '{entity.name}' in {file_path} is "
            f"{len(first_line)} characters, exceeding the 79-character limit."
        )
        failures.append(
            ValidationFailure(
                file_path=file_path,
                missing_sections=["Summary"],
                message=message,
            )
        )
        logger.warning(message)

    return failures


def validate_module_documentation(
    content: str,
    file_path: str,
) -> list[ValidationFailure]:
    """Validate that module documentation meets structural requirements.

    Checks for at least one non-empty paragraph, a list of classes/functions,
    and at least one fenced code example.
    """
    failures: list[ValidationFailure] = []
    lines = content.splitlines()

    has_paragraph = _check_has_paragraph(lines)
    if not has_paragraph:
        message = (
            f"Module documentation at {file_path} lacks a non-empty paragraph "
            f"describing the module's purpose."
        )
        failures.append(
            ValidationFailure(
                file_path=file_path,
                missing_sections=["Overview"],
                message=message,
            )
        )
        logger.warning(message)

    has_list = _check_has_list(lines)
    if not has_list:
        message = (
            f"Module documentation at {file_path} does not contain a list "
            f"of classes/functions (bullet or numbered list)."
        )
        failures.append(
            ValidationFailure(
                file_path=file_path,
                missing_sections=["List of Classes/Functions"],
                message=message,
            )
        )
        logger.warning(message)

    has_code_example = _check_has_code_example(lines)
    if not has_code_example:
        message = (
            f"Module documentation at {file_path} does not contain any "
            f"code examples (fenced code blocks)."
        )
        failures.append(
            ValidationFailure(
                file_path=file_path,
                missing_sections=["Code Examples"],
                message=message,
            )
        )
        logger.warning(message)

    return failures


def _check_has_paragraph(lines: list[str]) -> bool:
    """Check if lines contain at least one non-empty paragraph line.

    A paragraph line is a non-blank line that is not a heading, not inside
    a fenced code block, and not a list item.
    """
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith(("- ", "* ")):
            continue
        if _NUMBERED_LIST_PATTERN.match(stripped):
            continue
        return True
    return False


def _check_has_list(lines: list[str]) -> bool:
    """Check if lines contain at least one bullet or numbered list item."""
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith(("- ", "* ")):
            return True
        if _NUMBERED_LIST_PATTERN.match(stripped):
            return True
    return False


def _check_has_code_example(lines: list[str]) -> bool:
    """Check if lines contain at least one fenced code block."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            return True
    return False
