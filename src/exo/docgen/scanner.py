"""Scanner module for identifying documentable entities in Python source files.

Uses the ast module to parse source files and extract functions, classes,
methods, and module-level documentation information.
"""

from __future__ import annotations

import ast
from pathlib import Path

from loguru import logger

from exo.docgen.models import (
    AttributeInfo,
    DocumentableEntity,
    ParameterInfo,
    ScanResult,
)

TARGET_PREFIXES: tuple[str, ...] = ("src/exo/", "rust/", "dashboard/")


def is_target_file(path: Path) -> bool:
    """Return True iff path has .py extension and resides within a target directory."""
    if path.suffix != ".py":
        return False
    posix = path.as_posix()
    return any(prefix in posix for prefix in TARGET_PREFIXES)


def _has_docstring(body: list[ast.stmt]) -> bool:
    """Check if the first statement in body is a non-whitespace string constant."""
    if not body:
        return False
    first = body[0]
    if not isinstance(first, ast.Expr):
        return False
    value = first.value
    if not isinstance(value, ast.Constant):
        return False
    if not isinstance(value.value, str):
        return False
    return value.value.strip() != ""


def _extract_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    is_method: bool,
) -> list[ParameterInfo]:
    """Extract parameter info from a function definition node."""
    args = node.args
    parameters: list[ParameterInfo] = []

    num_pos = len(args.args)
    num_defaults = len(args.defaults)

    for i, arg in enumerate(args.args):
        name = arg.arg
        if is_method and i == 0 and name in ("self", "cls"):
            continue
        is_required = i < num_pos - num_defaults
        default_value: str | None = None
        if not is_required:
            idx = i - (num_pos - num_defaults)
            if idx < len(args.defaults):
                default_value = ast.unparse(args.defaults[idx])
        type_annotation = ast.unparse(arg.annotation) if arg.annotation else None
        parameters.append(ParameterInfo(
            name=name,
            type_annotation=type_annotation,
            default_value=default_value,
            is_required=is_required,
        ))

    for i, arg in enumerate(args.kwonlyargs):
        name = arg.arg
        kw_default = args.kw_defaults[i] if i < len(args.kw_defaults) else None
        is_required = kw_default is None
        default_value = ast.unparse(kw_default) if kw_default is not None else None
        type_annotation = ast.unparse(arg.annotation) if arg.annotation else None
        parameters.append(ParameterInfo(
            name=name,
            type_annotation=type_annotation,
            default_value=default_value,
            is_required=is_required,
        ))

    return parameters


def _extract_raises(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Extract raised exception names from a function body, deduplicated preserving order."""
    raised: list[str] = []
    seen: set[str] = set()

    for subnode in ast.walk(node):
        if not isinstance(subnode, ast.Raise):
            continue
        exc = subnode.exc
        if exc is None:
            continue
        name = _exception_name(exc)
        if name is not None and name not in seen:
            raised.append(name)
            seen.add(name)

    return raised


def _exception_name(exc: ast.expr) -> str | None:
    """Extract the exception class name from a raise expression."""
    if isinstance(exc, ast.Call):
        return _exception_name(exc.func)
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return f"{ast.unparse(exc.value)}.{exc.attr}"
    return None


def _extract_public_attributes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[AttributeInfo]:
    """Extract public instance attributes assigned to self in __init__."""
    if node.name != "__init__":
        return []
    attributes: list[AttributeInfo] = []
    seen_names: set[str] = set()

    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    attr_name = target.attr
                    if not attr_name.startswith("_") and attr_name not in seen_names:
                        seen_names.add(attr_name)
                        attributes.append(AttributeInfo(name=attr_name, type_annotation=None))
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                attr_name = target.attr
                if not attr_name.startswith("_") and attr_name not in seen_names:
                    seen_names.add(attr_name)
                    type_annotation = ast.unparse(stmt.annotation) if stmt.annotation else None
                    attributes.append(AttributeInfo(name=attr_name, type_annotation=type_annotation))

    return attributes


def _extract_function_entity(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    is_method: bool,
) -> DocumentableEntity:
    """Extract a DocumentableEntity from a function or async function definition."""
    return DocumentableEntity(
        kind="method" if is_method else "function",
        name=node.name,
        line_number=node.lineno,
        indentation_level=node.col_offset,
        has_docstring=_has_docstring(node.body),
        parameters=_extract_parameters(node, is_method),
        return_annotation=ast.unparse(node.returns) if node.returns else None,
        raises=_extract_raises(node),
    )


def _extract_class_entities(node: ast.ClassDef) -> list[DocumentableEntity]:
    """Extract class entity and all method entities from a ClassDef node."""
    entities: list[DocumentableEntity] = []

    # Find __init__ to extract public attributes for the class entity
    public_attributes: list[AttributeInfo] = []
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef) and stmt.name == "__init__":
            public_attributes = _extract_public_attributes(stmt)
            break

    class_entity = DocumentableEntity(
        kind="class",
        name=node.name,
        line_number=node.lineno,
        indentation_level=node.col_offset,
        has_docstring=_has_docstring(node.body),
        public_attributes=public_attributes,
    )
    entities.append(class_entity)

    # Method entities
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            entities.append(_extract_function_entity(stmt, is_method=True))

    return entities


def scan_file(path: Path) -> ScanResult:
    """Scan a Python source file and identify all documentable entities.

    Reads the file, parses the AST, and extracts functions, classes, methods,
    and the module entity. Computes a documentation score based on the ratio
    of documented entities to total entities.

    Args:
        path: Path to the Python source file.

    Returns:
        A ScanResult containing all identified entities and documentation score.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except PermissionError as error:
        logger.warning(f"Permission denied reading {path}: {error}")
        return ScanResult(file_path=str(path), entities=[], documentation_score=1.0, undocumented_count=0)

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        logger.warning(f"Syntax error in {path}: {error}")
        return ScanResult(file_path=str(path), entities=[], documentation_score=1.0, undocumented_count=0)

    entities: list[DocumentableEntity] = []

    # Module entity
    module_entity = DocumentableEntity(
        kind="module",
        name=path.stem,
        line_number=1,
        indentation_level=0,
        has_docstring=_has_docstring(tree.body),
    )
    entities.append(module_entity)

    # Top-level definitions
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            entities.append(_extract_function_entity(stmt, is_method=False))
        elif isinstance(stmt, ast.ClassDef):
            entities.extend(_extract_class_entities(stmt))

    # Compute documentation score
    total = len(entities)
    documented = sum(1 for entity in entities if entity.has_docstring)
    undocumented_count = total - documented

    documentation_score = round(documented / total, 2) if total > 0 else 1.0

    return ScanResult(
        file_path=str(path),
        entities=entities,
        documentation_score=documentation_score,
        undocumented_count=undocumented_count,
    )
