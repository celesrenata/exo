# Feature: local-model-documentation-generator, Property 1: Scanner identifies exactly the undocumented entities
# **Validates: Requirements 1.1**
"""Property-based test verifying the scanner correctly identifies documented vs undocumented entities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import NamedTuple

import hypothesis.strategies as st
from hypothesis import given, settings

from exo.docgen.scanner import scan_file


class EntitySpec(NamedTuple):
    kind: str
    has_docstring: bool
    methods: list[bool]


@st.composite
def source_spec(draw: st.DrawFn) -> tuple[bool, list[EntitySpec]]:
    """Generate a spec for a Python source file with a mix of documented/undocumented entities."""
    module_has_docstring = draw(st.booleans())
    num_entities = draw(st.integers(min_value=0, max_value=8))
    entities: list[EntitySpec] = []
    for _ in range(num_entities):
        kind = draw(st.sampled_from(["function", "class"]))
        has_docstring = draw(st.booleans())
        if kind == "function":
            methods: list[bool] = []
        else:
            num_methods = draw(st.integers(min_value=0, max_value=4))
            methods = [draw(st.booleans()) for _ in range(num_methods)]
        entities.append(EntitySpec(kind=kind, has_docstring=has_docstring, methods=methods))
    return (module_has_docstring, entities)


def render_source(module_has_docstring: bool, entities: list[EntitySpec]) -> str:
    """Render entity specs into valid Python source code."""
    lines: list[str] = []
    if module_has_docstring:
        lines.append('"""Module docstring."""')
    for i, spec in enumerate(entities):
        if spec.kind == "function":
            lines.append(f"def func_{i}():")
            if spec.has_docstring:
                lines.append('    """Docstring."""')
            lines.append("    pass")
        else:
            lines.append(f"class Class_{i}:")
            if spec.has_docstring:
                lines.append('    """Docstring."""')
            if not spec.has_docstring and not spec.methods:
                lines.append("    pass")
            for j, m_has_doc in enumerate(spec.methods):
                lines.append(f"    def method_{i}_{j}(self):")
                if m_has_doc:
                    lines.append('        """Docstring."""')
                lines.append("        pass")
    return "\n".join(lines)


def build_expected(entities: list[EntitySpec]) -> dict[str, bool]:
    """Build a mapping of non-module entity name -> expected has_docstring."""
    expected: dict[str, bool] = {}
    for i, spec in enumerate(entities):
        name = f"func_{i}" if spec.kind == "function" else f"Class_{i}"
        expected[name] = spec.has_docstring
        if spec.kind == "class":
            for j, m_has_doc in enumerate(spec.methods):
                expected[f"method_{i}_{j}"] = m_has_doc
    return expected


@settings(max_examples=100)
@given(spec=source_spec())
def test_scanner_identifies_exactly_undocumented_entities(
    spec: tuple[bool, list[EntitySpec]],
) -> None:
    """Property 1: Scanner identifies exactly the undocumented entities.

    For any valid Python source file containing a mix of documented and undocumented
    functions, classes, and methods, the scanner SHALL return exactly those entities
    whose first body statement is not a non-whitespace string constant, and no others.
    """
    module_has_docstring, entities = spec
    source = render_source(module_has_docstring, entities)
    expected_map = build_expected(entities)

    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)  # noqa: SIM115
        tmp_file.write(source)
        tmp_file.flush()
        tmp_file.close()
        tmp_path = Path(tmp_file.name)

        result = scan_file(tmp_path)

        # Verify module entity docstring status
        module_entities = [e for e in result.entities if e.kind == "module"]
        assert len(module_entities) == 1
        assert module_entities[0].has_docstring == module_has_docstring

        # Verify all non-module entities match expected
        non_module = [e for e in result.entities if e.kind != "module"]
        assert len(non_module) == len(expected_map), (
            f"Expected {len(expected_map)} entities, got {len(non_module)}. "
            f"Expected names: {sorted(expected_map.keys())}, "
            f"Got names: {sorted(e.name for e in non_module)}"
        )

        for entity in non_module:
            assert entity.name in expected_map, f"Unexpected entity: {entity.name}"
            assert entity.has_docstring == expected_map[entity.name], (
                f"Entity '{entity.name}': expected has_docstring={expected_map[entity.name]}, "
                f"got {entity.has_docstring}"
            )

        # Verify undocumented_count is correct
        total = len(result.entities)
        documented = sum(1 for e in result.entities if e.has_docstring)
        assert result.undocumented_count == total - documented
    finally:
        if tmp_file:
            Path(tmp_file.name).unlink(missing_ok=True)
