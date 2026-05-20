# Feature: local-model-documentation-generator, Property 5: Model selection is total and non-overlapping
"""Property-based tests for model selection totality and non-overlapping.

Validates: Requirements 2.1, 2.2, 2.3, 2.5
"""

from typing import get_args

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.docgen.model_selector import select_model
from exo.docgen.models import DocumentationType, ModelAlias

ALL_DOCUMENTATION_TYPES: list[DocumentationType] = list(get_args(DocumentationType))
ALL_MODEL_ALIASES: list[ModelAlias] = list(get_args(ModelAlias))

documentation_type_strategy = st.sampled_from(ALL_DOCUMENTATION_TYPES)


@settings(max_examples=100)
@given(doc_type=documentation_type_strategy)
def test_select_model_returns_valid_model_alias(doc_type: DocumentationType) -> None:
    """For any valid DocumentationType, select_model returns exactly one valid ModelAlias.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.5**
    """
    result = select_model(doc_type)
    assert result in ALL_MODEL_ALIASES, f"select_model({doc_type!r}) returned {result!r}, not a valid ModelAlias"


@settings(max_examples=100)
@given(doc_type=documentation_type_strategy)
def test_select_model_is_deterministic(doc_type: DocumentationType) -> None:
    """For any valid DocumentationType, calling select_model twice with the same
    input returns the same ModelAlias both times (deterministic mapping).

    **Validates: Requirements 2.1, 2.2, 2.3, 2.5**
    """
    first_call = select_model(doc_type)
    second_call = select_model(doc_type)
    assert first_call == second_call, (
        f"select_model({doc_type!r}) is non-deterministic: {first_call!r} != {second_call!r}"
    )


def test_select_model_is_total() -> None:
    """select_model covers all 6 DocumentationType values without raising exceptions.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.5**
    """
    assert len(ALL_DOCUMENTATION_TYPES) == 6, (
        f"Expected 6 documentation types, got {len(ALL_DOCUMENTATION_TYPES)}"
    )
    for doc_type in ALL_DOCUMENTATION_TYPES:
        result = select_model(doc_type)
        assert result in ALL_MODEL_ALIASES, (
            f"select_model({doc_type!r}) returned {result!r}, not a valid ModelAlias"
        )


def test_select_model_non_overlapping() -> None:
    """Each DocumentationType maps to exactly one ModelAlias. No type maps to
    multiple aliases (the mapping is a function, not a relation).

    **Validates: Requirements 2.1, 2.2, 2.3, 2.5**
    """
    mapping: dict[DocumentationType, ModelAlias] = {}
    for doc_type in ALL_DOCUMENTATION_TYPES:
        result = select_model(doc_type)
        if doc_type in mapping:
            assert mapping[doc_type] == result, (
                f"Overlapping: {doc_type!r} maps to both {mapping[doc_type]!r} and {result!r}"
            )
        mapping[doc_type] = result

    # Every documentation type has exactly one entry
    assert len(mapping) == len(ALL_DOCUMENTATION_TYPES)
