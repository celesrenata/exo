"""Pure function mapping documentation types to local model aliases."""

from exo.docgen.models import DocumentationType, ModelAlias

_DOCUMENTATION_TYPE_TO_MODEL: dict[DocumentationType, ModelAlias] = {
    "docstring": "deepseek",
    "api_documentation": "deepseek",
    "module_documentation": "condense",
    "architecture_documentation": "condense",
    "scaffolding": "fast",
    "formatting": "fast",
}


def select_model(doc_type: DocumentationType) -> ModelAlias:
    """Select the appropriate model alias for a given documentation type.

    The mapping is total (covers all DocumentationType values) and non-overlapping
    (each type maps to exactly one alias).

    Args:
        doc_type: The type of documentation to generate.

    Returns:
        The model alias that should handle this documentation type.
    """
    return _DOCUMENTATION_TYPE_TO_MODEL[doc_type]
