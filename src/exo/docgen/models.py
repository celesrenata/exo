"""Pydantic data models for the documentation generator tool."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel

# Type aliases
ModelAlias = Literal["deepseek", "condense", "fast"]
DocumentationType = Literal[
    "docstring",
    "api_documentation",
    "module_documentation",
    "architecture_documentation",
    "scaffolding",
    "formatting",
]
EntityKind = Literal["function", "class", "method", "module"]


class ParameterInfo(FrozenModel):
    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    is_required: bool


class AttributeInfo(FrozenModel):
    name: str
    type_annotation: str | None = None


class DocumentableEntity(FrozenModel):
    kind: EntityKind
    name: str
    line_number: int
    indentation_level: int
    has_docstring: bool
    parameters: list[ParameterInfo] = Field(default_factory=list)
    return_annotation: str | None = None
    raises: list[str] = Field(default_factory=list)
    public_attributes: list[AttributeInfo] = Field(default_factory=list)


class ScanResult(FrozenModel):
    file_path: str
    entities: list[DocumentableEntity] = Field(default_factory=list)
    documentation_score: float
    undocumented_count: int


class HashStore(FrozenModel):
    hashes: dict[str, str] = Field(default_factory=dict)


class GeneratedDocstring(FrozenModel):
    summary: str
    args_section: str | None = None
    returns_section: str | None = None
    raises_section: str | None = None
    attributes_section: str | None = None


class FieldInfo(FrozenModel):
    name: str
    type_annotation: str
    is_required: bool
    default_value: str | None = None


class RouteInfo(FrozenModel):
    http_method: str
    path: str
    path_prefix: str
    query_parameters: list[FieldInfo] = Field(default_factory=list)
    request_body_model: str | None = None
    request_body_fields: list[FieldInfo] = Field(default_factory=list)
    response_body_model: str | None = None
    response_body_fields: list[FieldInfo] = Field(default_factory=list)
    status_codes: list[int] = Field(default_factory=list)


class PlannedWrite(FrozenModel):
    destination: Path
    content: str
    action: Literal["create", "update"]
    existing_content: str | None = None


class WriteResult(FrozenModel):
    destination: Path
    action: Literal["create", "update", "skipped"]
    success: bool


class ValidationFailure(FrozenModel):
    file_path: str
    missing_sections: list[str]
    message: str


class GenerateDocsArgs(FrozenModel):
    dry_run: bool = False
    force: bool = False
    strict: bool = False
    target: Path | None = None
    output: Path = Field(default=Path("docs"))
