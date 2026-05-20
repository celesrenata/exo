# Implementation Plan: Local Model Documentation Generator

## Overview

Implement a standalone CLI tool (`generate-docs`) that scans exo source files, identifies undocumented code entities, invokes local LLM servers via HTTP to generate documentation, validates output, and writes results. The pipeline follows: scan → filter → generate → validate → write, with incremental hashing, dry-run support, and strict validation modes.

## Tasks

- [x] 1. Set up project structure and core data models
  - [x] 1.1 Create module layout and data models
    - Create `src/exo/docgen/__init__.py` with package exports
    - Create `src/exo/docgen/models.py` with all Pydantic data models: `ParameterInfo`, `AttributeInfo`, `DocumentableEntity`, `ScanResult`, `HashStore`, `GeneratedDocstring`, `RouteInfo`, `FieldInfo`, `PlannedWrite`, `WriteResult`, `ValidationFailure`, `
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    `
    - Define type aliases: `ModelAlias`, `DocumentationType`, `EntityKind`
    - All models use `frozen=True`, `strict=True`, camelCase aliases
    - Create `src/exo/docgen/generators/__init__.py`
    - Create `src/exo/docgen/tests/__init__.py`
    - Create test fixtures directory `src/exo/docgen/tests/fixtures/` with `documented.py`, `undocumented.py`, `mixed.py`, `syntax_error.py`
    - _Requirements: 1.1, 1.2, 3.1, 3.3, 5.1, 5.3_

- [x] 2. Implement source scanner
  - [x] 2.1 Implement scanner module
    - Create `src/exo/docgen/scanner.py`
    - Implement `scan_file(path: Path) -> ScanResult` using `ast.parse()` to walk the AST
    - Identify `DocumentableEntity` instances (functions, classes, methods, modules)
    - Check docstring presence: first statement is `ast.Expr` with `ast.Constant` string value that is non-whitespace
    - Extract parameters with type annotations from `ast.FunctionDef.args`
    - Extract return type annotations from `ast.FunctionDef.returns`
    - Extract raised exceptions by walking `ast.Raise` nodes
    - Extract public instance attributes from `self.x = ...` in `__init__`
    - Compute `documentation_score` as `round(documented / total, 2)`
    - Implement file filter: `.py` extension within `src/exo/`, `rust/`, `dashboard/`
    - Handle syntax errors and permission errors gracefully (log warning, skip)
    - _Requirements: 1.1, 1.2, 1.4, 1.5_

  - [x] 2.2 Write property test for scanner entity identification
    - **Property 1: Scanner identifies exactly the undocumented entities**
    - **Validates: Requirements 1.1**

  - [x] 2.3 Write property test for documentation score computation
    - **Property 2: Documentation score is correctly bounded and computed**
    - **Validates: Requirements 1.2**

  - [x] 2.4 Write property test for summary report sorting
    - **Property 3: Summary report is sorted by score ascending**
    - **Validates: Requirements 1.3**

  - [x] 2.5 Write property test for file path filtering
    - **Property 4: File path filter accepts only .py files in target directories**
    - **Validates: Requirements 1.5**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement hash store for incremental updates
  - [x] 4.1 Implement hash store module
    - Create `src/exo/docgen/hash_store.py`
    - Implement `load_hash_store(path: Path) -> HashStore` to read `.docgen_hashes.json`
    - Implement `save_hash_store(store: HashStore, path: Path) -> None`
    - Implement `compute_file_hash(path: Path) -> str` using `hashlib.sha256`
    - Implement `filter_changed_files(files: list[Path], store: HashStore, force: bool) -> list[Path]`
    - Implement `update_hash_store(store: HashStore, scanned_files: list[Path], existing_files: set[Path]) -> HashStore`
    - Handle corrupt/missing hash store by treating all files as changed
    - Remove entries for files that no longer exist on disk
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 4.2 Write property test for incremental hash filtering
    - **Property 11: Incremental hash filtering correctness**
    - **Validates: Requirements 7.1, 7.2**

  - [x] 4.3 Write property test for hash store post-run state
    - **Property 12: Hash store reflects current filesystem state after run**
    - **Validates: Requirements 7.3, 7.5**

  - [x] 4.4 Write property test for force mode bypass
    - **Property 13: Force mode bypasses incremental filtering**
    - **Validates: Requirements 8.2**

- [x] 5. Implement model client and model selector
  - [x] 5.1 Implement model client
    - Create `src/exo/docgen/model_client.py`
    - Implement HTTP client using `httpx` for POST to `/v1/chat/completions`
    - Configure base URL from `DOCGEN_MODEL_URL` env var (default: `http://localhost:11434`)
    - Configure model alias → actual model name mapping from `DOCGEN_MODEL_MAP` env var
    - Set timeout to 60 seconds, temperature to 0.3, max_tokens to 2048
    - Return `None` on timeout or HTTP error, log warning
    - _Requirements: 2.1, 2.4_

  - [x] 5.2 Implement model selector
    - Create `src/exo/docgen/model_selector.py`
    - Implement `select_model(doc_type: DocumentationType) -> ModelAlias` as a pure function
    - Map: docstring → deepseek, api_documentation → deepseek, module_documentation → condense, architecture_documentation → condense, scaffolding → fast, formatting → fast
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

  - [x] 5.3 Write property test for model selection totality
    - **Property 5: Model selection is total and non-overlapping**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.5**

- [x] 6. Implement generators
  - [x] 6.1 Implement docstring generator
    - Create `src/exo/docgen/generators/docstring.py`
    - Build prompts from function/class/module metadata (signature, params, return type, raises)
    - Parse model output into `GeneratedDocstring` sections
    - Handle indentation-aware insertion using line-based text manipulation
    - Omit empty sections (no Args if no params, no Returns if no return annotation, no Raises if no exceptions)
    - Summary line ≤ 79 characters
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 6.2 Write property test for docstring section matching
    - **Property 6: Generated docstring sections match function metadata**
    - **Validates: Requirements 3.1, 3.2**

  - [x] 6.3 Write property test for docstring indentation preservation
    - **Property 7: Docstring insertion preserves source indentation**
    - **Validates: Requirements 3.5, 3.6**

  - [x] 6.4 Implement module documentation generator
    - Create `src/exo/docgen/generators/module_doc.py`
    - Aggregate public API from all files in a module directory
    - Generate README.md with module summary, public classes/functions list, usage example
    - Preserve manual sections between `<!-- manual -->` and `<!-- /manual -->` markers
    - Handle modules with no public API (summary + note only)
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 6.5 Write property test for README manual section preservation
    - **Property 8: README regeneration preserves manual sections**
    - **Validates: Requirements 4.3**

  - [x] 6.6 Implement API documentation generator
    - Create `src/exo/docgen/generators/api_doc.py`
    - Extract FastAPI route decorators, parameter annotations, Pydantic model schemas from AST
    - Generate markdown sections with HTTP method, path, query params, request/response body, status codes
    - Group routes by first path segment, sort alphabetically within groups
    - Omit request body section when route has no request body parameter
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 6.7 Write property test for route documentation completeness
    - **Property 9: Route documentation completeness matches route parameters**
    - **Validates: Requirements 5.1, 5.4**

  - [x] 6.8 Write property test for route grouping and sorting
    - **Property 10: Routes are grouped by prefix and sorted alphabetically**
    - **Validates: Requirements 5.2**

  - [x] 6.9 Implement architecture documentation generator
    - Create `src/exo/docgen/generators/architecture.py`
    - Generate `docs/architecture.md` with sections for Router, Worker, Master, Election, API components
    - Include role summary and pub/sub topics for each component
    - Generate Mermaid diagrams with components as nodes and topics as labeled edges
    - Include event sourcing message flow section with all topic details
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement validator
  - [x] 8.1 Implement output validator
    - Create `src/exo/docgen/validator.py`
    - Implement docstring validation: Args section required if entity has parameters, Returns section required if entity has return annotation, summary ≤ 79 chars
    - Implement module doc validation: at least one non-empty paragraph, list of classes/functions, at least one code example
    - Return `list[ValidationFailure]` with file path and missing sections
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 8.2 Write property test for validation detection
    - **Property 15: Validation detects missing required sections**
    - **Validates: Requirements 9.1, 9.2**

  - [x] 8.3 Write property test for invalid documentation blocking
    - **Property 16: Invalid documentation is never written to disk**
    - **Validates: Requirements 9.3**

- [x] 9. Implement file writer
  - [x] 9.1 Implement writer module
    - Create `src/exo/docgen/writer.py`
    - Implement `write_planned(writes: list[PlannedWrite], dry_run: bool) -> list[WriteResult]`
    - In normal mode: create parent directories, write content, report results
    - In dry-run mode: produce unified diffs for updates, full content with `[create]` label for new files
    - Print summary list of all planned operations
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 9.2 Write property test for dry-run filesystem safety
    - **Property 17: Dry-run never modifies filesystem**
    - **Validates: Requirements 10.2**

  - [x] 9.3 Write property test for dry-run output correctness
    - **Property 18: Dry-run produces correct diff/content output**
    - **Validates: Requirements 10.3, 10.4**

- [x] 10. Implement CLI entry point and target filtering
  - [x] 10.1 Implement CLI module
    - Create `src/exo/docgen/cli.py`
    - Implement `main()` function using `argparse` with flags: `--dry-run`, `--force`, `--strict`, `--target <path>`, `--output <directory>`
    - Parse arguments into `
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ` model
    - Wire pipeline: scan → filter → generate → validate → write → update hash store
    - Implement target path restriction (recursive scan within target only)
    - Handle exit codes: 0 on success, 1 on strict validation failure or invalid target
    - Create output directory if it doesn't exist
    - Register entry point in `pyproject.toml` as `generate-docs = "exo.docgen.cli:main"`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 9.4, 9.5_

  - [x] 10.2 Write property test for target flag scope restriction
    - **Property 14: Target flag restricts scanning scope**
    - **Validates: Requirements 8.3**

  - [x] 10.3 Write unit tests for CLI argument parsing
    - Test all flag combinations (dry-run, force, strict, target, output)
    - Test invalid target path exits with non-zero code
    - Test output directory creation
    - _Requirements: 8.1, 8.5, 8.6_

- [x] 11. Integration and wiring
  - [x] 11.1 Wire full pipeline end-to-end
    - Connect scanner → hash store filter → model selector → generators → validator → writer → hash store update
    - Implement summary report output (file paths, scores, undocumented counts, sorted by score ascending)
    - Implement strict mode: continue processing all files, collect all validation failures, exit non-zero at end
    - Implement error handling: log warnings via `loguru.logger.warning()`, skip on errors, continue
    - _Requirements: 1.3, 2.4, 9.3, 9.4, 9.5_

  - [x] 11.2 Write integration tests
    - Test end-to-end run against fixture directory with known source files
    - Test model client integration with mock HTTP server
    - Test full pipeline dry-run verification against fixtures
    - Test hash store persistence across multiple runs
    - _Requirements: 1.1, 7.1, 7.3, 10.2_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (18 total)
- Unit tests validate specific examples and edge cases
- All models use Pydantic with `frozen=True`, `strict=True` per project conventions
- Uses `hypothesis` with `@settings(max_examples=100)` for property-based tests
- Uses `pytest` for unit and integration tests
- Test files go in `src/exo/docgen/tests/`
- Entry point registered in `pyproject.toml` under `[project.scripts]`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "4.1", "5.1", "5.2"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "2.5", "4.2", "4.3", "4.4", "5.3"] },
    { "id": 3, "tasks": ["6.1", "6.4", "6.6", "6.9"] },
    { "id": 4, "tasks": ["6.2", "6.3", "6.5", "6.7", "6.8"] },
    { "id": 5, "tasks": ["8.1", "9.1"] },
    { "id": 6, "tasks": ["8.2", "8.3", "9.2", "9.3"] },
    { "id": 7, "tasks": ["10.1"] },
    { "id": 8, "tasks": ["10.2", "10.3"] },
    { "id": 9, "tasks": ["11.1"] },
    { "id": 10, "tasks": ["11.2"] }
  ]
}
```
