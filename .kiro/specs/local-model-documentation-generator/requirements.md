# Requirements Document

## Introduction

The Local Model Documentation Generator analyzes source files in the exo project, identifies undocumented or poorly documented code, and uses local models (DeepSeek, Condense, Fast) to produce comprehensive documentation. The generator produces Python docstrings, module README files, API endpoint documentation, architecture overviews, and component interaction diagrams. It supports incremental generation to reprocess only changed files, provides a dry-run mode for previewing changes, validates output completeness, and operates as a CLI command via `uv`.

## Glossary

- **Source_File**: Any file under `src/exo/`, `rust/`, or `dashboard/` that contains code requiring documentation.
- **Documentation_Generator**: The CLI tool that orchestrates scanning, model invocation, and documentation output.
- **Local_Model**: One of the three MCP-accessible models (deepseek, condense, fast) used for documentation generation.
- **MCP_Tool**: The `mcp_local_model_local_model_chat` interface with an `alias` parameter to select a model.
- **Docstring**: Google-style inline documentation for Python functions, classes, and modules.
- **Module_Documentation**: A `README.md` file placed in the root directory of a source module.
- **API_Documentation**: Markdown describing FastAPI route parameters, request bodies, responses, and behavior.
- **Architecture_Documentation**: Markdown with Mermaid diagrams describing component interactions and data flow.
- **Content_Hash**: A SHA-256 hash of a source file's content used for change detection.
- **Hash_Store**: A JSON file that persists content hashes from the previous generation run.
- **Documentation_Score**: The ratio of documented entities to total documentable entities in a source file.
- **Dry_Run_Mode**: An execution mode that displays planned changes without writing any files to disk.
- **Documentable_Entity**: A Python function, class, method, or module that can receive a docstring.

## Requirements

### Requirement 1: Source File Scanning

**User Story:** As a developer, I want the Documentation_Generator to scan all source files and identify undocumented code, so that I can target documentation efforts efficiently.

#### Acceptance Criteria

1. WHEN the Documentation_Generator scans a Source_File, THE Documentation_Generator SHALL identify every Documentable_Entity that lacks a docstring, where "lacks a docstring" means the entity has no docstring or has a docstring containing only whitespace.
2. WHEN the Documentation_Generator completes scanning a Source_File, THE Documentation_Generator SHALL compute a Documentation_Score as a decimal value between 0.0 and 1.0, rounded to 2 decimal places, representing the ratio of documented entities to total Documentable_Entities.
3. WHEN the Documentation_Generator completes scanning all Source_Files, THE Documentation_Generator SHALL produce a summary report to stdout listing each Source_File path, its Documentation_Score, and the count of undocumented entities, sorted by Documentation_Score in ascending order.
4. IF a Source_File cannot be parsed due to syntax errors or cannot be read due to permission errors, THEN THE Documentation_Generator SHALL log a warning containing the file path and error description, skip that file, and continue scanning remaining Source_Files.
5. WHEN the Documentation_Generator scans directories, THE Documentation_Generator SHALL process only Python files (`.py` extension) within the `src/exo/`, `rust/`, and `dashboard/` directories as candidates for Documentable_Entity identification.

### Requirement 2: Model Selection

**User Story:** As a developer, I want the Documentation_Generator to select the appropriate Local_Model for each documentation type, so that documentation quality and resource usage are optimized.

#### Acceptance Criteria

1. WHEN the Documentation_Generator generates Docstrings or API_Documentation, THE Documentation_Generator SHALL invoke the MCP_Tool with alias "deepseek".
2. WHEN the Documentation_Generator generates Module_Documentation or Architecture_Documentation, THE Documentation_Generator SHALL invoke the MCP_Tool with alias "condense".
3. WHEN the Documentation_Generator generates initial file scaffolding from templates or applies formatting corrections to existing docstrings without changing semantic content, THE Documentation_Generator SHALL invoke the MCP_Tool with alias "fast".
4. IF the selected Local_Model fails to respond within 60 seconds or returns an error, THEN THE Documentation_Generator SHALL log a warning identifying the failed model and the target Source_File, skip generation for that entity, and continue processing remaining entities.
5. THE Documentation_Generator SHALL assign each documentation generation task to exactly one Local_Model based on the documentation type, with no task matching more than one model selection criterion.

### Requirement 3: Docstring Generation

**User Story:** As a developer, I want the Documentation_Generator to produce Google-style docstrings for Python functions, classes, and modules, so that inline documentation follows a consistent standard.

#### Acceptance Criteria

1. WHEN the Documentation_Generator processes an undocumented Python function, THE Documentation_Generator SHALL generate a Google-style docstring containing a one-line summary of at most 79 characters, an Args section listing each parameter with its type annotation and description, a Returns section describing the return value and type, and a Raises section listing exceptions that the function explicitly raises.
2. IF a processed Python function has no parameters, no return value, or raises no exceptions, THEN THE Documentation_Generator SHALL omit the corresponding Args, Returns, or Raises section from the generated docstring rather than including an empty section.
3. WHEN the Documentation_Generator processes an undocumented Python class, THE Documentation_Generator SHALL generate a Google-style docstring containing a one-line summary of at most 79 characters, an Attributes section listing each public instance attribute assigned in `__init__` with its type and description, and an Args section documenting the `__init__` method parameters with their types and descriptions.
4. WHEN the Documentation_Generator processes an undocumented Python module, THE Documentation_Generator SHALL generate a module-level docstring containing a one-line summary and a description of the module's purpose and public interface.
5. WHEN the Documentation_Generator inserts a generated docstring, THE Documentation_Generator SHALL place it as the first statement in the function, class, or module body, indented to match the indentation level of that body.
6. THE Documentation_Generator SHALL preserve existing code indentation when inserting generated docstrings.

### Requirement 4: Module Documentation

**User Story:** As a developer, I want the Documentation_Generator to create a README.md for each source module directory, so that module-level purpose and usage are documented.

#### Acceptance Criteria

1. WHEN the Documentation_Generator processes a module directory, THE Documentation_Generator SHALL generate a README.md containing a module summary of 1 to 3 sentences, a list of all public classes and public functions defined in the module's source files, and at least one usage example demonstrating a primary entry point of the module.
2. THE Documentation_Generator SHALL place the generated README.md in the root of the corresponding module directory.
3. WHEN a README.md already exists in a module directory, THE Documentation_Generator SHALL update the existing file by regenerating auto-generated sections while preserving any manually written sections enclosed between `<!-- manual -->` and `<!-- /manual -->` comment pairs.
4. IF a module directory contains no public classes or public functions, THEN THE Documentation_Generator SHALL generate a README.md containing only the module summary and a note indicating no public API is exported.

### Requirement 5: API Endpoint Documentation

**User Story:** As a developer, I want the Documentation_Generator to document every FastAPI route in the API component, so that endpoint behavior is clear for consumers.

#### Acceptance Criteria

1. WHEN the Documentation_Generator processes a FastAPI route definition in `src/exo/api/`, THE Documentation_Generator SHALL generate a markdown section listing the HTTP method, path, query parameters (with name, type, and default value), request body schema (Pydantic model name and fields), response body schema (Pydantic model name and fields), and documented status codes (success and error codes returned by the handler).
2. THE Documentation_Generator SHALL produce a single `docs/api_endpoints.md` file that aggregates all route documentation grouped by URL path prefix (the first path segment after the leading slash, such as `v1`, `ollama`, `bench`, `instance`, `models`, `images`, `download`, `state`, `events`) and sorted alphabetically within each group.
3. WHEN a route uses Pydantic models for request or response bodies, THE Documentation_Generator SHALL include each model's field name, field type annotation, whether the field is required or optional, and the default value if one is defined.
4. IF a route has no request body parameter, THEN THE Documentation_Generator SHALL omit the request body section for that endpoint and display only the HTTP method, path, query parameters, response schema, and status codes.
5. IF a route file in `src/exo/api/` cannot be parsed due to a syntax error or import failure, THEN THE Documentation_Generator SHALL skip that file, log a warning message identifying the file path and error, and continue processing the remaining route files.

### Requirement 6: Architecture Documentation

**User Story:** As a developer, I want the Documentation_Generator to create an architecture overview with Mermaid diagrams, so that the system design is visible and maintainable.

#### Acceptance Criteria

1. THE Documentation_Generator SHALL generate a `docs/architecture.md` file containing a dedicated section for each of the Router, Worker, Master, Election, and API components, where each section includes a one-paragraph role summary and a list of the pub/sub topics that component publishes to or subscribes from.
2. THE Documentation_Generator SHALL embed at least one Mermaid diagram in `docs/architecture.md` using a fenced code block with the `mermaid` language identifier, and each embedded diagram SHALL be syntactically valid such that a Mermaid renderer produces a visual output without parse errors.
3. THE Documentation_Generator SHALL include a section describing the event sourcing message flow that lists each of the GLOBAL_EVENTS, LOCAL_EVENTS, COMMANDS, ELECTION_MESSAGES, and CONNECTION_MESSAGES topics with its publishing component, subscribing component(s), message type, and publish policy.
4. WHEN the Documentation_Generator generates a Mermaid diagram for component interactions, THE Documentation_Generator SHALL represent each of the five components (Router, Worker, Master, Election, API) as named nodes and each pub/sub topic as a labeled edge indicating message direction.

### Requirement 7: Incremental Updates

**User Story:** As a developer, I want the Documentation_Generator to reprocess only source files that have changed since the last run, so that redundant work is avoided.

#### Acceptance Criteria

1. WHILE the Documentation_Generator runs in incremental mode, THE Documentation_Generator SHALL compare the Content_Hash of each Source_File against the corresponding entry in the Hash_Store, and SHALL treat any Source_File with no corresponding entry in the Hash_Store as changed.
2. WHILE the Documentation_Generator runs in incremental mode, THE Documentation_Generator SHALL skip documentation generation for any Source_File whose Content_Hash matches the stored hash.
3. WHEN the Documentation_Generator completes a generation run, THE Documentation_Generator SHALL update the Hash_Store with the current Content_Hash of every Source_File that was scanned, including both regenerated and skipped files.
4. IF the Hash_Store file does not exist or cannot be parsed as valid JSON, THEN THE Documentation_Generator SHALL treat all Source_Files as changed and create a new Hash_Store upon completion.
5. WHEN a Source_File that has an entry in the Hash_Store no longer exists on disk, THE Documentation_Generator SHALL remove that entry from the Hash_Store upon completion of the generation run.

### Requirement 8: CLI Interface

**User Story:** As a developer, I want to invoke the Documentation_Generator as a CLI command via `uv`, so that I can integrate documentation generation into development workflows.

#### Acceptance Criteria

1. THE Documentation_Generator SHALL provide a CLI command `uv run generate-docs` that accepts the flags `--dry-run`, `--force`, `--strict`, `--target <path>`, and `--output <directory>`.
2. WHEN the `--force` flag is provided, THE Documentation_Generator SHALL regenerate documentation for all Source_Files regardless of Content_Hash state.
3. WHEN the `--target` flag is provided with a path, THE Documentation_Generator SHALL recursively scan and generate documentation for all Source_Files within the specified path and its subdirectories.
4. WHEN the `--output` flag is provided with a directory path, THE Documentation_Generator SHALL write all generated markdown files to the specified directory instead of the default `docs/` directory.
5. IF the `--target` flag specifies a path that does not exist or contains no Source_Files, THEN THE Documentation_Generator SHALL exit with a non-zero status code and print an error message indicating the invalid target path.
6. IF the `--output` flag specifies a directory that does not exist, THEN THE Documentation_Generator SHALL create the directory before writing output files.
7. WHEN the Documentation_Generator completes successfully with no validation failures, THE Documentation_Generator SHALL exit with status code 0.

### Requirement 9: Output Validation

**User Story:** As a developer, I want the Documentation_Generator to validate generated documentation for completeness, so that incomplete output is flagged before it is written.

#### Acceptance Criteria

1. WHEN the Documentation_Generator generates a docstring for a function that has a return type annotation, THE Documentation_Generator SHALL verify that the docstring contains a Returns section, and WHEN the function has one or more parameters, THE Documentation_Generator SHALL verify that the docstring contains an Args section listing each parameter.
2. WHEN the Documentation_Generator generates a Module_Documentation file, THE Documentation_Generator SHALL verify that the file contains a module summary section of at least one non-empty paragraph, a list of key classes and functions, and at least one usage example.
3. IF validation fails for any generated documentation, THEN THE Documentation_Generator SHALL log a warning to stderr containing the file path and the list of missing required sections, and SHALL skip writing the invalid documentation to disk.
4. IF validation fails and the `--strict` flag is active, THEN THE Documentation_Generator SHALL complete validation of all remaining files, log all validation failures, and exit with a non-zero status code after all files have been processed.
5. IF validation fails and the `--strict` flag is not active, THEN THE Documentation_Generator SHALL continue processing subsequent files after skipping the invalid documentation.

### Requirement 10: Dry-Run Mode

**User Story:** As a developer, I want to preview planned documentation changes without modifying any files, so that I can review and approve changes before generation.

#### Acceptance Criteria

1. WHEN the Documentation_Generator runs in Dry_Run_Mode, THE Documentation_Generator SHALL print to stdout a list of every Source_File that would be processed and every documentation file that would be created or updated, labeling each output file as either "create" or "update".
2. WHILE Dry_Run_Mode is active, THE Documentation_Generator SHALL not create, modify, or delete any files on disk.
3. WHEN the Documentation_Generator runs in Dry_Run_Mode and a documentation file would be modified, THE Documentation_Generator SHALL display a unified diff preview comparing the existing file content to the proposed content.
4. WHEN the Documentation_Generator runs in Dry_Run_Mode and a documentation file would be created, THE Documentation_Generator SHALL display the full proposed file content prefixed with a "create" label and the destination path.
5. WHEN the Documentation_Generator completes a Dry_Run_Mode execution, THE Documentation_Generator SHALL exit with status code 0.
