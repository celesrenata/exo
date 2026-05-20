"""API documentation generator for FastAPI routes.

Extracts route definitions from Python source files using AST analysis,
groups them by URL path prefix, and generates markdown documentation
for each endpoint including HTTP method, path, parameters, request/response
bodies, and status codes.
"""

from __future__ import annotations

import ast
from pathlib import Path

from loguru import logger

from exo.docgen.models import FieldInfo, RouteInfo

HTTP_METHODS: frozenset[str] = frozenset(
    {"get", "post", "put", "delete", "patch", "head"}
)

SIMPLE_TYPES: frozenset[str] = frozenset(
    {"str", "int", "float", "bool", "bytes", "None"}
)

SKIP_PARAM_TYPES: frozenset[str] = frozenset(
    {"Request", "UploadFile", "File", "Form"}
)


def _extract_path_prefix(path: str) -> str:
    """Extract the first non-empty segment from a URL path."""
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return ""
    first = segments[0]
    if first.startswith("{") and first.endswith("}"):
        return first[1:-1]
    return first


def _extract_path_parameters(path: str) -> set[str]:
    """Extract path parameter names from a URL path template."""
    parameters: set[str] = set()
    for segment in path.split("/"):
        if segment.startswith("{") and segment.endswith("}"):
            param_name = segment[1:-1].split(":")[0]
            parameters.add(param_name)
    return parameters


def _is_simple_type(type_annotation: str) -> bool:
    """Determine if a type annotation represents a simple/query parameter type."""
    cleaned = type_annotation.replace(" ", "")

    if cleaned in SIMPLE_TYPES:
        return True

    if cleaned.startswith(("list[", "List[")):
        return True

    if " | None" in type_annotation or "| None" in type_annotation:
        base = type_annotation.replace(" | None", "").replace("| None", "").strip()
        if base in SIMPLE_TYPES:
            return True
        if base.startswith(("list[", "List[")):
            return True

    if "Annotated[" in type_annotation or "Annotated [" in type_annotation:
        return True

    return cleaned.endswith(("Id", "ID"))


def _has_query_annotation(node: ast.expr) -> bool:
    """Check if an annotation contains Query() in an Annotated type."""
    source = ast.unparse(node)
    return "Query" in source and "Annotated" in source


def _is_request_type(type_annotation: str) -> bool:
    """Check if a type annotation is a Request or similar non-parameter type."""
    cleaned = type_annotation.strip()
    return cleaned in SKIP_PARAM_TYPES or any(
        cleaned.startswith(skip_type) for skip_type in SKIP_PARAM_TYPES
    )


def _extract_status_codes(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[int]:
    """Extract HTTP status codes from HTTPException raises in a function body."""
    codes: set[int] = {200}

    for node in ast.walk(func_node):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc
        if exc is None:
            continue

        if isinstance(exc, ast.Call):
            func = exc.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr

            if func_name == "HTTPException":
                for keyword in exc.keywords:
                    if keyword.arg == "status_code":
                        value = keyword.value
                        if isinstance(value, ast.Constant) and isinstance(
                            value.value, int
                        ):
                            codes.add(value.value)
                        elif isinstance(value, ast.Attribute):
                            attr_name = value.attr
                            status_map: dict[str, int] = {
                                "BAD_REQUEST": 400,
                                "UNAUTHORIZED": 401,
                                "FORBIDDEN": 403,
                                "NOT_FOUND": 404,
                                "METHOD_NOT_ALLOWED": 405,
                                "CONFLICT": 409,
                                "INTERNAL_SERVER_ERROR": 500,
                                "NOT_IMPLEMENTED": 501,
                                "SERVICE_UNAVAILABLE": 503,
                            }
                            if attr_name in status_map:
                                codes.add(status_map[attr_name])
                if exc.args:
                    first_arg = exc.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(
                        first_arg.value, int
                    ):
                        codes.add(first_arg.value)

    return sorted(codes)


def _extract_handler_parameters(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    path: str,
) -> tuple[list[FieldInfo], str | None, list[FieldInfo]]:
    """Extract query parameters and request body info from a handler function."""
    path_params = _extract_path_parameters(path)
    query_parameters: list[FieldInfo] = []
    request_body_model: str | None = None
    request_body_fields: list[FieldInfo] = []

    args = func_node.args
    num_pos = len(args.args)
    num_defaults = len(args.defaults)

    for i, arg in enumerate(args.args):
        name = arg.arg

        if i == 0 and name in ("self", "cls"):
            continue

        if name in path_params:
            continue

        if arg.annotation is None:
            continue

        type_annotation = ast.unparse(arg.annotation)

        if _is_request_type(type_annotation):
            continue

        is_required = i < num_pos - num_defaults
        default_value: str | None = None
        if not is_required:
            idx = i - (num_pos - num_defaults)
            if idx < len(args.defaults):
                default_value = ast.unparse(args.defaults[idx])

        has_query = _has_query_annotation(arg.annotation)

        if has_query or _is_simple_type(type_annotation):
            query_parameters.append(
                FieldInfo(
                    name=name,
                    type_annotation=type_annotation,
                    is_required=is_required,
                    default_value=default_value,
                )
            )
        elif request_body_model is None:
            request_body_model = type_annotation

    for i, arg in enumerate(args.kwonlyargs):
        name = arg.arg
        if name in path_params:
            continue
        if arg.annotation is None:
            continue

        type_annotation = ast.unparse(arg.annotation)
        if _is_request_type(type_annotation):
            continue

        kw_default = args.kw_defaults[i] if i < len(args.kw_defaults) else None
        is_required = kw_default is None
        default_value = (
            ast.unparse(kw_default) if kw_default is not None else None
        )

        has_query = _has_query_annotation(arg.annotation)
        if has_query or _is_simple_type(type_annotation):
            query_parameters.append(
                FieldInfo(
                    name=name,
                    type_annotation=type_annotation,
                    is_required=is_required,
                    default_value=default_value,
                )
            )
        elif request_body_model is None:
            request_body_model = type_annotation

    return query_parameters, request_body_model, request_body_fields


def _extract_return_model(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    """Extract the response model name from a function's return annotation."""
    if func_node.returns is None:
        return None
    annotation = ast.unparse(func_node.returns)
    if annotation in SIMPLE_TYPES or annotation == "None":
        return None
    skip_returns = {"JSONResponse", "StreamingResponse", "FileResponse", "Response"}
    if annotation in skip_returns:
        return None
    return annotation


def _find_method_in_class(
    class_node: ast.ClassDef, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find a method definition within a class by name."""
    for stmt in class_node.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef) and stmt.name == method_name:
            return stmt
    return None


def _extract_route_call(
    node: ast.Call,
) -> tuple[str, str, dict[str, ast.expr]] | None:
    """Extract HTTP method, path, and kwargs from a route registration call."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None

    method_name = func.attr
    if method_name not in HTTP_METHODS:
        return None

    value = func.value
    if isinstance(value, ast.Attribute):
        if value.attr != "app":
            return None
    elif isinstance(value, ast.Name):
        if value.id not in ("app", "router"):
            return None
    else:
        return None

    if not node.args:
        return None
    path_arg = node.args[0]
    if not isinstance(path_arg, ast.Constant) or not isinstance(path_arg.value, str):
        return None

    path: str = path_arg.value

    kwargs: dict[str, ast.expr] = {}
    for keyword in node.keywords:
        if keyword.arg is not None:
            kwargs[keyword.arg] = keyword.value

    return method_name, path, kwargs


def _extract_handler_name(node: ast.expr) -> str | None:
    """Extract the handler function/method name from a route handler expression."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def extract_routes_from_file(path: Path) -> list[RouteInfo]:
    """Parse a Python file and extract all FastAPI route definitions."""
    try:
        source = path.read_text(encoding="utf-8")
    except (PermissionError, OSError) as error:
        logger.warning("Cannot read file {}: {}", path, error)
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        logger.warning("Syntax error in {}: {}", path, error)
        return []

    routes: list[RouteInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _extract_routes_from_class(node, routes)

    _extract_decorated_routes(tree, routes)

    return routes


def _extract_routes_from_class(
    class_node: ast.ClassDef, routes: list[RouteInfo]
) -> None:
    """Extract routes from a class using self.app.<method>(<path>)(<handler>) pattern."""
    for method_node in class_node.body:
        if not isinstance(method_node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue

        for stmt in ast.walk(method_node):
            if not isinstance(stmt, ast.Expr):
                continue
            expr = stmt.value
            if not isinstance(expr, ast.Call):
                continue

            inner_func = expr.func
            if not isinstance(inner_func, ast.Call):
                continue

            route_info = _extract_route_call(inner_func)
            if route_info is None:
                continue

            http_method, route_path, _kwargs = route_info

            handler_name: str | None = None
            if expr.args:
                handler_name = _extract_handler_name(expr.args[0])

            query_parameters: list[FieldInfo] = []
            request_body_model: str | None = None
            request_body_fields: list[FieldInfo] = []
            response_body_model: str | None = None
            status_codes: list[int] = [200]

            if handler_name is not None:
                handler_func = _find_method_in_class(class_node, handler_name)
                if handler_func is not None:
                    query_parameters, request_body_model, request_body_fields = (
                        _extract_handler_parameters(handler_func, route_path)
                    )
                    response_body_model = _extract_return_model(handler_func)
                    status_codes = _extract_status_codes(handler_func)

            path_prefix = _extract_path_prefix(route_path)

            routes.append(
                RouteInfo(
                    http_method=http_method.upper(),
                    path=route_path,
                    path_prefix=path_prefix,
                    query_parameters=query_parameters,
                    request_body_model=request_body_model,
                    request_body_fields=request_body_fields,
                    response_body_model=response_body_model,
                    response_body_fields=[],
                    status_codes=status_codes,
                )
            )


def _extract_decorated_routes(
    tree: ast.Module, routes: list[RouteInfo]
) -> None:
    """Extract routes defined with decorator syntax (@app.get, @router.get, etc.)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue

            route_info = _extract_route_call(decorator)
            if route_info is None:
                continue

            http_method, route_path, _kwargs = route_info

            query_parameters, request_body_model, request_body_fields = (
                _extract_handler_parameters(node, route_path)
            )
            response_body_model = _extract_return_model(node)
            status_codes = _extract_status_codes(node)
            path_prefix = _extract_path_prefix(route_path)

            routes.append(
                RouteInfo(
                    http_method=http_method.upper(),
                    path=route_path,
                    path_prefix=path_prefix,
                    query_parameters=query_parameters,
                    request_body_model=request_body_model,
                    request_body_fields=request_body_fields,
                    response_body_model=response_body_model,
                    response_body_fields=[],
                    status_codes=status_codes,
                )
            )


def group_routes_by_prefix(routes: list[RouteInfo]) -> dict[str, list[RouteInfo]]:
    """Group routes by their URL path prefix and sort alphabetically within groups."""
    groups: dict[str, list[RouteInfo]] = {}
    for route in routes:
        prefix = route.path_prefix
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(route)

    for prefix in groups:
        groups[prefix] = sorted(groups[prefix], key=lambda r: (r.path, r.http_method))

    return groups


def generate_route_markdown(route: RouteInfo) -> str:
    """Generate markdown documentation for a single API route."""
    lines: list[str] = []

    lines.append(f"### {route.http_method} `{route.path}`")
    lines.append("")

    if route.query_parameters:
        lines.append("**Query Parameters:**")
        lines.append("")
        lines.append("| Name | Type | Required | Default |")
        lines.append("|------|------|----------|---------|")
        for param in route.query_parameters:
            required = "Yes" if param.is_required else "No"
            default = param.default_value if param.default_value else "\u2014"
            lines.append(
                f"| `{param.name}` | `{param.type_annotation}` | {required} | {default} |"
            )
        lines.append("")

    if route.request_body_model is not None:
        lines.append(f"**Request Body:** `{route.request_body_model}`")
        lines.append("")
        if route.request_body_fields:
            lines.append("| Field | Type | Required | Default |")
            lines.append("|-------|------|----------|---------|")
            for field in route.request_body_fields:
                required = "Yes" if field.is_required else "No"
                default = field.default_value if field.default_value else "\u2014"
                lines.append(
                    f"| `{field.name}` | `{field.type_annotation}` | {required} | {default} |"
                )
            lines.append("")

    if route.response_body_model is not None:
        lines.append(f"**Response Body:** `{route.response_body_model}`")
        lines.append("")
        if route.response_body_fields:
            lines.append("| Field | Type | Required | Default |")
            lines.append("|-------|------|----------|---------|")
            for field in route.response_body_fields:
                required = "Yes" if field.is_required else "No"
                default = field.default_value if field.default_value else "\u2014"
                lines.append(
                    f"| `{field.name}` | `{field.type_annotation}` | {required} | {default} |"
                )
            lines.append("")

    if route.status_codes:
        lines.append("**Status Codes:**")
        lines.append("")
        for code in route.status_codes:
            description = _status_code_description(code)
            lines.append(f"- `{code}` {description}")
        lines.append("")

    return "\n".join(lines)


def _status_code_description(code: int) -> str:
    """Get a human-readable description for an HTTP status code."""
    descriptions: dict[int, str] = {
        200: "OK",
        201: "Created",
        204: "No Content",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        422: "Unprocessable Entity",
        500: "Internal Server Error",
        501: "Not Implemented",
        503: "Service Unavailable",
    }
    return descriptions.get(code, "")


def generate_api_documentation(routes: list[RouteInfo]) -> str:
    """Generate complete API documentation markdown from a list of routes."""
    lines: list[str] = []

    lines.append("# API Endpoints")
    lines.append("")

    if not routes:
        lines.append("No API endpoints found.")
        lines.append("")
        return "\n".join(lines)

    groups = group_routes_by_prefix(routes)
    sorted_prefixes = sorted(groups.keys())

    lines.append("## Table of Contents")
    lines.append("")
    for prefix in sorted_prefixes:
        display_prefix = prefix if prefix else "root"
        lines.append(f"- [{display_prefix}](#{display_prefix})")
    lines.append("")

    for prefix in sorted_prefixes:
        display_prefix = prefix if prefix else "root"
        lines.append(f"## {display_prefix}")
        lines.append("")

        for route in groups[prefix]:
            lines.append(generate_route_markdown(route))

    return "\n".join(lines)
