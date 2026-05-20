# Feature: local-model-documentation-generator, Property 9: Route documentation completeness matches route parameters
"""Property-based tests for API documentation generators.

Validates: Requirements 5.1, 5.4
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.docgen.generators.api_doc import generate_route_markdown
from exo.docgen.models import FieldInfo, RouteInfo

_field_info_strategy = st.builds(
    lambda **kwargs: FieldInfo(**kwargs),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    name=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20),
    type_annotation=st.sampled_from(["str", "int", "float", "bool", "list[str]"]),
    is_required=st.booleans(),
    default_value=st.one_of(st.none(), st.text(min_size=1, max_size=10)),
)


@st.composite
def route_info_strategy(draw: st.DrawFn) -> RouteInfo:
    """Generate random RouteInfo instances with and without request/response body models."""
    http_method = draw(st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH"]))
    prefix = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10))
    suffix = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_{}", min_size=1, max_size=15))
    path = f"/{prefix}/{suffix}"
    path_prefix = prefix

    query_parameters = draw(st.lists(_field_info_strategy, min_size=0, max_size=3))

    request_body_model = draw(
        st.one_of(
            st.none(),
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20),
        )
    )
    request_body_fields = (
        draw(st.lists(_field_info_strategy, min_size=0, max_size=3))
        if request_body_model is not None
        else []
    )

    response_body_model = draw(
        st.one_of(
            st.none(),
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20),
        )
    )
    response_body_fields = (
        draw(st.lists(_field_info_strategy, min_size=0, max_size=3))
        if response_body_model is not None
        else []
    )

    status_codes = draw(
        st.lists(st.sampled_from([200, 201, 400, 401, 403, 404, 500]), min_size=0, max_size=4, unique=True)
    )

    return RouteInfo(
        http_method=http_method,
        path=path,
        path_prefix=path_prefix,
        query_parameters=query_parameters,
        request_body_model=request_body_model,
        request_body_fields=request_body_fields,
        response_body_model=response_body_model,
        response_body_fields=response_body_fields,
        status_codes=sorted(status_codes),
    )


@given(route=route_info_strategy())
@settings(max_examples=100)
def test_route_documentation_completeness_matches_route_parameters(route: RouteInfo) -> None:
    """Property 9: Route documentation completeness matches route parameters.

    **Validates: Requirements 5.1, 5.4**
    """
    markdown = generate_route_markdown(route)

    # 1. The markdown always contains the HTTP method and path
    assert route.http_method in markdown
    assert route.path in markdown

    # 2. Request Body section appears iff request_body_model is not None
    has_request_body_section = "**Request Body:**" in markdown
    if route.request_body_model is not None:
        assert has_request_body_section
    else:
        assert not has_request_body_section

    # 3. Response Body section appears iff response_body_model is not None
    has_response_body_section = "**Response Body:**" in markdown
    if route.response_body_model is not None:
        assert has_response_body_section
    else:
        assert not has_response_body_section
