# Feature: local-model-documentation-generator, Property 18: Dry-run produces correct diff/content output
# **Validates: Requirements 10.3, 10.4**

import contextlib
import io
from pathlib import Path

import hypothesis.strategies as st
from hypothesis import given, settings

from exo.docgen.models import PlannedWrite
from exo.docgen.writer import write_planned

# Strategy for 'update' actions: two distinct texts so a diff is produced
update_strategy = st.builds(
    PlannedWrite,
    destination=st.just(Path("/tmp/test_file.md")),
    content=st.text(min_size=1, max_size=100),
    action=st.just("update"),
    existing_content=st.text(min_size=1, max_size=100),
).filter(lambda pw: pw.content != pw.existing_content)

# Strategy for 'create' actions: any non-empty content
create_strategy = st.builds(
    PlannedWrite,
    destination=st.just(Path("/tmp/new_file.md")),
    content=st.text(min_size=1, max_size=100),
    action=st.just("create"),
    existing_content=st.none(),
)


@settings(max_examples=100)
@given(planned_write=update_strategy)
def test_dry_run_update_produces_unified_diff(planned_write: PlannedWrite) -> None:
    """For 'update' actions, dry-run output must contain unified diff markers."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        results = write_planned([planned_write], dry_run=True)
    output = captured.getvalue()
    assert "---" in output, f"Expected '---' marker in diff output but got:\n{output}"
    assert "+++" in output, f"Expected '+++' marker in diff output but got:\n{output}"
    assert len(results) == 1
    assert results[0].action == "update"
    assert results[0].success is True


@settings(max_examples=100)
@given(planned_write=create_strategy)
def test_dry_run_create_produces_labeled_content(planned_write: PlannedWrite) -> None:
    """For 'create' actions, dry-run output must show [create] label and full content."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        results = write_planned([planned_write], dry_run=True)
    output = captured.getvalue()
    assert "[create]" in output, f"Expected '[create]' label but got:\n{output}"
    assert planned_write.content in output, f"Expected content in output but got:\n{output}"
    assert len(results) == 1
    assert results[0].action == "create"
    assert results[0].success is True
