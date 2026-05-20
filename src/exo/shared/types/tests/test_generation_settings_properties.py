# Feature: dashboard-inference-controls, Property 2: Event apply correctness for GenerationSettingsUpdated
"""Property-based tests for GenerationSettingsUpdated event application.

Validates: Requirements 1.2
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.shared.apply import event_apply
from exo.shared.types.events import GenerationSettingsUpdated
from exo.shared.types.generation_settings import GenerationSettings
from exo.shared.types.state import State

# Strategy for valid GenerationSettings
generation_settings_strategy = st.builds(
    GenerationSettings,
    thinking_mode=st.booleans(),
    thinking_token_budget=st.one_of(st.none(), st.integers(min_value=1, max_value=1_048_576)),
    output_token_budget=st.one_of(st.none(), st.integers(min_value=1, max_value=1_048_576)),
)


@settings(max_examples=200)
@given(new_settings=generation_settings_strategy)
def test_event_apply_updates_generation_settings_and_preserves_other_fields(
    new_settings: GenerationSettings,
) -> None:
    """Applying GenerationSettingsUpdated produces a state where generation_settings
    equals the event's generation_settings, and all other fields remain unchanged.

    **Validates: Requirements 1.2**
    """
    state = State()
    event = GenerationSettingsUpdated(generation_settings=new_settings)

    result = event_apply(event, state)

    # The generation_settings field must equal the event's value
    assert result.generation_settings == new_settings

    # All other fields must remain unchanged
    assert result.instances == state.instances
    assert result.runners == state.runners
    assert result.downloads == state.downloads
    assert result.tasks == state.tasks
    assert result.last_seen == state.last_seen
    assert result.topology.to_snapshot() == state.topology.to_snapshot()
    assert result.last_event_applied_idx == state.last_event_applied_idx
    assert result.node_identities == state.node_identities
    assert result.node_memory == state.node_memory
    assert result.node_disk == state.node_disk
    assert result.node_system == state.node_system
    assert result.node_network == state.node_network
    assert result.node_thunderbolt == state.node_thunderbolt
    assert result.node_thunderbolt_bridge == state.node_thunderbolt_bridge
    assert result.node_rdma_ctl == state.node_rdma_ctl
    assert result.thunderbolt_bridge_cycles == state.thunderbolt_bridge_cycles
    assert result.instance_links == state.instance_links
    assert result.prefill_server_ports == state.prefill_server_ports


# Feature: dashboard-inference-controls, Property 3: Partial patch merge preserves unpatched fields

from exo.api.generation_settings import GenerationSettingsPatch, _merge_settings


@st.composite
def partial_patch_strategy(draw: st.DrawFn) -> GenerationSettingsPatch:
    """Build a GenerationSettingsPatch with a random subset of fields set."""
    include_thinking_mode = draw(st.booleans())
    include_thinking_token_budget = draw(st.booleans())
    include_output_token_budget = draw(st.booleans())

    kwargs: dict[str, object] = {}
    if include_thinking_mode:
        kwargs["thinking_mode"] = draw(st.booleans())
    if include_thinking_token_budget:
        kwargs["thinking_token_budget"] = draw(
            st.one_of(st.none(), st.integers(min_value=1, max_value=1_048_576))
        )
    if include_output_token_budget:
        kwargs["output_token_budget"] = draw(
            st.one_of(st.none(), st.integers(min_value=1, max_value=1_048_576))
        )

    return GenerationSettingsPatch(**kwargs)


@settings(max_examples=200)
@given(
    current=generation_settings_strategy,
    patch=partial_patch_strategy(),
)
def test_partial_patch_merge_preserves_unpatched_fields(
    current: GenerationSettings,
    patch: GenerationSettingsPatch,
) -> None:
    """Merging a partial patch into current settings updates only patched fields
    and leaves unpatched fields equal to the original values.

    **Validates: Requirements 2.2**
    """
    merged = _merge_settings(current, patch)

    patched_fields = patch.model_dump(exclude_unset=True).keys()

    # Patched fields must equal the patch values
    for field in patched_fields:
        assert getattr(merged, field) == getattr(patch, field), (
            f"Patched field '{field}': expected {getattr(patch, field)}, got {getattr(merged, field)}"
        )

    # Unpatched fields must equal the original values
    unpatched_fields = set(GenerationSettings.model_fields.keys()) - patched_fields
    for field in unpatched_fields:
        assert getattr(merged, field) == getattr(current, field), (
            f"Unpatched field '{field}': expected {getattr(current, field)}, got {getattr(merged, field)}"
        )


# Feature: dashboard-inference-controls, Property 4: Settings-to-inference parameter resolution

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.runner.runner import resolve_generation_settings

# Strategy for TextGenerationTaskParams with None optional fields (cluster defaults apply)
params_with_none_strategy = st.builds(
    TextGenerationTaskParams,
    model=st.just("test/model"),
    input=st.just([]),
    enable_thinking=st.none(),
    max_output_tokens=st.none(),
)

# Strategy for TextGenerationTaskParams with explicit (non-None) optional fields
params_with_explicit_strategy = st.builds(
    TextGenerationTaskParams,
    model=st.just("test/model"),
    input=st.just([]),
    enable_thinking=st.booleans(),
    max_output_tokens=st.integers(min_value=1, max_value=1_048_576),
)


@settings(max_examples=200)
@given(cluster_settings=generation_settings_strategy, params=params_with_none_strategy)
def test_resolve_settings_applies_cluster_defaults_when_params_are_none(
    cluster_settings: GenerationSettings,
    params: TextGenerationTaskParams,
) -> None:
    """When params have None for enable_thinking and max_output_tokens,
    the resolved params should get values from cluster settings
    (thinking_mode -> enable_thinking, output_token_budget -> max_output_tokens).

    **Validates: Requirements 3.4, 3.5, 4.4, 4.5, 5.3, 5.4**
    """
    resolved = resolve_generation_settings(cluster_settings, params)

    assert resolved.enable_thinking == cluster_settings.thinking_mode
    assert resolved.max_output_tokens == cluster_settings.output_token_budget


@settings(max_examples=200)
@given(cluster_settings=generation_settings_strategy, params=params_with_explicit_strategy)
def test_resolve_settings_request_params_take_precedence(
    cluster_settings: GenerationSettings,
    params: TextGenerationTaskParams,
) -> None:
    """When params have explicit (non-None) values for enable_thinking and
    max_output_tokens, those values are preserved regardless of cluster settings.

    **Validates: Requirements 3.4, 3.5, 4.4, 4.5, 5.3, 5.4**
    """
    resolved = resolve_generation_settings(cluster_settings, params)

    assert resolved.enable_thinking == params.enable_thinking
    assert resolved.max_output_tokens == params.max_output_tokens
