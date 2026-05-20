"""Property-based tests for telemetry parsing functions.

Feature: dashboard-inference-controls
"""

import json
from datetime import datetime, timedelta, timezone

from hypothesis import assume, given, settings, strategies as st

from exo.shared.types.common import NodeId
from exo.telemetry.aggregator import build_cluster_telemetry
from exo.telemetry.models import GpuMetrics, NetworkMetrics, NodeTelemetry
from exo.telemetry.parsers import (
    STALENESS_THRESHOLD_SECONDS,
    compute_throughput,
    find_cluster_interface_by_ip,
    is_cluster_interface,
    is_metric_stale,
    parse_intel_gpu_top,
    parse_proc_net_dev,
)


# Feature: dashboard-inference-controls, Property 5: intel_gpu_top JSON parsing correctness

# Strategy for busy percentage values [0, 100]
_busy_percent_strategy = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Strategy for the engines dict: always includes "Render/3D", optionally "Memory"
_engines_strategy = st.builds(
    lambda render_busy, memory_busy, include_memory: {
        "Render/3D": {"busy": render_busy},
        **({"Memory": {"busy": memory_busy}} if include_memory else {}),
    },
    render_busy=_busy_percent_strategy,
    memory_busy=_busy_percent_strategy,
    include_memory=st.booleans(),
)

# Strategy for frequency dict: non-negative integers
_frequency_strategy = st.fixed_dictionaries(
    {"actual": st.integers(min_value=0, max_value=10_000), "requested": st.integers(min_value=0, max_value=10_000)}
)

# Strategy for rc6 dict: idle percentage in [0, 100]
_rc6_strategy = st.fixed_dictionaries({"value": st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)})

# Strategy for a complete valid intel_gpu_top JSON string
_intel_gpu_top_json_strategy = st.builds(
    lambda engines, frequency, rc6: json.dumps({"engines": engines, "frequency": frequency, "rc6": rc6}),
    engines=_engines_strategy,
    frequency=_frequency_strategy,
    rc6=_rc6_strategy,
)


@settings(max_examples=200)
@given(json_output=_intel_gpu_top_json_strategy)
def test_intel_gpu_top_parsing_correctness(json_output: str) -> None:
    """Property 5: intel_gpu_top JSON parsing correctness.

    For any valid intel_gpu_top -J JSON output containing engine and frequency data,
    the parser SHALL extract frequency_mhz as a non-negative integer,
    utilization_percent as a float in [0, 100], render_busy_percent as a float
    in [0, 100], and memory_bandwidth_percent as None or in [0, 100].

    **Validates: Requirements 6.4, 6.8**
    """
    result = parse_intel_gpu_top(json_output)
    assert result is not None, "Parser must return a result for valid input"

    # frequency_mhz is a non-negative integer
    assert isinstance(result.frequency_mhz, int)
    assert result.frequency_mhz >= 0

    # utilization_percent is in [0, 100]
    assert 0.0 <= result.utilization_percent <= 100.0

    # render_busy_percent is in [0, 100]
    assert 0.0 <= result.render_busy_percent <= 100.0

    # memory_bandwidth_percent is None or in [0, 100]
    if result.memory_bandwidth_percent is not None:
        assert 0.0 <= result.memory_bandwidth_percent <= 100.0


# Feature: dashboard-inference-controls, Property 6: /proc/net/dev parsing correctness

# Strategy for valid Linux interface names: lowercase alpha start, alphanumeric, 1-15 chars
interface_name_strategy = st.from_regex(r"[a-z][a-z0-9]{0,14}", fullmatch=True)

# Strategy for a single row of 16 non-negative integer columns
columns_strategy = st.lists(st.integers(min_value=0, max_value=2**48), min_size=16, max_size=16)

# Strategy for a single data entry: (interface_name, columns)
entry_strategy = st.tuples(interface_name_strategy, columns_strategy)

# Strategy for 1-5 data lines
entries_strategy = st.lists(entry_strategy, min_size=1, max_size=5)


HEADER_LINE_1 = "Inter-|   Receive                                                |  Transmit"
HEADER_LINE_2 = " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed"


@settings(max_examples=200)
@given(entries=entries_strategy)
def test_proc_net_dev_parsing_correctness(entries: list[tuple[str, list[int]]]) -> None:
    """Property 6: /proc/net/dev parsing correctness.

    For any valid /proc/net/dev line containing an interface name and numeric columns,
    the parser SHALL extract bytes_received (column 1) and bytes_sent (column 9) as
    non-negative integers.

    **Validates: Requirements 7.2**
    """
    # Build valid /proc/net/dev content
    content_lines = [HEADER_LINE_1, HEADER_LINE_2]
    for interface_name, columns in entries:
        numbers_str = " ".join(str(c) for c in columns)
        content_lines.append(f"  {interface_name}: {numbers_str}")
    content = "\n".join(content_lines)

    # Parse
    parsed = parse_proc_net_dev(content)

    # Verify count matches
    assert len(parsed) == len(entries)

    # Verify each entry
    for parsed_entry, (expected_name, expected_columns) in zip(parsed, entries):
        assert parsed_entry.interface_name == expected_name
        assert parsed_entry.bytes_received == expected_columns[0]
        assert parsed_entry.bytes_sent == expected_columns[8]
        assert parsed_entry.bytes_received >= 0
        assert parsed_entry.bytes_sent >= 0


# Feature: dashboard-inference-controls, Property 7: Network throughput computation


@settings(max_examples=200)
@given(
    bytes_previous=st.integers(min_value=0, max_value=2**48),
    bytes_delta=st.integers(min_value=0, max_value=2**48),
    time_delta_seconds=st.floats(
        min_value=0.001,
        max_value=3600.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_network_throughput_computation(
    bytes_previous: int, bytes_delta: int, time_delta_seconds: float
) -> None:
    """Property 7: Network throughput computation.

    For any two consecutive network samples where the second has byte counts >= the first
    and a positive time delta, the computed throughput SHALL equal (bytes_delta / time_delta)
    and be non-negative.

    **Validates: Requirements 7.3**
    """
    bytes_current = bytes_previous + bytes_delta

    result = compute_throughput(bytes_current, bytes_previous, time_delta_seconds)

    expected = float(bytes_delta) / time_delta_seconds
    assert result == expected
    assert result >= 0.0


# Feature: dashboard-inference-controls, Property 8: Interface name filtering

# Strategy for excluded interface names
_excluded_suffix = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
    min_size=0,
    max_size=8,
)

_excluded_name_strategy = st.one_of(
    st.just("lo"),
    _excluded_suffix.map(lambda s: f"docker{s}"),
    _excluded_suffix.map(lambda s: f"veth{s}"),
    _excluded_suffix.map(lambda s: f"br-{s}"),
    _excluded_suffix.map(lambda s: f"virbr{s}"),
)

# Strategy for valid interface names (common Linux interface names)
_valid_name_strategy = st.sampled_from(
    ["eth0", "eth1", "eno1", "eno2", "enp3s0", "enp0s25", "wlan0", "wlp2s0", "ens192", "bond0"]
)

# Strategy for IPs in the 10.1.1.0/24 subnet (host part 1-254)
_ip_in_subnet_strategy = st.integers(min_value=1, max_value=254).map(lambda h: f"10.1.1.{h}")

# Strategy for IPs outside the 10.1.1.0/24 subnet
_ip_out_subnet_strategy = st.sampled_from(
    ["192.168.1.1", "10.0.0.1", "172.16.0.1", "10.1.2.1", "10.2.1.1", "8.8.8.8"]
)


@settings(max_examples=200)
@given(
    excluded_name=_excluded_name_strategy,
    valid_name=_valid_name_strategy,
    ip_in_subnet=_ip_in_subnet_strategy,
    ip_out_subnet=_ip_out_subnet_strategy,
)
def test_interface_name_filtering(
    excluded_name: str,
    valid_name: str,
    ip_in_subnet: str,
    ip_out_subnet: str,
) -> None:
    """Property 8: Interface name filtering.

    For any interface name, the cluster interface filter SHALL exclude names matching
    lo, docker*, veth*, br-*, and virbr*, and SHALL include interfaces whose IP address
    falls within the 10.1.1.0/24 subnet.

    **Validates: Requirements 7.5**
    """
    # Excluded names must be rejected by is_cluster_interface
    assert not is_cluster_interface(excluded_name)

    # Valid names must be accepted by is_cluster_interface
    assert is_cluster_interface(valid_name)

    # Excluded name with in-subnet IP should NOT be found
    excluded_interfaces: dict[str, list[str]] = {excluded_name: [ip_in_subnet]}
    assert find_cluster_interface_by_ip(excluded_interfaces) is None

    # Valid name with in-subnet IP should be found
    valid_interfaces: dict[str, list[str]] = {valid_name: [ip_in_subnet]}
    assert find_cluster_interface_by_ip(valid_interfaces) == valid_name

    # Valid name with out-of-subnet IP should NOT be found
    out_subnet_interfaces: dict[str, list[str]] = {valid_name: [ip_out_subnet]}
    assert find_cluster_interface_by_ip(out_subnet_interfaces) is None


# Feature: dashboard-inference-controls, Property 9: Metric staleness detection


@settings(max_examples=200)
@given(
    base_timestamp=st.floats(min_value=0, max_value=1_000_000_000, allow_infinity=False, allow_nan=False),
    delta_seconds=st.floats(min_value=0, max_value=10.0, allow_infinity=False, allow_nan=False),
)
def test_metric_staleness_detection(base_timestamp: float, delta_seconds: float) -> None:
    """Property 9: Metric staleness detection.

    For any metric timestamp and current UTC time, the metric SHALL be marked stale
    if and only if (current_time - timestamp) > 3 seconds.

    **Validates: Requirements 8.3**
    """
    metric_timestamp = datetime.fromtimestamp(base_timestamp, tz=timezone.utc)
    current_time = metric_timestamp + timedelta(seconds=delta_seconds)

    is_stale = is_metric_stale(metric_timestamp, current_time)
    expected_stale = delta_seconds > STALENESS_THRESHOLD_SECONDS

    assert is_stale == expected_stale, (
        f"Expected stale={expected_stale} but got {is_stale} for delta={delta_seconds}"
    )

    # Biconditional check: reverse direction
    if is_stale:
        assert delta_seconds > STALENESS_THRESHOLD_SECONDS
    else:
        assert delta_seconds <= STALENESS_THRESHOLD_SECONDS

# Feature: dashboard-inference-controls, Property 10: Offline node omission from telemetry response

_FIXED_TIMESTAMP = datetime(2025, 1, 1, tzinfo=timezone.utc)

_node_id_strategy = st.uuids().map(lambda u: NodeId(str(u)))


@st.composite
def _reporting_and_known_nodes_strategy(draw):  # type: ignore[no-untyped-def]
    """Generate a set of reporting nodes and a superset of all known node IDs."""
    all_node_ids = draw(st.lists(_node_id_strategy, min_size=1, max_size=8, unique=True))
    reporting_count = draw(st.integers(min_value=0, max_value=len(all_node_ids)))
    reporting_ids = all_node_ids[:reporting_count]
    reporting_nodes: dict[NodeId, NodeTelemetry] = {}
    for node_id in reporting_ids:
        gpu = GpuMetrics(node_id=node_id, timestamp=_FIXED_TIMESTAMP)
        network = NetworkMetrics(
            node_id=node_id,
            timestamp=_FIXED_TIMESTAMP,
            interface_name="eth0",
            bytes_sent=0,
            bytes_received=0,
            throughput_sent_bytes_per_sec=0.0,
            throughput_received_bytes_per_sec=0.0,
        )
        telemetry = NodeTelemetry(node_id=node_id, gpu=gpu, network=network)
        reporting_nodes[node_id] = telemetry
    return reporting_nodes, all_node_ids


@settings(max_examples=200)
@given(data=_reporting_and_known_nodes_strategy())
def test_offline_node_omission_from_telemetry_response(
    data: tuple[dict[NodeId, NodeTelemetry], list[NodeId]],
) -> None:
    """Property 10: Offline node omission from telemetry response.

    For any cluster telemetry snapshot, the response SHALL contain entries only
    for nodes that have reported at least one metric. Nodes that have never
    reported or have been removed from topology SHALL be omitted entirely.

    **Validates: Requirements 8.5**
    """
    reporting_nodes, all_known_node_ids = data

    result = build_cluster_telemetry(reporting_nodes, all_known_node_ids)

    # The result nodes must be EXACTLY the reporting nodes
    assert set(result.nodes.keys()) == set(reporting_nodes.keys())

    # No offline node appears in the result
    offline_node_ids = set(all_known_node_ids) - set(reporting_nodes.keys())
    for offline_id in offline_node_ids:
        assert offline_id not in result.nodes

    # Every node in the result must be a reporting node
    for node_id in result.nodes:
        assert node_id in reporting_nodes
