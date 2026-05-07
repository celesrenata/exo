"""Property-based tests for transport layer fallback logic.

Uses Hypothesis to verify that detect_transport() falls back to Ethernet
when the preferred transport is unavailable (RDMA not present, LACP bond
not configured), and that the resulting TransportInfo always has
transport_type == "ethernet" in fallback scenarios.

**Validates: Requirements 7.9**
"""

from __future__ import annotations

import socket
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.worker.engines.pytorch.distributed.transport import (
    TransportInfo,
    detect_transport,
)


# --- Mock helpers ---

# A fake ethernet interface with a valid IPv4 address
_FAKE_ETHERNET_ADDRS: dict[str, list[Any]] = {
    "eth0": [
        type(
            "snicaddr",
            (),
            {
                "family": socket.AF_INET,
                "address": "10.1.1.100",
                "netmask": "255.255.255.0",
                "broadcast": "10.1.1.255",
                "ptp": None,
            },
        )()
    ],
}


@contextmanager
def _mock_no_rdma_no_lacp() -> Generator[None, None, None]:
    """Mock environment where RDMA and LACP are unavailable but ethernet works."""
    with (
        patch("psutil.net_if_addrs", return_value=_FAKE_ETHERNET_ADDRS),
        patch("os.path.exists", return_value=False),
    ):
        yield


@contextmanager
def _mock_no_rdma_with_ethernet() -> Generator[None, None, None]:
    """Mock environment where RDMA is unavailable but ethernet is present."""
    with (
        patch("psutil.net_if_addrs", return_value=_FAKE_ETHERNET_ADDRS),
        patch("os.path.exists", return_value=False),
    ):
        yield


@contextmanager
def _mock_no_lacp_with_ethernet() -> Generator[None, None, None]:
    """Mock environment where LACP bond is not configured but ethernet is present."""
    with (
        patch("psutil.net_if_addrs", return_value=_FAKE_ETHERNET_ADDRS),
        patch("os.path.exists", return_value=False),
    ):
        yield


# --- Strategies ---

# Transport types that can be requested
transport_type_st = st.sampled_from(["rdma", "lacp", "ethernet"])


class TestTransportFallbackProperty:
    """Property 7: Transport fallback defaults to Ethernet.

    *For any* transport type in {"rdma", "lacp", "ethernet"}, if the selected
    transport is detected as unavailable (interface not found, bond not
    configured, RDMA not present), the transport layer SHALL fall back to
    "ethernet" and the resulting TransportInfo.transport_type SHALL be
    "ethernet".

    **Validates: Requirements 7.9**
    """

    @settings(max_examples=100)
    @given(preferred=st.just("rdma"))
    def test_rdma_unavailable_falls_back_to_ethernet(
        self, preferred: str
    ) -> None:
        """When RDMA is unavailable and preferred is "rdma", result is "ethernet".

        Requirement 7.9: IF the selected transport is unavailable, THEN THE
        Transport_Layer SHALL fall back to Ethernet_Transport.
        """
        with _mock_no_rdma_with_ethernet():
            result = detect_transport(preferred)  # type: ignore[arg-type]

        assert isinstance(result, TransportInfo)
        assert result.transport_type == "ethernet", (
            f"Expected fallback to 'ethernet' when RDMA unavailable, "
            f"got '{result.transport_type}'"
        )

    @settings(max_examples=100)
    @given(preferred=st.just("lacp"))
    def test_lacp_unavailable_falls_back_to_ethernet(
        self, preferred: str
    ) -> None:
        """When LACP is unavailable and preferred is "lacp", result is "ethernet".

        Requirement 7.9: IF the selected transport is unavailable, THEN THE
        Transport_Layer SHALL fall back to Ethernet_Transport.
        """
        with _mock_no_lacp_with_ethernet():
            result = detect_transport(preferred)  # type: ignore[arg-type]

        assert isinstance(result, TransportInfo)
        assert result.transport_type == "ethernet", (
            f"Expected fallback to 'ethernet' when LACP unavailable, "
            f"got '{result.transport_type}'"
        )

    @settings(max_examples=100)
    @given(preferred=st.just("ethernet"))
    def test_ethernet_preferred_returns_ethernet(
        self, preferred: str
    ) -> None:
        """When ethernet is available, preferred "ethernet" returns "ethernet".

        When the preferred transport is ethernet and it's available, the
        result should be ethernet (no fallback needed).
        """
        with _mock_no_rdma_no_lacp():
            result = detect_transport(preferred)  # type: ignore[arg-type]

        assert isinstance(result, TransportInfo)
        assert result.transport_type == "ethernet", (
            f"Expected 'ethernet' when preferred is 'ethernet', "
            f"got '{result.transport_type}'"
        )

    @settings(max_examples=100)
    @given(preferred=transport_type_st)
    def test_fallback_always_produces_valid_transport_info(
        self, preferred: str
    ) -> None:
        """The fallback always produces a valid TransportInfo with transport_type == "ethernet".

        For any transport type, when the preferred transport is unavailable
        (no RDMA hardware, no LACP bond), the result SHALL be a valid
        TransportInfo with transport_type == "ethernet", a non-empty
        interface_name, a valid bind_address, and positive bandwidth.

        **Validates: Requirements 7.9**
        """
        with _mock_no_rdma_no_lacp():
            result = detect_transport(preferred)  # type: ignore[arg-type]

        assert isinstance(result, TransportInfo)
        assert result.transport_type == "ethernet", (
            f"Expected fallback to 'ethernet' for preferred='{preferred}', "
            f"got '{result.transport_type}'"
        )
        assert result.interface_name != "", (
            "TransportInfo.interface_name must not be empty"
        )
        assert result.bind_address != "", (
            "TransportInfo.bind_address must not be empty"
        )
        assert result.bandwidth_gbps > 0, (
            f"TransportInfo.bandwidth_gbps must be positive, got {result.bandwidth_gbps}"
        )
