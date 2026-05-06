# Bugfix Requirements Document

## Introduction

Peer connections on LACP-bonded gremlin nodes flap indefinitely, preventing stable cluster formation. The Rust libp2p retry loop dials static peers every 5 seconds, but on nodes with 802.3ad LACP bonding (layer3+4 hash policy), each new TCP connection uses a different ephemeral source port, which may route through a different bond slave. The connection completes the full pnet+noise+yamux handshake but is immediately torn down. Both `Connected` and `Disconnected` events arrive at the Python election layer within the same 200ms batch window, where the port-aware flap detection correctly identifies the flap and ignores it. The peer never registers as connected, and the cycle repeats every 5 seconds indefinitely. This prevents affected nodes (currently gremlin-3, but non-deterministic) from joining the exo cluster.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the Rust retry loop dials a static peer on an LACP-bonded node every 5 seconds THEN the system establishes a TCP connection that is immediately torn down, producing a Connected+Disconnected event pair on the same (node_id, port) within a single 200ms batch

1.2 WHEN the Python election layer receives a Connected+Disconnected pair for the same (node_id, port) in one batch THEN the system classifies it as a connection flap and discards both events, so the peer is never registered as connected

1.3 WHEN a peer repeatedly flaps every retry interval (5 seconds) without ever achieving a stable connection THEN the system has no escalation mechanism and continues the same failing strategy indefinitely, leaving the peer permanently unable to join the cluster

1.4 WHEN the Rust retry loop fires and the peer is not in `connected_peers` THEN the system dials using a new ephemeral source port each time, which on LACP layer3+4 hash may route through a different bond slave than the previous attempt, contributing to connection instability

### Expected Behavior (Correct)

2.1 WHEN the Rust retry loop dials a static peer on an LACP-bonded node THEN the system SHALL establish a TCP connection that persists beyond the handshake phase, using TCP keepalive to maintain the connection through brief LACP slave transitions

2.2 WHEN the Python election layer detects repeated connection flaps for the same peer across multiple consecutive retry intervals THEN the system SHALL escalate by treating the peer as requiring connection stabilization rather than silently discarding all events indefinitely

2.3 WHEN a peer has flapped N consecutive times without achieving a stable connection THEN the system SHALL attempt an alternative connection strategy (such as accepting an inbound connection from the flapping peer, or increasing the batch window for that peer) to break the flap cycle

2.4 WHEN the Rust retry loop dials a static peer that is already being dialed (connection in progress) THEN the system SHALL NOT initiate a duplicate dial attempt, preventing simultaneous-connection races that contribute to immediate teardown

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a peer is discovered via mDNS THEN the system SHALL CONTINUE TO track it in `mdns_discovered`, dial it on discovery, and remove it on expiry

3.2 WHEN a single legitimate connection flap occurs (e.g., brief network blip) THEN the system SHALL CONTINUE TO ignore it via the existing port-aware flap detection, avoiding unnecessary re-elections

3.3 WHEN a connection is established or closed (regardless of origin) THEN the system SHALL CONTINUE TO emit `ConnectionEstablished` and `ConnectionClosed` events to the swarm and track connection counts in `connected_peers`

3.4 WHEN the current master disconnects and does not reconnect within the grace period THEN the system SHALL CONTINUE TO trigger a re-election after the 5-second grace period

3.5 WHEN a ping to any connected peer fails THEN the system SHALL CONTINUE TO close that specific connection

---

### Bug Condition (Formal)

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type ConnectionAttempt
  OUTPUT: boolean

  // Returns true when the connection attempt targets a static peer
  // on an LACP-bonded node where the connection is torn down within
  // the batch window (200ms), producing a flap every retry interval
  RETURN X.peer_id IN static_peers
    AND X.connection_lifetime < BATCH_WINDOW (200ms)
    AND X.consecutive_flap_count > 0
END FUNCTION
```

### Fix Checking Property

```pascal
// Property: Peers that repeatedly flap eventually achieve stable connection
FOR ALL X WHERE isBugCondition(X) DO
  result ← connection_strategy_after_N_flaps(X)
  ASSERT result.peer_eventually_connected = TRUE
    OR result.escalation_triggered = TRUE
  ASSERT result.infinite_flap_loop = FALSE
END FOR
```

### Preservation Checking Property

```pascal
// Property: Non-flapping peers and single-flap events behave identically
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT F(X) = F'(X)
END FOR
```

Where:
- **F**: The original system (ignores all flaps unconditionally, retries with new port every 5s)
- **F'**: The fixed system (detects persistent flaps and escalates connection strategy; uses TCP keepalive; prevents duplicate dials)
