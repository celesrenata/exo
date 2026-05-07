# Node Communication Fix — Bugfix Design

## Overview

The 4-node gremlin cluster fails to form a gossipsub mesh because the discovery behaviour's `NetworkBehaviour` implementation uses `dummy::ConnectionHandler` in its connection handler methods. When composed with gossipsub via `#[derive(NetworkBehaviour)]` at the swarm level, the `V1Lazy` (optimistic 0-RTT) protocol negotiation sends gossipsub data before the remote confirms protocol support. Since the dummy handler advertises no protocols, the remote closes the stream with "Confirmation from remote for optimistic protocol negotiation still pending." The fix replaces `dummy::ConnectionHandler` with the existing `keep_alive::ConnectionHandler` (which already wraps dummy but forces `connection_keep_alive() -> true`) and switches the transport from `V1Lazy` to `V1` to eliminate optimistic negotiation failures.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug — gossipsub substream negotiation fails when the discovery behaviour's handler slot uses `dummy::ConnectionHandler` under `V1Lazy` transport
- **Property (P)**: The desired behavior — gossipsub substreams negotiate successfully and connections remain stable
- **Preservation**: Existing mDNS discovery, ping-based disconnection, static peer retry, connection event emission, and election flap detection must remain unchanged
- **`discovery::Behaviour`**: The custom `NetworkBehaviour` in `rust/networking/src/discovery.rs` that composes mDNS + ping and manages peer lifecycle
- **`keep_alive::ConnectionHandler`**: A wrapper around `dummy::ConnectionHandler` in `rust/networking/src/keep_alive.rs` that overrides `connection_keep_alive() -> true`
- **`V1Lazy`**: libp2p protocol negotiation version that sends data optimistically before receiving protocol confirmation (0-RTT)
- **`V1`**: libp2p protocol negotiation version that waits for protocol confirmation before sending data
- **Composed Behaviour**: The swarm-level `Behaviour` in `swarm.rs` that derives `NetworkBehaviour` over `discovery::Behaviour` + `gossipsub::Behaviour`

## Bug Details

### Bug Condition

The bug manifests when two nodes attempt gossipsub protocol negotiation over a connection where the discovery behaviour's handler slot uses `dummy::ConnectionHandler`. Under `V1Lazy` transport, the initiating side sends gossipsub protocol data optimistically (before the remote confirms the protocol). The composed handler on the remote side routes the inbound substream to the discovery handler slot first (based on `ConnectionHandlerSelect` ordering), which advertises no protocols via `dummy::ConnectionHandler`. The remote cannot confirm the protocol, so it closes the stream.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type LibP2PConnectionAttempt
  OUTPUT: boolean

  RETURN input.transport_version = V1Lazy
     AND input.discovery_handler = dummy::ConnectionHandler
     AND input.gossipsub_substream_requested = true
     AND input.remote_protocol_confirmation_pending = true
END FUNCTION
```

### Examples

- **Example 1**: gremlin-1 dials gremlin-2 on TCP 4001. Transport negotiates (noise + yamux). Gossipsub opens a substream. V1Lazy sends `/meshsub/1.1.0` data immediately. Remote's discovery handler slot (dummy) can't confirm → "Stream closed" error. Connection flaps.
- **Example 2**: gremlin-3 receives inbound connection from gremlin-4. Same V1Lazy + dummy handler interaction causes gossipsub substream failure within 5 seconds of connection establishment.
- **Example 3**: After 180 "Stream closed" messages in 3 minutes, the Python election code sees connect+disconnect pairs in the same 200ms batch and discards them as flaps. Cluster stays at 1 node.
- **Edge case**: If only mDNS/ping traffic flows (no gossipsub substreams), connections remain stable — the bug only triggers when gossipsub attempts substream negotiation.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- mDNS peer discovery and expiry tracking in `mdns_discovered` HashMap must continue to work exactly as before
- Retry timer must continue to re-dial only disconnected peers from both `mdns_discovered` and `static_peers` (skipping already-connected peers)
- `Event::ConnectionEstablished` and `Event::ConnectionClosed` must continue to emit with correct peer_id, connection_id, remote_ip, and remote_tcp_port
- Ping failure must continue to trigger immediate connection closure via `close_connection`
- Static peer promotion from `pending_static_addrs` on connection establishment must remain unchanged
- The private network (pnet) pre-shared key derivation must remain unchanged
- The Python election code's flap detection and 5-second grace period logic must remain unchanged

**Scope:**
All inputs that do NOT involve gossipsub substream negotiation should be completely unaffected by this fix. This includes:
- mDNS discover/expire events
- Ping request/response cycles
- Static peer dial and retry logic
- Connection lifecycle event emission
- Election message routing over already-established gossipsub channels

## Hypothesized Root Cause

Based on the bug description and code analysis, the most likely issues are:

1. **`dummy::ConnectionHandler` in discovery's `handle_established_*_connection`**: In `discovery.rs` lines ~270-295, both `handle_established_inbound_connection` and `handle_established_outbound_connection` return `ConnectionHandler::select(dummy::ConnectionHandler, self.managed.handle_established_*_connection(...))`. The `dummy::ConnectionHandler` advertises no protocols. When the composed swarm behaviour (`discovery` + `gossipsub`) creates a `ConnectionHandlerSelect` at the swarm level, the discovery slot's dummy handler interferes with protocol negotiation ordering under V1Lazy.

2. **`V1Lazy` optimistic negotiation**: In `swarm.rs` line ~88, `let upgrade_version = Version::V1Lazy` causes the initiator to send protocol data before receiving confirmation. With a handler that supports no protocols in the first slot of the composed handler, the remote side cannot confirm before the optimistic data arrives, causing stream closure.

3. **Missing keep-alive on discovery handler**: The `keep_alive::ConnectionHandler` exists in `rust/networking/src/keep_alive.rs` and is imported in `discovery.rs` (`use crate::keep_alive`), but is never actually used. Without keep-alive on the discovery handler slot, connections may be prematurely closed if the managed sub-behaviour (mDNS + ping) doesn't generate traffic quickly enough.

4. **Handler composition ordering**: `ConnectionHandlerSelect<dummy::ConnectionHandler, THandler<managed::Behaviour>>` places the dummy handler as the "left" handler. When the swarm composes this with gossipsub's handler, the protocol negotiation may route through the wrong slot under V1Lazy's optimistic path.

## Correctness Properties

Property 1: Bug Condition - Gossipsub Substreams Succeed After Fix

_For any_ connection attempt where the bug condition holds (gossipsub substream requested over a connection with the discovery behaviour's handler slot), the fixed code SHALL successfully negotiate gossipsub substreams without "Stream closed" errors, and the connection SHALL remain stable (no rapid connect/disconnect cycles within seconds of establishment).

**Validates: Requirements 2.1, 2.2**

Property 2: Preservation - mDNS and Connection Lifecycle Unchanged

_For any_ input that does NOT involve gossipsub substream negotiation (mDNS discover/expire events, ping cycles, static peer retry, connection event emission), the fixed code SHALL produce exactly the same behavior as the original code, preserving all existing peer lifecycle management, event emission, and retry logic.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `rust/networking/src/discovery.rs`

**Function**: `handle_established_inbound_connection` and `handle_established_outbound_connection`

**Specific Changes**:
1. **Replace `dummy::ConnectionHandler` with `keep_alive::ConnectionHandler::new()`**: In both `handle_established_inbound_connection` and `handle_established_outbound_connection`, change:
   ```rust
   // Before
   Ok(ConnectionHandler::select(
       dummy::ConnectionHandler,
       self.managed.handle_established_inbound_connection(...)?,
   ))
   // After
   Ok(ConnectionHandler::select(
       keep_alive::ConnectionHandler::new(),
       self.managed.handle_established_inbound_connection(...)?,
   ))
   ```

2. **Update the `ConnectionHandler` type alias**: Change the associated type from:
   ```rust
   type ConnectionHandler = ConnectionHandlerSelect<dummy::ConnectionHandler, THandler<managed::Behaviour>>;
   ```
   to:
   ```rust
   type ConnectionHandler = ConnectionHandlerSelect<keep_alive::ConnectionHandler, THandler<managed::Behaviour>>;
   ```

3. **Update `on_connection_handler_event` match arm**: The `Either::Left` arm currently calls `unreachable()` on the dummy handler's void event. With `keep_alive::ConnectionHandler` (which also has a void `ToBehaviour` type), this remains the same — no change needed here.

**File**: `rust/networking/src/swarm.rs`

**Function**: `tcp_transport`

**Specific Changes**:
4. **Switch from `V1Lazy` to `V1`**: Change:
   ```rust
   let upgrade_version = Version::V1Lazy;
   ```
   to:
   ```rust
   let upgrade_version = Version::V1;
   ```
   This eliminates optimistic protocol negotiation entirely, ensuring the remote always confirms protocol support before data is sent. The latency cost is one additional round-trip per connection establishment, which is negligible for a 4-node LAN cluster.

5. **Update the comment**: Change the comment from `// V1 + lazy flushing => 0-RTT negotiation` to reflect the new semantics:
   ```rust
   // V1 => wait for protocol confirmation before sending data (reliable negotiation)
   ```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write integration tests that establish connections between two libp2p swarms (using the current `dummy::ConnectionHandler` + `V1Lazy` configuration) and attempt gossipsub substream negotiation. Run these tests on the UNFIXED code to observe failures.

**Test Cases**:
1. **Gossipsub Subscribe Test**: Create two swarms with current config, connect them, subscribe to a topic on both, and assert that subscription messages propagate (will fail on unfixed code — substream closes before subscription exchange)
2. **Gossipsub Publish Test**: Connect two swarms, subscribe to a topic, publish a message, and assert delivery (will fail on unfixed code — no stable substream for message delivery)
3. **Connection Stability Test**: Connect two swarms and monitor for connection close events within 30 seconds (will fail on unfixed code — rapid disconnect cycles)
4. **Multi-Peer Mesh Test**: Connect 4 swarms in a mesh and verify all peers appear in gossipsub's mesh (will fail on unfixed code — no peer joins mesh)

**Expected Counterexamples**:
- Gossipsub substreams fail to establish (no topic subscriptions propagate)
- Connections enter rapid connect/disconnect cycles
- Possible causes: dummy handler can't confirm protocol, V1Lazy sends data before confirmation

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := negotiate_gossipsub_substream_fixed(input)
  ASSERT result.stream_established = true
  ASSERT result.no_stream_closed_error = true
  ASSERT result.connection_stable_after_30s = true
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT discovery_behaviour_original(input) = discovery_behaviour_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many random peer configurations and lifecycle sequences
- It catches edge cases in mDNS discover/expire ordering that manual tests miss
- It provides strong guarantees that the retry loop, event emission, and connection tracking are unchanged
- The existing proptest tests in `discovery.rs` already validate these properties

**Test Plan**: The existing `preservation_tests` module in `discovery.rs` already captures the correct mDNS/connection lifecycle behavior with proptest. These tests must PASS both before and after the fix. Additional preservation tests should verify that the `keep_alive::ConnectionHandler` replacement doesn't alter event emission or dial behavior.

**Test Cases**:
1. **mDNS Discover/Expire Consistency**: Verify `mdns_discovered` state remains consistent after random discover/expire sequences (existing proptest — must continue passing)
2. **Retry Loop Targets**: Verify retry loop dials exactly `mdns_discovered` + `static_peers` minus `connected_peers` (existing proptest — must continue passing)
3. **Connection Event Emission**: Verify `ConnectionEstablished`/`ConnectionClosed` events emit correctly for any peer (existing proptest — must continue passing)
4. **Ping Failure Closure**: Verify `close_connection` targets the specific failed peer/connection (existing proptest — must continue passing)
5. **Static Peer Promotion**: Verify peers from `pending_static_addrs` are promoted to `static_peers` on connection (new test)

### Unit Tests

- Test that `keep_alive::ConnectionHandler::new()` returns `connection_keep_alive() == true`
- Test that the discovery behaviour's `handle_established_inbound_connection` returns a handler with keep-alive semantics
- Test that `on_connection_handler_event` still handles `Either::Left` correctly with the new handler type
- Test that switching from `V1Lazy` to `V1` doesn't affect transport construction (tcp_transport still returns a valid boxed transport)

### Property-Based Tests

- Generate random sequences of mDNS discover/expire/connect/disconnect events and verify `mdns_discovered` + `connected_peers` state consistency (existing proptests)
- Generate random peer configurations and verify retry loop dial targets match expected set (existing proptests)
- Generate random connection lifecycle events and verify event emission correctness (existing proptests)
- Generate random static peer addresses and verify promotion logic on connection establishment

### Integration Tests

- Deploy to 4 gremlin nodes and verify `/state` API shows 4 nodes with gossipsub topic subscriptions
- Verify no "Stream closed" messages in journalctl logs after 5 minutes of operation
- Verify election completes successfully (master elected, all workers join session)
- Verify LACP bond failover (pull one cable) causes graceful reconnection, not permanent cluster split
- Verify log output shows stable gossipsub mesh formation: topic subscriptions for GLOBAL_EVENTS, LOCAL_EVENTS, COMMANDS, ELECTION_MESSAGES, CONNECTION_MESSAGES
