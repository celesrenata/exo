# Stable Peer Connections Bugfix Design — LACP Bond Flapping

## Overview

Peer connections on LACP-bonded gremlin nodes (802.3ad, layer3+4 hash, fast LACP rate) flap indefinitely, preventing cluster formation. The libp2p retry loop dials static peers every 5 seconds using a new ephemeral source port each time. On LACP bonds with layer3+4 hash policy, the 4-tuple (src IP, dst IP, src port, dst port) determines which bond slave carries the flow. Each new ephemeral port produces a different hash, potentially routing through a different slave. The connection completes the full pnet+noise+yamux handshake but is immediately torn down within ~200ms. The Python election layer correctly identifies the Connected+Disconnected pair as a flap and discards it. The peer never registers as connected, and the cycle repeats every 5 seconds indefinitely.

The fix introduces three complementary mechanisms:
1. **Connection stabilization grace period** in the Rust discovery layer — suppress event emission until a connection has survived a configurable hold-down timer (default 2s), absorbing transient LACP-induced teardowns at the source
2. **Exponential backoff with jitter** on the retry loop for peers that repeatedly fail to establish stable connections, preventing rapid re-dial storms that exacerbate LACP hash instability
3. **TCP keepalive** on the transport to maintain established connections through brief LACP slave transitions and prevent idle-timeout teardowns

## Glossary

- **Bug_Condition (C)**: A static peer connection on an LACP-bonded node that is torn down within the stabilization grace period (< 2s), producing a flap that is never reported to the Python layer
- **Property (P)**: Peers that repeatedly flap eventually achieve a stable connection (or the system escalates), and the Python election layer receives only durable connection events
- **Preservation**: mDNS peer lifecycle, single-flap detection, ping-based disconnection, gossipsub routing, and connection event semantics for stable connections must remain unchanged
- **LACP (802.3ad)**: Link Aggregation Control Protocol — bonds multiple physical links into one logical interface for redundancy and bandwidth
- **layer3+4 hash**: LACP transmit hash policy using src/dst IP + src/dst port to select the outgoing slave interface
- **Bond slave transition**: When an LACP slave link goes up/down or the LACPDU negotiation changes the active slave set, in-flight TCP connections on the affected slave may be disrupted
- **Grace period / hold-down timer**: Duration a connection must survive before being reported as established to the Python layer
- **`Behaviour`**: The outer discovery behaviour in `rust/networking/src/discovery.rs` that wraps mDNS + ping and manages peer lifecycle
- **`connected_peers`**: Existing `HashMap<PeerId, u32>` tracking peers with at least one active connection
- **`static_peers`**: Existing `HashMap<PeerId, BTreeSet<Multiaddr>>` tracking manually-dialed peers for retry
- **V1Lazy (0-RTT)**: libp2p upgrade version that reports connection as established before full protocol negotiation completes

## Bug Details

### Bug Condition

The bug manifests when the Rust retry loop dials a static peer on an LACP-bonded node every 5 seconds. Each dial uses a new ephemeral source port, which changes the layer3+4 hash and may route through a different bond slave. The connection completes the pnet+noise+yamux handshake (aided by V1Lazy 0-RTT) but is immediately torn down — likely because:
- The LACP bond has 0ms up/down delay, so slave transitions are instantaneous
- The new flow's hash lands on a slave that is mid-transition or about to transition
- The MII polling interval (100ms) detects the issue and tears down the connection before it stabilizes

The result is that `ConnectionEstablished` and `ConnectionClosed` events arrive at the Python layer within the same 200ms batch window. The port-aware flap detection correctly identifies this as a flap (same node+port connected and disconnected) and discards both events. The peer is never registered as connected.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type ConnectionAttempt
  OUTPUT: boolean

  RETURN input.peer_id IN static_peers
    AND input.node_has_lacp_bond = TRUE
    AND input.connection_lifetime < STABILIZATION_GRACE_PERIOD (2 seconds)
    AND input.consecutive_unstable_attempts > 0
END FUNCTION
```

### Examples

- **Typical flap cycle**: Retry loop dials gremlin-3 at `10.1.1.14:5678` using ephemeral port 49152. LACP hash routes through slave `eth0`. Connection completes handshake in 50ms. At 150ms, the bond's MII polling detects a slave transition on `eth0`. Connection is torn down at 180ms. Python receives Connected+Disconnected in same batch, discards as flap. 5 seconds later, retry loop dials again with ephemeral port 49153 — different hash, routes through `eth1`. Same result.
- **Successful connection (non-LACP node)**: Retry loop dials gremlin-2 (no LACP bond). Connection establishes and persists indefinitely. `ConnectionEstablished` is emitted, Python registers the peer, election proceeds normally.
- **Eventual success after backoff**: After 3 consecutive flaps, the retry interval backs off to 20s. During this longer interval, the LACP bond stabilizes. The next dial attempt uses a port whose hash routes through a stable slave. Connection persists beyond the 2s grace period and is reported to Python.
- **Edge case — connection dies at 1.9s**: Connection survives 1.9 seconds (just under the 2s grace period). It is NOT reported to Python. The retry loop treats this as another unstable attempt and backs off further.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- mDNS-discovered peers must continue to be tracked, dialed on discovery, and removed on expiry — the `handle_mdns_discovered` and `handle_mdns_expired` logic is untouched
- Single legitimate connection flaps (brief network blip on non-LACP nodes) must continue to be ignored by the Python port-aware flap detection — the election module logic is untouched
- `ConnectionEstablished` and `ConnectionClosed` events must continue to fire for connections that survive the grace period — the event semantics are preserved for stable connections
- Ping-based disconnection must continue to close connections when pings fail
- The current master disconnect grace period (5 seconds before re-election) must remain unchanged
- Gossipsub message routing operates on connected peers regardless of discovery origin — unchanged

**Scope:**
All inputs that do NOT involve connections torn down within the stabilization grace period should be completely unaffected by this fix. This includes:
- Connections that survive beyond 2 seconds (reported normally)
- mDNS discovery and expiry events
- Ping success/failure handling
- Gossipsub subscribe/unsubscribe/publish operations
- Election protocol messages
- Connections on non-LACP nodes (which naturally survive the grace period)

## Hypothesized Root Cause

Based on the bug description and LACP bond configuration, the most likely causes are:

1. **Ephemeral port rotation destabilizes LACP hash**: Each retry dial uses a new ephemeral source port. With layer3+4 hash policy, this changes the hash value and may route the new flow through a different bond slave. If that slave is mid-LACPDU negotiation or about to transition, the connection is torn down immediately.

2. **Zero up/down delay amplifies instability**: The LACP bond has `up_delay=0ms` and `down_delay=0ms`. This means slave transitions are reported instantly to the bonding driver, with no dampening. A slave that flickers (e.g., brief carrier loss during LACPDU renegotiation) immediately affects all flows hashed to it.

3. **V1Lazy (0-RTT) reports connection too early**: `Version::V1Lazy` allows libp2p to report a connection as established before the full protocol negotiation is confirmed end-to-end. On LACP bonds, the connection may be reported as established while the underlying TCP socket is still on an unstable path.

4. **No TCP keepalive**: The TCP transport uses `Config::default().nodelay(true)` but does not enable TCP keepalive. Without keepalive probes, the OS has no mechanism to detect and recover from brief path disruptions caused by LACP slave transitions. Connections that survive the initial handshake may still be torn down by idle-timeout mechanisms.

5. **Fixed 5-second retry with no backoff**: The retry loop fires every 5 seconds regardless of how many times the connection has flapped. This creates a deterministic pattern that may consistently hit the same LACP timing window, especially with fast LACP rate (1-second LACPDU interval).

6. **No connection stabilization at the event emission layer**: The Rust `on_connection_established` method immediately emits `ConnectionEstablished` to the Python layer. There is no hold-down period to verify the connection is durable before reporting it. This means every transient connection (even those lasting < 200ms) generates events that the Python layer must filter.

## Correctness Properties

Property 1: Bug Condition — Unstable Connections Are Suppressed

_For any_ connection attempt where the connection is torn down within the stabilization grace period (isBugCondition returns true), the fixed discovery behaviour SHALL NOT emit `ConnectionEstablished` or `ConnectionClosed` events to the Python layer, and SHALL apply exponential backoff to subsequent retry attempts for that peer, eventually achieving a stable connection or reaching a maximum backoff ceiling.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Preservation — Stable Connections Report Normally

_For any_ connection that survives beyond the stabilization grace period (isBugCondition returns false), the fixed discovery behaviour SHALL emit `ConnectionEstablished` exactly once when the grace period expires, and SHALL emit `ConnectionClosed` when the connection is actually closed, preserving the existing event semantics for all stable connections and maintaining identical mDNS lifecycle, ping disconnection, and gossipsub behaviour.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `rust/networking/src/discovery.rs`

**Struct**: `Behaviour`

**Specific Changes**:

1. **Add `pending_connections` field**: Add a `HashMap<ConnectionId, PendingConnection>` where `PendingConnection` stores `peer_id`, `remote_ip`, `remote_tcp_port`, and a `Delay` (grace period timer). Connections enter this map on establishment and are promoted to "stable" (event emitted) when the timer expires.

   ```rust
   struct PendingConnection {
       peer_id: PeerId,
       remote_ip: IpAddr,
       remote_tcp_port: u16,
       grace_timer: Delay,
   }
   ```

2. **Add `peer_flap_counts` field**: Add a `HashMap<PeerId, u32>` tracking consecutive unstable connection attempts per peer. Reset to 0 when a connection survives the grace period. Used to compute exponential backoff.

3. **Add `peer_next_retry` field**: Add a `HashMap<PeerId, Instant>` tracking the earliest time a peer should be re-dialed. The retry loop skips peers whose `next_retry` is in the future.

4. **Modify `on_connection_established`**: Instead of immediately emitting `ConnectionEstablished`, insert the connection into `pending_connections` with a grace period timer (default: `STABILIZATION_GRACE_PERIOD = 2 seconds`). Still increment `connected_peers` (to prevent duplicate dials during the grace period).

5. **Modify `on_connection_closed`**: Check if the closed connection is in `pending_connections`. If yes, remove it from `pending_connections` WITHOUT emitting `ConnectionClosed` (the Python layer never knew about it). Increment `peer_flap_counts` for that peer and compute the next retry time with exponential backoff. If the connection is NOT in `pending_connections` (it was already promoted to stable), emit `ConnectionClosed` as before.

6. **Add grace period polling in `poll()`**: Before the retry timer check, poll all `pending_connections` grace timers. When a timer expires, remove the entry from `pending_connections`, reset `peer_flap_counts[peer_id]` to 0, clear `peer_next_retry[peer_id]`, and emit `ConnectionEstablished`.

7. **Modify retry loop to respect backoff**: In the retry timer block, before dialing a peer, check `peer_next_retry`. If `Instant::now() < peer_next_retry[peer_id]`, skip that peer for this cycle.

8. **Add constants**:
   ```rust
   const STABILIZATION_GRACE_PERIOD: Duration = Duration::from_secs(2);
   const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(60);
   const BACKOFF_BASE: Duration = Duration::from_secs(5);
   ```

   Backoff formula: `min(BACKOFF_BASE * 2^flap_count, MAX_RETRY_BACKOFF)` with ±25% jitter.

---

**File**: `rust/networking/src/swarm.rs`

**Function**: `tcp_transport`

**Specific Changes**:

1. **Enable TCP keepalive**: Configure the TCP transport with keepalive enabled to maintain connections through brief LACP slave transitions:

   ```rust
   let tcp_config = Config::default()
       .nodelay(true)
       .port_reuse(true);  // SO_REUSEPORT for consistent port binding
   ```

   Note: libp2p's TCP `Config` may not directly expose `TCP_KEEPALIVE`. If not available via the config API, we can set it via a socket option on the underlying transport. The key parameters are:
   - `TCP_KEEPIDLE`: 10 seconds (start probes after 10s idle)
   - `TCP_KEEPINTVL`: 5 seconds (probe every 5s)
   - `TCP_KEEPCNT`: 3 (give up after 3 failed probes)

2. **Enable port reuse**: `port_reuse(true)` allows the transport to reuse the listening port for outgoing connections. This means dial attempts use the same source port as the listener, producing a consistent LACP hash for connections to the same destination. This is the most impactful single change for LACP stability.

---

**File**: `rust/networking/src/discovery.rs`

**Function**: `poll()`

**Specific Changes**:

1. **Poll pending connection grace timers**: Add a loop before the retry timer that checks each `pending_connections` entry. If the grace timer has expired, promote the connection (emit event, reset flap count).

   ```rust
   // Poll grace period timers for pending connections
   let mut promoted = Vec::new();
   for (conn_id, pending) in self.pending_connections.iter_mut() {
       if pending.grace_timer.poll_unpin(cx).is_ready() {
           promoted.push(*conn_id);
       }
   }
   for conn_id in promoted {
       if let Some(pending) = self.pending_connections.remove(&conn_id) {
           // Connection survived grace period — it's stable
           self.peer_flap_counts.remove(&pending.peer_id);
           self.peer_next_retry.remove(&pending.peer_id);
           self.pending_events.push_back(
               ToSwarm::GenerateEvent(Event::ConnectionEstablished {
                   peer_id: pending.peer_id,
                   connection_id: conn_id,
                   remote_ip: pending.remote_ip,
                   remote_tcp_port: pending.remote_tcp_port,
               })
           );
       }
   }
   ```

2. **Add backoff check to retry loop**:
   ```rust
   for (p, mas) in self.static_peers.clone() {
       if self.connected_peers.contains_key(&p) {
           continue;
       }
       // Skip peers in backoff
       if let Some(next_retry) = self.peer_next_retry.get(&p) {
           if Instant::now() < *next_retry {
               continue;
           }
       }
       for ma in mas {
           self.dial(p, ma)
       }
   }
   ```

---

**File**: No changes required to `src/exo/shared/election.py`

The Python election layer's port-aware flap detection remains unchanged. With the Rust-side grace period, the Python layer will only receive events for connections that have already proven stable (survived 2+ seconds). The existing flap detection becomes a secondary safety net for non-LACP flaps (e.g., brief network blips on stable connections).

---

**File**: No changes required to `rust/exo_pyo3_bindings/src/networking.rs`

The PyO3 bindings layer passes through whatever events the discovery behaviour emits. Since the grace period filtering happens inside the discovery behaviour, the bindings layer is unchanged.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behaviour.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write unit tests against the `Behaviour` struct that simulate rapid connection establishment followed by immediate teardown (mimicking LACP-induced flaps). Run these tests on the UNFIXED code to observe that `ConnectionEstablished` and `ConnectionClosed` events are both emitted immediately, confirming the Python layer receives flap pairs.

**Test Cases**:
1. **Immediate teardown emits both events**: Establish a connection, then close it within 100ms — assert both `ConnectionEstablished` and `ConnectionClosed` are emitted (will pass on unfixed code, confirming the bug mechanism)
2. **Repeated flaps produce repeated event pairs**: Simulate 5 consecutive connect+disconnect cycles at 5-second intervals — assert 5 pairs of events are emitted with no backoff (will pass on unfixed code)
3. **No suppression of short-lived connections**: Establish a connection that lives 500ms then dies — assert `ConnectionEstablished` is emitted immediately (will pass on unfixed code, showing no grace period exists)
4. **Retry loop fires at fixed interval regardless of flap history**: After 10 consecutive flaps, assert the retry loop still fires every 5 seconds (will pass on unfixed code, showing no backoff)

**Expected Counterexamples**:
- Every connection, no matter how short-lived, produces a `ConnectionEstablished` event immediately
- The retry loop has no backoff mechanism — it fires every 5 seconds regardless of flap history
- Possible causes: no grace period, no flap counting, no backoff logic

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behaviour.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  behaviour := create_behaviour_with_grace_period(2s)
  conn_id := simulate_connection_established(behaviour, input.peer_id)
  simulate_connection_closed(behaviour, conn_id, after=100ms)
  events := collect_emitted_events(behaviour)
  ASSERT ConnectionEstablished NOT IN events  // suppressed by grace period
  ASSERT ConnectionClosed NOT IN events       // never reported
  ASSERT peer_flap_counts[input.peer_id] > 0  // flap tracked
  ASSERT peer_next_retry[input.peer_id] > now  // backoff applied
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  // Connection survives grace period
  behaviour := create_behaviour_with_grace_period(2s)
  conn_id := simulate_connection_established(behaviour, input.peer_id)
  advance_time(2.1s)  // past grace period
  events := collect_emitted_events(behaviour)
  ASSERT ConnectionEstablished IN events  // reported after grace period
  
  simulate_connection_closed(behaviour, conn_id)
  close_events := collect_emitted_events(behaviour)
  ASSERT ConnectionClosed IN close_events  // reported normally
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many random connection durations and verifies that connections surviving the grace period are always reported
- It catches edge cases like connections dying exactly at the grace period boundary
- It provides strong guarantees that stable connection event semantics are unchanged
- It can generate random interleaved mDNS and static peer events to verify no interference

**Test Plan**: Observe behaviour on UNFIXED code first for stable connections (those lasting > 2s), then write property-based tests capturing that behaviour and verifying it's preserved in the fixed code.

**Test Cases**:
1. **Stable connection event timing**: Generate random connection durations > 2s — verify `ConnectionEstablished` is emitted exactly once, at the grace period expiry time (not at connection time)
2. **mDNS lifecycle preservation**: Generate random mDNS discover/expire sequences — verify `mdns_discovered` state is identical in fixed vs unfixed code
3. **Ping disconnection preservation**: Generate random ping failure events for stable connections — verify `ConnectionClosed` is emitted identically
4. **Retry loop mDNS preservation**: Generate random `mdns_discovered` states with no backoff peers, trigger retry timer — verify the same set of dial actions are produced

### Unit Tests

- Test grace period suppression: connection dies at 1s → no events emitted
- Test grace period promotion: connection survives 2.1s → `ConnectionEstablished` emitted
- Test grace period boundary: connection dies at exactly 2s → no events (timer hasn't fired yet)
- Test `connected_peers` tracking during grace period: peer is in `connected_peers` even before promotion (prevents duplicate dials)
- Test flap count increment: 3 consecutive sub-grace-period connections → `peer_flap_counts == 3`
- Test flap count reset: after one stable connection → `peer_flap_counts == 0`
- Test exponential backoff calculation: flap_count=0 → 5s, flap_count=1 → 10s, flap_count=2 → 20s, capped at 60s
- Test backoff jitter: retry times have ±25% variance (not deterministic)
- Test retry loop skips backed-off peers: peer with `next_retry` in future is not dialed
- Test `ConnectionClosed` for stable connection: connection promoted, then closed → `ConnectionClosed` emitted
- Test `ConnectionClosed` for pending connection: connection in grace period, then closed → no `ConnectionClosed` emitted
- Test TCP keepalive configuration: verify transport config includes keepalive parameters
- Test port reuse configuration: verify transport config has `port_reuse(true)`

### Property-Based Tests

- Generate random sequences of (connect, duration, disconnect) events with durations drawn from [0ms, 10s]. Verify that events are emitted if and only if duration ≥ grace period.
- Generate random peer sets with random flap counts. Verify backoff durations follow the exponential formula with jitter bounds.
- Generate random interleaved mDNS discover/expire and static peer add/remove operations. Verify `mdns_discovered` and `static_peers` remain independent tracking sets.
- Generate random connection IDs and verify `pending_connections` cleanup: no memory leaks (entries removed on close or promotion).
- Generate random retry loop invocations with mixed backed-off and ready peers. Verify only ready peers produce dial actions.

### Integration Tests

- Deploy to two LACP-bonded gremlin nodes with the fix. Verify stable cluster formation within 60 seconds (previously never formed).
- Verify that non-LACP nodes (goblin RPi5s) still form clusters immediately (grace period is transparent for stable connections).
- Verify that after a genuine network disruption (e.g., `ip link set bond0 down` for 5s then up), the connection re-establishes after the backoff period.
- Verify that the Python election layer receives clean Connected events (no flap pairs) after the fix.
- Verify gossipsub message delivery works correctly with the delayed `ConnectionEstablished` event (gossipsub peer addition happens at promotion time, not connection time).
