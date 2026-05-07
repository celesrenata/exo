# Bugfix Requirements Document

## Introduction

The 4-node gremlin cluster (10.1.1.12–15) cannot form a stable gossipsub mesh over LACP-bonded ethernet connections. While TCP connections establish successfully and the libp2p transport layer (noise + yamux) negotiates correctly, gossipsub protocol substreams immediately fail with "Stream closed. Confirmation from remote for optimistic protocol negotiation still pending." This causes rapid connect/disconnect cycles that the Python election code interprets as connection flaps, resulting in the cluster permanently staying at 1 node.

The root cause is in the Rust networking stack: the discovery behaviour's `NetworkBehaviour` implementation uses a `dummy::ConnectionHandler` in its `handle_established_inbound_connection` and `handle_established_outbound_connection` methods. This handler does not support any protocols. When composed with gossipsub via `#[derive(NetworkBehaviour)]`, the `V1Lazy` (optimistic 0-RTT) protocol negotiation interacts poorly with the handler composition — gossipsub substream opens are not properly confirmed before the optimistic data arrives, causing the remote to close the stream. Additionally, the `keep_alive::ConnectionHandler` (which forces `connection_keep_alive() -> true`) is defined but never actually used in the discovery behaviour's connection handler setup.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN two nodes attempt gossipsub protocol negotiation over a V1Lazy transport with the discovery behaviour using `dummy::ConnectionHandler` THEN the system closes the gossipsub substream with "Stream closed. Confirmation from remote for optimistic protocol negotiation still pending" every 5 seconds

1.2 WHEN gossipsub substreams repeatedly fail THEN the system enters rapid connect/disconnect cycles at the libp2p connection level (180 "Stream closed" messages in 3 minutes observed)

1.3 WHEN the Python election code receives a connect event followed by a disconnect event within the same 200ms batch THEN the system logs "Connection flap detected (connect+disconnect in same batch), ignoring" and discards both events

1.4 WHEN all gossipsub substreams fail and connection events are discarded as flaps THEN the system never forms a multi-node cluster (stays at 1 node indefinitely, no gossipsub topic subscriptions ever appear)

1.5 WHEN the discovery behaviour creates connection handlers via `handle_established_inbound_connection` THEN the system uses `dummy::ConnectionHandler` instead of `keep_alive::ConnectionHandler`, providing no protocol support and no keep-alive guarantee for the discovery handler slot

### Expected Behavior (Correct)

2.1 WHEN two nodes attempt gossipsub protocol negotiation THEN the system SHALL successfully negotiate gossipsub substreams and establish stable protocol communication without "Stream closed" errors

2.2 WHEN gossipsub substreams are established THEN the system SHALL maintain stable connections without rapid connect/disconnect cycles (no repeated stream failures within seconds)

2.3 WHEN nodes establish stable connections THEN the system SHALL deliver connection events to the Python election code as genuine connects (not paired with immediate disconnects in the same batch)

2.4 WHEN all 4 nodes have stable gossipsub connections THEN the system SHALL form a complete cluster visible via the /state API endpoint with all nodes subscribed to shared topics (GLOBAL_EVENTS, LOCAL_EVENTS, COMMANDS, ELECTION_MESSAGES, CONNECTION_MESSAGES)

2.5 WHEN the discovery behaviour creates connection handlers THEN the system SHALL use a connection handler that properly supports keep-alive semantics and does not interfere with gossipsub protocol negotiation in the composed behaviour

### Unchanged Behavior (Regression Prevention)

3.1 WHEN mDNS discovers or expires peers THEN the system SHALL CONTINUE TO track peers in `mdns_discovered` and emit dial/close actions correctly

3.2 WHEN the retry timer fires THEN the system SHALL CONTINUE TO re-dial only disconnected peers from both `mdns_discovered` and `static_peers` (skipping already-connected peers)

3.3 WHEN a connection is established or closed THEN the system SHALL CONTINUE TO emit `Event::ConnectionEstablished` and `Event::ConnectionClosed` events with correct peer_id, connection_id, remote_ip, and remote_tcp_port

3.4 WHEN a ping to a connected peer fails THEN the system SHALL CONTINUE TO close that connection immediately

3.5 WHEN the Python election code receives a genuine master disconnection (not a flap) THEN the system SHALL CONTINUE TO wait the 5-second grace period and trigger re-election if the master does not reconnect

3.6 WHEN static peers are configured via EXO_PEERS THEN the system SHALL CONTINUE TO promote unknown peers to `static_peers` upon connection establishment and retry disconnected static peers on the retry timer

3.7 WHEN the private network (pnet) layer is active THEN the system SHALL CONTINUE TO use the SHA3-256 hash of "exo_discovery_network" + NETWORK_VERSION as the pre-shared key, preventing cross-version communication

---

## Bug Condition (Formal)

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type LibP2PConnectionAttempt
  OUTPUT: boolean

  // The bug triggers when gossipsub attempts to negotiate a substream
  // over a connection where the discovery behaviour's handler slot uses
  // dummy::ConnectionHandler with V1Lazy optimistic negotiation
  RETURN X.transport_version = V1Lazy
     AND X.discovery_handler = dummy::ConnectionHandler
     AND X.gossipsub_substream_requested = true
END FUNCTION
```

```pascal
// Property: Fix Checking — Gossipsub Substreams Succeed
FOR ALL X WHERE isBugCondition(X) DO
  result ← negotiate_gossipsub_substream'(X)
  ASSERT result.stream_established = true
     AND result.no_stream_closed_error = true
     AND result.topic_subscriptions_visible = true
END FOR
```

```pascal
// Property: Preservation Checking — Non-gossipsub Behaviour Unchanged
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT F(X) = F'(X)
  // mDNS discovery, ping, static peer retry, connection event emission,
  // and election flap detection all behave identically
END FOR
```
