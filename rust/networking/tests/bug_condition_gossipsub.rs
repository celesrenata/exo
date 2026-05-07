//! Bug Condition Exploration Test — Gossipsub Substream Failure Under V1Lazy + Dummy Handler
//!
//! **Validates: Requirements 1.1, 1.2, 1.4, 1.5**
//!
//! This test encodes the EXPECTED behavior after the fix is applied:
//! - The discovery behaviour's connection handler MUST use `keep_alive::ConnectionHandler`
//!   (which returns `connection_keep_alive() == true`) instead of `dummy::ConnectionHandler`
//!   (which returns `connection_keep_alive() == false` and uses `DeniedUpgrade`).
//! - The composed handler's keep_alive MUST be true to prevent premature connection closure.
//!
//! On UNFIXED code, these tests MUST FAIL because:
//! - `dummy::ConnectionHandler` is used in the discovery handler slot
//! - `dummy::ConnectionHandler` returns `connection_keep_alive() == false`
//! - `dummy::ConnectionHandler` uses `DeniedUpgrade` (advertises NO protocols)
//! - Under V1Lazy transport, this causes gossipsub substream negotiation to fail with
//!   "Stream closed. Confirmation from remote for optimistic protocol negotiation still pending"
//!
//! Bug Condition (formal):
//!   isBugCondition(input) WHERE
//!     input.transport_version = V1Lazy
//!     AND input.discovery_handler = dummy::ConnectionHandler
//!     AND input.gossipsub_substream_requested = true
//!
//! The structural defect: `dummy::ConnectionHandler` in the discovery handler slot
//! advertises no protocols and does not keep connections alive. When composed with
//! gossipsub via `ConnectionHandlerSelect`, the V1Lazy optimistic negotiation sends
//! data before the remote confirms protocol support. Since the discovery slot's handler
//! (dummy) denies all upgrades, the remote cannot confirm, causing stream closure.
//!
//! Expected Counterexamples (on unfixed code):
//! - Discovery handler's left slot returns `connection_keep_alive() == false`
//! - Discovery handler's left slot uses `DeniedUpgrade` (no protocol support)
//! - The composed handler does NOT guarantee keep-alive from the discovery slot

#[cfg(test)]
mod bug_condition_tests {
    use libp2p::core::transport::PortUse;
    use libp2p::core::Endpoint;
    use libp2p::identity::Keypair;
    use libp2p::multiaddr::Protocol;
    use libp2p::swarm::handler::ConnectionHandler as ConnectionHandlerTrait;
    use libp2p::swarm::{ConnectionId, NetworkBehaviour};
    use libp2p::Multiaddr;
    use networking::discovery::Behaviour;
    use std::net::Ipv4Addr;

    /// Helper: create a Behaviour inside a tokio runtime (required for mDNS init).
    fn new_behaviour() -> Behaviour {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        let keypair = Keypair::generate_ed25519();
        rt.block_on(async { Behaviour::new(&keypair).expect("behaviour should initialize") })
    }

    /// Helper: build a `/ip4/127.0.0.1/tcp/{port}` multiaddr.
    fn make_multiaddr(port: u16) -> Multiaddr {
        let mut ma = Multiaddr::empty();
        ma.push(Protocol::Ip4(Ipv4Addr::LOCALHOST));
        ma.push(Protocol::Tcp(port));
        ma
    }

    /// Property 1: Bug Condition — Discovery Handler MUST Keep Connections Alive
    ///
    /// **Validates: Requirements 1.5, 2.5**
    ///
    /// The discovery behaviour's connection handler (left slot of the composed handler)
    /// MUST return `connection_keep_alive() == true` to prevent premature connection
    /// closure that interferes with gossipsub substream negotiation.
    ///
    /// On UNFIXED code: FAILS because `dummy::ConnectionHandler` returns false.
    /// After fix: PASSES because `keep_alive::ConnectionHandler` returns true.
    #[test]
    fn discovery_handler_keeps_connection_alive_inbound() {
        let mut behaviour = new_behaviour();

        let peer_kp = Keypair::generate_ed25519();
        let peer_id = peer_kp.public().to_peer_id();
        let conn_id = ConnectionId::new_unchecked(0);
        let local_addr = make_multiaddr(4001);
        let remote_addr = make_multiaddr(5001);

        // Get the connection handler from the discovery behaviour
        let handler = behaviour
            .handle_established_inbound_connection(conn_id, peer_id, &local_addr, &remote_addr)
            .expect("handler creation should succeed");

        // Decompose the ConnectionHandlerSelect to get the left (discovery) handler
        let (discovery_handler, _managed_handler) = handler.into_inner();

        // ASSERTION: The discovery handler MUST keep connections alive.
        // On UNFIXED code, `dummy::ConnectionHandler` returns `false` here.
        // After the fix, `keep_alive::ConnectionHandler` returns `true`.
        assert!(
            discovery_handler.connection_keep_alive(),
            "BUG CONFIRMED: Discovery handler does NOT keep connections alive. \
             The handler is `dummy::ConnectionHandler` which returns `connection_keep_alive() == false`. \
             This causes premature connection closure under V1Lazy transport, preventing \
             gossipsub substream negotiation from completing. \
             Expected: `keep_alive::ConnectionHandler` with `connection_keep_alive() == true`. \
             (Requirements 1.5, 2.5)"
        );
    }

    /// Property 1: Bug Condition — Discovery Handler MUST Keep Connections Alive (Outbound)
    ///
    /// **Validates: Requirements 1.5, 2.5**
    ///
    /// Same as above but for outbound connections. Both inbound and outbound handlers
    /// must use `keep_alive::ConnectionHandler` to prevent the V1Lazy race condition.
    ///
    /// On UNFIXED code: FAILS because `dummy::ConnectionHandler` returns false.
    /// After fix: PASSES because `keep_alive::ConnectionHandler` returns true.
    #[test]
    fn discovery_handler_keeps_connection_alive_outbound() {
        let mut behaviour = new_behaviour();

        let peer_kp = Keypair::generate_ed25519();
        let peer_id = peer_kp.public().to_peer_id();
        let conn_id = ConnectionId::new_unchecked(0);
        let addr = make_multiaddr(4001);

        // Get the outbound connection handler from the discovery behaviour
        let handler = behaviour
            .handle_established_outbound_connection(
                conn_id,
                peer_id,
                &addr,
                Endpoint::Dialer,
                PortUse::Reuse,
            )
            .expect("handler creation should succeed");

        // Decompose the ConnectionHandlerSelect to get the left (discovery) handler
        let (discovery_handler, _managed_handler) = handler.into_inner();

        // ASSERTION: The discovery handler MUST keep connections alive.
        // On UNFIXED code, `dummy::ConnectionHandler` returns `false` here.
        // After the fix, `keep_alive::ConnectionHandler` returns `true`.
        assert!(
            discovery_handler.connection_keep_alive(),
            "BUG CONFIRMED: Discovery outbound handler does NOT keep connections alive. \
             The handler is `dummy::ConnectionHandler` which returns `connection_keep_alive() == false`. \
             This causes premature connection closure under V1Lazy transport, preventing \
             gossipsub substream negotiation from completing on outbound connections. \
             Expected: `keep_alive::ConnectionHandler` with `connection_keep_alive() == true`. \
             (Requirements 1.5, 2.5)"
        );
    }

    /// Property 1: Bug Condition — Composed Handler MUST Guarantee Keep-Alive
    ///
    /// **Validates: Requirements 1.1, 1.2**
    ///
    /// The full composed handler (ConnectionHandlerSelect<discovery_handler, managed_handler>)
    /// MUST return `connection_keep_alive() == true`. The `ConnectionHandlerSelect` uses
    /// `max(left.keep_alive, right.keep_alive)` — so if the left (discovery) handler
    /// returns false, the composed handler's keep-alive depends entirely on the managed
    /// handler (mDNS + ping). This is unreliable because the managed handler may not
    /// immediately generate traffic, allowing the connection to be closed before gossipsub
    /// can negotiate its substream.
    ///
    /// On UNFIXED code: The left handler (dummy) returns false, making keep-alive
    /// dependent on the managed handler's state — which is unreliable.
    /// After fix: The left handler (keep_alive) always returns true, guaranteeing
    /// the connection stays alive for gossipsub negotiation.
    #[test]
    fn composed_handler_guarantees_keep_alive_from_discovery_slot() {
        let mut behaviour = new_behaviour();

        let peer_kp = Keypair::generate_ed25519();
        let peer_id = peer_kp.public().to_peer_id();
        let conn_id = ConnectionId::new_unchecked(0);
        let local_addr = make_multiaddr(4001);
        let remote_addr = make_multiaddr(5001);

        let handler = behaviour
            .handle_established_inbound_connection(conn_id, peer_id, &local_addr, &remote_addr)
            .expect("handler creation should succeed");

        // Get the left (discovery) handler specifically
        let (discovery_handler, _) = handler.into_inner();

        // The discovery handler slot MUST independently guarantee keep-alive.
        // It must NOT rely on the managed handler to keep the connection alive.
        // On UNFIXED code: dummy::ConnectionHandler returns false — the discovery
        // slot does NOT independently guarantee keep-alive.
        assert!(
            discovery_handler.connection_keep_alive(),
            "BUG CONFIRMED: The discovery handler slot does NOT independently guarantee \
             connection keep-alive. Using `dummy::ConnectionHandler` means the connection's \
             liveness depends entirely on the managed sub-behaviour (mDNS + ping) generating \
             traffic quickly enough. Under V1Lazy transport with network latency, this race \
             condition causes gossipsub substreams to fail with 'Stream closed. Confirmation \
             from remote for optimistic protocol negotiation still pending.' \
             (Requirements 1.1, 1.2)"
        );
    }
}
