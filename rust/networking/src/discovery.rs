use crate::ext::MultiaddrExt;
use crate::keep_alive;
use delegate::delegate;
use either::Either;
use futures::FutureExt;
use futures_timer::Delay;
use libp2p::core::transport::PortUse;
use libp2p::core::{ConnectedPoint, Endpoint};
use libp2p::swarm::behaviour::ConnectionEstablished;
use libp2p::swarm::dial_opts::DialOpts;
use libp2p::swarm::{
    CloseConnection, ConnectionClosed, ConnectionDenied, ConnectionHandler,
    ConnectionHandlerSelect, ConnectionId, FromSwarm, NetworkBehaviour, THandler, THandlerInEvent,
    THandlerOutEvent, ToSwarm, dummy,
};
use libp2p::{Multiaddr, PeerId, identity, mdns};
use std::collections::{BTreeSet, HashMap};
use std::convert::Infallible;
use std::io;
use std::net::IpAddr;
use std::task::{Context, Poll};
use std::time::Duration;
use util::wakerdeque::WakerDeque;

const RETRY_CONNECT_INTERVAL: Duration = Duration::from_secs(5);

mod managed {
    use libp2p::swarm::NetworkBehaviour;
    use libp2p::{identity, mdns, ping};
    use std::io;
    use std::time::Duration;

    const MDNS_RECORD_TTL: Duration = Duration::from_secs(2_500);
    const MDNS_QUERY_INTERVAL: Duration = Duration::from_secs(1_500);
    const PING_TIMEOUT: Duration = Duration::from_secs(10);
    const PING_INTERVAL: Duration = Duration::from_secs(10);

    #[derive(NetworkBehaviour)]
    pub struct Behaviour {
        mdns: mdns::tokio::Behaviour,
        ping: ping::Behaviour,
    }

    impl Behaviour {
        pub fn new(keypair: &identity::Keypair) -> io::Result<Self> {
            Ok(Self {
                mdns: mdns_behaviour(keypair)?,
                ping: ping_behaviour(),
            })
        }
    }

    fn mdns_behaviour(keypair: &identity::Keypair) -> io::Result<mdns::tokio::Behaviour> {
        use mdns::{Config, tokio};

        // mDNS config => enable IPv6
        let mdns_config = Config {
            ttl: MDNS_RECORD_TTL,
            query_interval: MDNS_QUERY_INTERVAL,

            // enable_ipv6: true, // TODO: for some reason, TCP+mDNS don't work well with ipv6?? figure out how to make work
            ..Default::default()
        };

        let mdns_behaviour = tokio::Behaviour::new(mdns_config, keypair.public().to_peer_id());
        Ok(mdns_behaviour?)
    }

    fn ping_behaviour() -> ping::Behaviour {
        ping::Behaviour::new(
            ping::Config::new()
                .with_timeout(PING_TIMEOUT)
                .with_interval(PING_INTERVAL),
        )
    }
}

/// Events for when a listening connection is truly established and truly closed.
#[derive(Debug, Clone)]
pub enum Event {
    ConnectionEstablished {
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    },
    ConnectionClosed {
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    },
}

/// Discovery behavior that wraps mDNS to produce truly discovered durable peer-connections.
///
/// The behaviour operates as such:
///  1) All true (listening) connections/disconnections are tracked, emitting corresponding events
///     to the swarm.
///  1) mDNS discovered/expired peers are tracked; discovered but not connected peers are dialed
///     immediately, and expired but connected peers are disconnected from immediately.
///  2) Every fixed interval: discovered but not connected peers are dialed, and expired but
///     connected peers are disconnected from.
pub struct Behaviour {
    // state-tracking for managed behaviors & mDNS-discovered peers
    managed: managed::Behaviour,
    mdns_discovered: HashMap<PeerId, BTreeSet<Multiaddr>>,
    static_peers: HashMap<PeerId, BTreeSet<Multiaddr>>,
    /// Multiaddrs dialed without a peer ID (from EXO_PEERS).
    /// When a connection is established, we match the remote address
    /// and promote the peer to `static_peers` for retry tracking.
    pending_static_addrs: BTreeSet<Multiaddr>,

    retry_delay: Delay, // retry interval

    // pending events to emmit => waker-backed Deque to control polling
    pending_events: WakerDeque<ToSwarm<Event, Infallible>>,
}

impl Behaviour {
    pub fn new(keypair: &identity::Keypair) -> io::Result<Self> {
        Ok(Self {
            managed: managed::Behaviour::new(keypair)?,
            mdns_discovered: HashMap::new(),
            static_peers: HashMap::new(),
            pending_static_addrs: BTreeSet::new(),
            retry_delay: Delay::new(RETRY_CONNECT_INTERVAL),
            pending_events: WakerDeque::new(),
        })
    }

    pub fn add_static_peer(&mut self, peer_id: PeerId, addr: Multiaddr) {
        self.static_peers
            .entry(peer_id)
            .or_insert_with(BTreeSet::new)
            .insert(addr.clone());
        self.dial(peer_id, addr);
    }

    /// Register a multiaddr for static peer tracking without a known peer ID.
    /// The address is stored in `pending_static_addrs`. When a connection is
    /// established to this address, the peer ID is learned and the peer is
    /// promoted to `static_peers` for automatic retry.
    pub fn dial_unknown_peer(&mut self, addr: Multiaddr) {
        self.pending_static_addrs.insert(addr);
    }

    fn dial(&mut self, peer_id: PeerId, addr: Multiaddr) {
        self.pending_events.push_back(ToSwarm::Dial {
            opts: DialOpts::peer_id(peer_id).addresses(vec![addr]).build(),
        })
    }

    fn close_connection(&mut self, peer_id: PeerId, connection: ConnectionId) {
        // push front to make this IMMEDIATE
        self.pending_events.push_front(ToSwarm::CloseConnection {
            peer_id,
            connection: CloseConnection::One(connection),
        })
    }

    fn handle_mdns_discovered(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (p, ma) in peers {
            self.dial(p, ma.clone()); // always connect

            // get peer's multi-addresses or insert if missing
            let Some(mas) = self.mdns_discovered.get_mut(&p) else {
                self.mdns_discovered.insert(p, BTreeSet::from([ma]));
                continue;
            };

            // multiaddress should never already be present - else something has gone wrong
            let is_new_addr = mas.insert(ma);
            assert!(is_new_addr, "cannot discover a discovered peer");
        }
    }

    fn handle_mdns_expired(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (p, ma) in peers {
            // at this point, we *must* have the peer
            let mas = self
                .mdns_discovered
                .get_mut(&p)
                .expect("nonexistent peer cannot expire");

            // at this point, we *must* have the multiaddress
            let was_present = mas.remove(&ma);
            assert!(was_present, "nonexistent multiaddress cannot expire");

            // if empty, remove the peer-id entirely
            if mas.is_empty() {
                self.mdns_discovered.remove(&p);
            }
        }
    }

    fn on_connection_established(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        // Check if this connection matches a pending static addr (dialed without peer ID).
        // If so, promote to static_peers for automatic retry on disconnect.
        if !self.pending_static_addrs.is_empty() {
            // Build the multiaddr for this connection to match against pending
            let mut connected_addr = Multiaddr::empty();
            match remote_ip {
                IpAddr::V4(ip) => connected_addr.push(libp2p::multiaddr::Protocol::Ip4(ip)),
                IpAddr::V6(ip) => connected_addr.push(libp2p::multiaddr::Protocol::Ip6(ip)),
            }
            connected_addr.push(libp2p::multiaddr::Protocol::Tcp(remote_tcp_port));

            // Check if any pending addr matches this connection's IP (port may differ
            // due to ephemeral ports, so match on IP only)
            let matching_addr = self.pending_static_addrs.iter().find(|pending| {
                pending.iter().any(|p| match (p, remote_ip) {
                    (libp2p::multiaddr::Protocol::Ip4(a), IpAddr::V4(b)) => a == b,
                    (libp2p::multiaddr::Protocol::Ip6(a), IpAddr::V6(b)) => a == b,
                    _ => false,
                })
            }).cloned();

            if let Some(addr) = matching_addr {
                log::info!(
                    "RUST: promoting peer {} to static_peers (connected via {})",
                    peer_id, addr
                );
                self.static_peers
                    .entry(peer_id)
                    .or_insert_with(BTreeSet::new)
                    .insert(addr.clone());
                // Don't remove from pending — we want to keep the addr for future
                // connections if this peer's identity changes on restart
            }
        }

        // send out connected event
        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionEstablished {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
    }

    fn on_connection_closed(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        // send out disconnected event
        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionClosed {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
    }
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler =
        ConnectionHandlerSelect<dummy::ConnectionHandler, THandler<managed::Behaviour>>;
    type ToSwarm = Event;

    // simply delegate to underlying mDNS behaviour

    delegate! {
        to self.managed {
            fn handle_pending_inbound_connection(&mut self, connection_id: ConnectionId, local_addr: &Multiaddr, remote_addr: &Multiaddr) -> Result<(), ConnectionDenied>;
            fn handle_pending_outbound_connection(&mut self, connection_id: ConnectionId, maybe_peer: Option<PeerId>, addresses: &[Multiaddr], effective_role: Endpoint) -> Result<Vec<Multiaddr>, ConnectionDenied>;
        }
    }

    fn handle_established_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(ConnectionHandler::select(
            dummy::ConnectionHandler,
            self.managed.handle_established_inbound_connection(
                connection_id,
                peer,
                local_addr,
                remote_addr,
            )?,
        ))
    }

    #[allow(clippy::needless_question_mark)]
    fn handle_established_outbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        addr: &Multiaddr,
        role_override: Endpoint,
        port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(ConnectionHandler::select(
            dummy::ConnectionHandler,
            self.managed.handle_established_outbound_connection(
                connection_id,
                peer,
                addr,
                role_override,
                port_use,
            )?,
        ))
    }

    fn on_connection_handler_event(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        event: THandlerOutEvent<Self>,
    ) {
        match event {
            Either::Left(ev) => libp2p::core::util::unreachable(ev),
            Either::Right(ev) => {
                self.managed
                    .on_connection_handler_event(peer_id, connection_id, ev)
            }
        }
    }

    // hook into these methods to drive behavior

    fn on_swarm_event(&mut self, event: FromSwarm) {
        self.managed.on_swarm_event(event); // let mDNS handle swarm events

        // handle swarm events to update internal state:
        match event {
            FromSwarm::ConnectionEstablished(ConnectionEstablished {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                let remote_address = match endpoint {
                    ConnectedPoint::Dialer { address, .. } => address,
                    ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                };

                if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                    // handle connection established event which is filtered correctly
                    self.on_connection_established(peer_id, connection_id, ip, port)
                }
            }
            FromSwarm::ConnectionClosed(ConnectionClosed {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                let remote_address = match endpoint {
                    ConnectedPoint::Dialer { address, .. } => address,
                    ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                };

                if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                    // handle connection closed event which is filtered correctly
                    self.on_connection_closed(peer_id, connection_id, ip, port)
                }
            }

            // since we are running TCP/IP transport layer, we are assuming that
            // no address changes can occur, hence encountering one is a fatal error
            FromSwarm::AddressChange(a) => {
                unreachable!("unhandlable: address change encountered: {:?}", a)
            }
            _ => {}
        }
    }

    fn poll(&mut self, cx: &mut Context) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        // delegate to managed behaviors for any behaviors they need to perform
        match self.managed.poll(cx) {
            Poll::Ready(ToSwarm::GenerateEvent(e)) => {
                match e {
                    // handle discovered and expired events from mDNS
                    managed::BehaviourEvent::Mdns(e) => match e.clone() {
                        mdns::Event::Discovered(peers) => {
                            self.handle_mdns_discovered(peers);
                        }
                        mdns::Event::Expired(peers) => {
                            self.handle_mdns_expired(peers);
                        }
                    },

                    // handle ping events => if error then disconnect
                    managed::BehaviourEvent::Ping(e) => {
                        if let Err(_) = e.result {
                            self.close_connection(e.peer, e.connection.clone())
                        }
                    }
                }

                // since we just consumed an event, we should immediately wake just in case
                // there are more events to come where that came from
                cx.waker().wake_by_ref();
            }

            // forward any other mDNS event to the swarm or its connection handler(s)
            Poll::Ready(e) => {
                return Poll::Ready(
                    e.map_out(|_| unreachable!("events returning to swarm already handled"))
                        .map_in(Either::Right),
                );
            }

            Poll::Pending => {}
        }

        // retry connecting to all mDNS peers periodically (fails safely if already connected)
        if self.retry_delay.poll_unpin(cx).is_ready() {
            for (p, mas) in self.mdns_discovered.clone() {
                for ma in mas {
                    self.dial(p, ma)
                }
            }
            // also retry static peers
            for (p, mas) in self.static_peers.clone() {
                for ma in mas {
                    self.dial(p, ma)
                }
            }
            self.retry_delay.reset(RETRY_CONNECT_INTERVAL) // reset timeout
        }

        // send out any pending events from our own service
        if let Some(e) = self.pending_events.pop_front(cx) {
            return Poll::Ready(e.map_in(Either::Left));
        }

        // wait for pending events
        Poll::Pending
    }
}


#[cfg(test)]
mod preservation_tests {
    //! Preservation Property Tests — mDNS Peer Lifecycle Unchanged
    //!
    //! **Validates: Requirements 3.1, 3.2, 3.3**
    //!
    //! These tests capture the EXISTING correct mDNS behaviour on UNFIXED code.
    //! They must PASS before and after the fix, ensuring no regressions in:
    //!   - mDNS discover/expire tracking in `mdns_discovered`
    //!   - Retry timer dial actions matching `mdns_discovered` entries
    //!   - `ConnectionEstablished` / `ConnectionClosed` event emission
    //!   - Ping-failure-based connection closure

    use super::*;
    use libp2p::identity::Keypair;
    use libp2p::multiaddr::Protocol;
    use libp2p::swarm::ConnectionId;
    use proptest::prelude::*;
    use std::collections::{BTreeSet, HashSet};
    use std::net::Ipv4Addr;

    /// Helper: build a `/ip4/127.0.0.1/tcp/{port}/p2p/{peer_id}` multiaddr.
    fn make_multiaddr(port: u16, peer_id: &PeerId) -> Multiaddr {
        let mut ma = Multiaddr::empty();
        ma.push(Protocol::Ip4(Ipv4Addr::LOCALHOST));
        ma.push(Protocol::Tcp(port));
        ma.push(Protocol::P2p(*peer_id));
        ma
    }

    /// Helper: create a Behaviour inside a tokio runtime (required for mDNS init).
    fn new_behaviour() -> Behaviour {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        let keypair = Keypair::generate_ed25519();
        rt.block_on(async { Behaviour::new(&keypair).expect("behaviour should initialize") })
    }

    /// Helper: generate N unique (PeerId, Multiaddr) pairs.
    fn generate_peers(n: usize) -> Vec<(PeerId, Multiaddr)> {
        (0..n)
            .map(|i| {
                let kp = Keypair::generate_ed25519();
                let pid = kp.public().to_peer_id();
                // Use port 10000 + i to ensure unique addresses
                #[allow(clippy::as_conversions)]
                let ma = make_multiaddr(10_000_u16.saturating_add(i as u16), &pid);
                (pid, ma)
            })
            .collect()
    }

    /// Helper: drain all pending events from a Behaviour using a noop waker.
    fn drain_pending_events(
        behaviour: &mut Behaviour,
    ) -> Vec<ToSwarm<Event, std::convert::Infallible>> {
        let waker = futures::task::noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        let mut events = Vec::new();
        while let Some(ev) = behaviour.pending_events.pop_front(&mut cx) {
            events.push(ev);
        }
        events
    }

    /// Helper: extract PeerIds from Dial actions in pending events.
    fn extract_dial_targets(
        events: &[ToSwarm<Event, std::convert::Infallible>],
    ) -> HashSet<PeerId> {
        let mut targets = HashSet::new();
        for ev in events {
            if let ToSwarm::Dial { opts } = ev {
                if let Some(pid) = opts.get_peer_id() {
                    targets.insert(pid);
                }
            }
        }
        targets
    }

    // =========================================================================
    // Property 2a: mDNS discover/expire consistency
    // For random sequences of mDNS discover/expire events, verify
    // `mdns_discovered` state is consistent.
    // **Validates: Requirements 3.1**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property: After discovering N peers, `mdns_discovered` contains exactly
        /// those N peers. After expiring a subset, only the non-expired remain.
        ///
        /// **Validates: Requirements 3.1**
        #[test]
        fn mdns_discover_then_expire_subset_is_consistent(
            total_peers in 1_usize..=10,
            expire_count_pct in 0_usize..=100,
        ) {
            let mut behaviour = new_behaviour();

            let peers = generate_peers(total_peers);

            // Discover all peers
            behaviour.handle_mdns_discovered(peers.clone());

            // Verify all peers are in mdns_discovered
            for (pid, ma) in &peers {
                let mas = behaviour.mdns_discovered.get(pid)
                    .expect("discovered peer must be in mdns_discovered");
                prop_assert!(mas.contains(ma), "discovered address must be tracked");
            }
            prop_assert_eq!(behaviour.mdns_discovered.len(), total_peers);

            // Expire a subset (percentage-based to let proptest shrink)
            let expire_count = (total_peers * expire_count_pct) / 100;
            let to_expire: Vec<_> = peers.iter().take(expire_count).cloned().collect();
            let to_remain: Vec<_> = peers.iter().skip(expire_count).cloned().collect();

            if !to_expire.is_empty() {
                behaviour.handle_mdns_expired(to_expire.clone());
            }

            // Verify expired peers are gone
            for (pid, _ma) in &to_expire {
                prop_assert!(
                    !behaviour.mdns_discovered.contains_key(pid),
                    "expired peer must be removed from mdns_discovered"
                );
            }

            // Verify remaining peers are still present
            for (pid, ma) in &to_remain {
                let mas = behaviour.mdns_discovered.get(pid)
                    .expect("non-expired peer must remain in mdns_discovered");
                prop_assert!(mas.contains(ma), "non-expired address must remain tracked");
            }

            prop_assert_eq!(behaviour.mdns_discovered.len(), to_remain.len());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property: Discovering peers emits a Dial action for each discovered peer.
        ///
        /// **Validates: Requirements 3.1**
        #[test]
        fn mdns_discover_emits_dial_for_each_peer(
            total_peers in 1_usize..=8,
        ) {
            let mut behaviour = new_behaviour();

            let peers = generate_peers(total_peers);

            behaviour.handle_mdns_discovered(peers.clone());

            // Drain pending events and check dial actions
            let events = drain_pending_events(&mut behaviour);
            let dialed = extract_dial_targets(&events);

            for (pid, _ma) in &peers {
                prop_assert!(
                    dialed.contains(pid),
                    "discovered peer must be dialed immediately"
                );
            }
        }
    }

    // =========================================================================
    // Property 2b: Retry timer dial actions match mdns_discovered
    // For random `mdns_discovered` states, the retry loop dials exactly those
    // peers (and ONLY those peers).
    // **Validates: Requirements 3.1**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property: The retry loop dials exactly the peers in `mdns_discovered`.
        /// We simulate the retry loop logic (same code path as poll()) and verify
        /// the dial targets match.
        ///
        /// **Validates: Requirements 3.1**
        #[test]
        fn retry_loop_dials_exactly_mdns_discovered_peers(
            total_peers in 1_usize..=10,
        ) {
            let mut behaviour = new_behaviour();

            let peers = generate_peers(total_peers);

            // Populate mdns_discovered directly (simulating prior discovery)
            for (pid, ma) in &peers {
                behaviour.mdns_discovered
                    .entry(*pid)
                    .or_insert_with(BTreeSet::new)
                    .insert(ma.clone());
            }

            // Drain any existing events (from setup)
            let _ = drain_pending_events(&mut behaviour);

            // Simulate the retry loop (same logic as poll() retry timer block)
            for (p, mas) in behaviour.mdns_discovered.clone() {
                for ma in mas {
                    behaviour.dial(p, ma);
                }
            }

            // Collect dial actions
            let events = drain_pending_events(&mut behaviour);
            let dialed = extract_dial_targets(&events);

            // Expected: exactly the peers in mdns_discovered
            let expected: HashSet<PeerId> = peers.iter().map(|(pid, _)| *pid).collect();

            prop_assert_eq!(
                dialed, expected,
                "retry loop must dial exactly the mdns_discovered peers"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// Property: After discovering and then expiring some peers, the retry
        /// loop only dials the remaining (non-expired) peers.
        ///
        /// **Validates: Requirements 3.1**
        #[test]
        fn retry_loop_excludes_expired_peers(
            total_peers in 2_usize..=8,
            expire_count in 1_usize..=4,
        ) {
            let expire_count = expire_count.min(total_peers.saturating_sub(1));
            let mut behaviour = new_behaviour();

            let peers = generate_peers(total_peers);

            // Discover all
            behaviour.handle_mdns_discovered(peers.clone());
            let _ = drain_pending_events(&mut behaviour); // clear discovery dials

            // Expire a subset
            let to_expire: Vec<_> = peers.iter().take(expire_count).cloned().collect();
            behaviour.handle_mdns_expired(to_expire.clone());

            // Simulate retry loop
            for (p, mas) in behaviour.mdns_discovered.clone() {
                for ma in mas {
                    behaviour.dial(p, ma);
                }
            }

            let events = drain_pending_events(&mut behaviour);
            let dialed = extract_dial_targets(&events);

            // Expired peers must NOT be dialed
            for (pid, _) in &to_expire {
                prop_assert!(
                    !dialed.contains(pid),
                    "expired peer must not be dialed by retry loop"
                );
            }

            // Remaining peers must be dialed
            let remaining: HashSet<PeerId> = peers.iter()
                .skip(expire_count)
                .map(|(pid, _)| *pid)
                .collect();
            prop_assert_eq!(
                dialed, remaining,
                "retry loop must dial only remaining mdns_discovered peers"
            );
        }
    }

    // =========================================================================
    // Property 2c: ConnectionEstablished and ConnectionClosed events emitted
    // for all connection lifecycle events regardless of peer origin.
    // **Validates: Requirements 3.2, 3.3**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property: `on_connection_established` emits `Event::ConnectionEstablished`
        /// for any peer, regardless of origin.
        ///
        /// **Validates: Requirements 3.3**
        #[test]
        fn connection_established_event_emitted_for_any_peer(
            port in 1024_u16..=65534,
        ) {
            let mut behaviour = new_behaviour();

            let peer_kp = Keypair::generate_ed25519();
            let peer_id = peer_kp.public().to_peer_id();
            let conn_id = ConnectionId::new_unchecked(0);
            let ip = std::net::IpAddr::V4(Ipv4Addr::new(192, 168, 1, port.to_be_bytes()[0].saturating_add(1)));

            behaviour.on_connection_established(peer_id, conn_id, ip, port);

            let events = drain_pending_events(&mut behaviour);

            // Must have exactly one ConnectionEstablished event
            let established_events: Vec<_> = events.iter().filter(|e| {
                matches!(e, ToSwarm::GenerateEvent(Event::ConnectionEstablished { .. }))
            }).collect();

            prop_assert_eq!(
                established_events.len(), 1,
                "exactly one ConnectionEstablished event must be emitted"
            );

            // Verify the event contents
            if let ToSwarm::GenerateEvent(Event::ConnectionEstablished {
                peer_id: ev_pid,
                connection_id: ev_cid,
                remote_tcp_port: ev_port,
                ..
            }) = &established_events[0] {
                prop_assert_eq!(*ev_pid, peer_id);
                prop_assert_eq!(*ev_cid, conn_id);
                prop_assert_eq!(*ev_port, port);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property: `on_connection_closed` emits `Event::ConnectionClosed`
        /// for any peer, regardless of origin.
        ///
        /// **Validates: Requirements 3.3**
        #[test]
        fn connection_closed_event_emitted_for_any_peer(
            port in 1024_u16..=65534,
        ) {
            let mut behaviour = new_behaviour();

            let peer_kp = Keypair::generate_ed25519();
            let peer_id = peer_kp.public().to_peer_id();
            let conn_id = ConnectionId::new_unchecked(0);
            let ip = std::net::IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));

            behaviour.on_connection_closed(peer_id, conn_id, ip, port);

            let events = drain_pending_events(&mut behaviour);

            let closed_events: Vec<_> = events.iter().filter(|e| {
                matches!(e, ToSwarm::GenerateEvent(Event::ConnectionClosed { .. }))
            }).collect();

            prop_assert_eq!(
                closed_events.len(), 1,
                "exactly one ConnectionClosed event must be emitted"
            );

            if let ToSwarm::GenerateEvent(Event::ConnectionClosed {
                peer_id: ev_pid,
                connection_id: ev_cid,
                remote_tcp_port: ev_port,
                ..
            }) = &closed_events[0] {
                prop_assert_eq!(*ev_pid, peer_id);
                prop_assert_eq!(*ev_cid, conn_id);
                prop_assert_eq!(*ev_port, port);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// Property: `close_connection` emits a `CloseConnection` action for the
        /// specific peer/connection when a ping fails.
        ///
        /// **Validates: Requirements 3.2**
        #[test]
        fn close_connection_targets_specific_peer(
            conn_idx in 0_usize..=100,
        ) {
            let mut behaviour = new_behaviour();

            let peer_kp = Keypair::generate_ed25519();
            let peer_id = peer_kp.public().to_peer_id();
            let conn_id = ConnectionId::new_unchecked(conn_idx);

            behaviour.close_connection(peer_id, conn_id);

            let events = drain_pending_events(&mut behaviour);

            let close_events: Vec<_> = events.iter().filter(|e| {
                matches!(e, ToSwarm::CloseConnection { .. })
            }).collect();

            prop_assert_eq!(
                close_events.len(), 1,
                "exactly one CloseConnection action must be emitted"
            );

            if let ToSwarm::CloseConnection {
                peer_id: ev_pid,
                connection: CloseConnection::One(ev_cid),
            } = &close_events[0] {
                prop_assert_eq!(*ev_pid, peer_id);
                prop_assert_eq!(*ev_cid, conn_id);
            } else {
                prop_assert!(false, "CloseConnection must target One specific connection");
            }
        }
    }

    // =========================================================================
    // Deterministic unit tests for edge cases
    // =========================================================================

    /// Verify that discovering then fully expiring all peers leaves
    /// `mdns_discovered` empty.
    ///
    /// **Validates: Requirements 3.1**
    #[test]
    fn discover_then_expire_all_leaves_empty_map() {
        let mut behaviour = new_behaviour();

        let peers = generate_peers(5);
        behaviour.handle_mdns_discovered(peers.clone());
        assert_eq!(behaviour.mdns_discovered.len(), 5);

        behaviour.handle_mdns_expired(peers);
        assert!(
            behaviour.mdns_discovered.is_empty(),
            "all peers expired => mdns_discovered must be empty"
        );
    }

    /// Verify that connection events are emitted in order: established then closed.
    ///
    /// **Validates: Requirements 3.3**
    #[test]
    fn connection_lifecycle_events_emitted_in_order() {
        let mut behaviour = new_behaviour();

        let peer_kp = Keypair::generate_ed25519();
        let peer_id = peer_kp.public().to_peer_id();
        let conn_id = ConnectionId::new_unchecked(42);
        let ip = std::net::IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));

        behaviour.on_connection_established(peer_id, conn_id, ip, 5678);
        behaviour.on_connection_closed(peer_id, conn_id, ip, 5678);

        let events = drain_pending_events(&mut behaviour);

        assert_eq!(events.len(), 2, "must have exactly 2 events");
        assert!(
            matches!(
                &events[0],
                ToSwarm::GenerateEvent(Event::ConnectionEstablished { .. })
            ),
            "first event must be ConnectionEstablished"
        );
        assert!(
            matches!(
                &events[1],
                ToSwarm::GenerateEvent(Event::ConnectionClosed { .. })
            ),
            "second event must be ConnectionClosed"
        );
    }

    /// Verify that close_connection is pushed to the FRONT of pending events
    /// (immediate priority).
    ///
    /// **Validates: Requirements 3.2**
    #[test]
    fn close_connection_has_immediate_priority() {
        let mut behaviour = new_behaviour();

        let peer_a_kp = Keypair::generate_ed25519();
        let peer_a = peer_a_kp.public().to_peer_id();
        let peer_b_kp = Keypair::generate_ed25519();
        let peer_b = peer_b_kp.public().to_peer_id();

        // First push a dial (goes to back)
        let addr = make_multiaddr(5678, &peer_a);
        behaviour.dial(peer_a, addr);

        // Then push a close_connection (goes to front)
        let conn_id = ConnectionId::new_unchecked(0);
        behaviour.close_connection(peer_b, conn_id);

        let events = drain_pending_events(&mut behaviour);

        assert_eq!(events.len(), 2);
        // CloseConnection should be first (pushed to front)
        assert!(
            matches!(&events[0], ToSwarm::CloseConnection { .. }),
            "CloseConnection must be at front (immediate priority)"
        );
        assert!(
            matches!(&events[1], ToSwarm::Dial { .. }),
            "Dial must be after CloseConnection"
        );
    }
}
