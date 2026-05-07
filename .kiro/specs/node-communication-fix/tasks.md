# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Gossipsub Substream Failure Under V1Lazy + Dummy Handler
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate gossipsub substreams fail when the discovery behaviour uses `dummy::ConnectionHandler` with `V1Lazy` transport
  - **Scoped PBT Approach**: Create two swarms with current config (`dummy::ConnectionHandler` + `V1Lazy`), connect them, subscribe to a gossipsub topic on both, and assert that subscription messages propagate within 10 seconds
  - Write test in `rust/networking/src/discovery.rs` (new test module `bug_condition_tests`) or as a separate integration test in `rust/networking/tests/`
  - Test setup: create two swarms using `create_swarm()`, connect them via dial, subscribe both to a shared topic, attempt to publish a message
  - Assert: gossipsub mesh forms (peer appears in mesh peers for the topic) AND message is delivered to subscriber
  - The bug condition from design: `isBugCondition(input)` where `input.transport_version = V1Lazy AND input.discovery_handler = dummy::ConnectionHandler AND input.gossipsub_substream_requested = true`
  - Run test on UNFIXED code with `cargo test -p networking`
  - **EXPECTED OUTCOME**: Test FAILS (gossipsub substreams close with "Confirmation from remote for optimistic protocol negotiation still pending" — this proves the bug exists)
  - Document counterexamples found (e.g., "Two swarms connected but gossipsub subscription never propagates; substream closed within 5 seconds")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.4, 1.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - mDNS Peer Lifecycle and Connection Events Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - **NOTE**: The existing `preservation_tests` module in `discovery.rs` already captures the correct mDNS/connection lifecycle behavior with proptest — verify these pass on UNFIXED code
  - Observe: Run `cargo test -p networking preservation_tests` on unfixed code — all existing proptests must pass
  - Observe: `handle_mdns_discovered(peers)` populates `mdns_discovered` and emits Dial actions for each peer
  - Observe: `handle_mdns_expired(peers)` removes peers from `mdns_discovered`
  - Observe: Retry loop dials exactly `mdns_discovered` + `static_peers` minus `connected_peers`
  - Observe: `on_connection_established` emits `Event::ConnectionEstablished` with correct fields
  - Observe: `on_connection_closed` emits `Event::ConnectionClosed` with correct fields
  - Observe: `close_connection` emits `CloseConnection::One` targeting the specific peer/connection (pushed to front for immediate priority)
  - Write property-based test: for all non-bug-condition inputs (mDNS discover/expire sequences, connection lifecycle events, ping failures), the discovery behaviour produces the same actions regardless of whether the handler is `dummy::ConnectionHandler` or `keep_alive::ConnectionHandler` (from Preservation Requirements in design)
  - Verify all existing preservation proptests pass on UNFIXED code: `cargo test -p networking preservation_tests`
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when existing tests are verified passing and any additional preservation tests are written and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 3. Fix for gossipsub substream negotiation failure

  - [x] 3.1 Implement the fix in `discovery.rs` and `swarm.rs`
    - In `rust/networking/src/discovery.rs`: Replace `dummy::ConnectionHandler` with `keep_alive::ConnectionHandler::new()` in `handle_established_inbound_connection` (line ~275)
    - In `rust/networking/src/discovery.rs`: Replace `dummy::ConnectionHandler` with `keep_alive::ConnectionHandler::new()` in `handle_established_outbound_connection` (line ~290)
    - In `rust/networking/src/discovery.rs`: Update the `ConnectionHandler` type alias from `ConnectionHandlerSelect<dummy::ConnectionHandler, THandler<managed::Behaviour>>` to `ConnectionHandlerSelect<keep_alive::ConnectionHandler, THandler<managed::Behaviour>>`
    - In `rust/networking/src/swarm.rs`: Change `let upgrade_version = Version::V1Lazy;` to `let upgrade_version = Version::V1;`
    - In `rust/networking/src/swarm.rs`: Update comment from `// V1 + lazy flushing => 0-RTT negotiation` to `// V1 => wait for protocol confirmation before sending data (reliable negotiation)`
    - Verify build compiles: `cargo build -p networking` (from rust/ directory)
    - _Bug_Condition: isBugCondition(input) where input.transport_version = V1Lazy AND input.discovery_handler = dummy::ConnectionHandler AND input.gossipsub_substream_requested = true_
    - _Expected_Behavior: gossipsub substreams negotiate successfully, connections remain stable, no "Stream closed" errors_
    - _Preservation: mDNS discovery/expiry, retry timer, connection event emission, ping failure closure, static peer promotion all unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Gossipsub Substreams Succeed After Fix
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (gossipsub mesh forms and messages deliver)
    - When this test passes, it confirms gossipsub substreams negotiate successfully with `keep_alive::ConnectionHandler` + `V1`
    - Run bug condition exploration test from step 1: `cargo test -p networking` (the gossipsub substream test)
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed — gossipsub substreams now negotiate successfully)
    - _Requirements: 2.1, 2.2_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - mDNS Peer Lifecycle and Connection Events Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2: `cargo test -p networking preservation_tests`
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions in mDNS discovery, retry logic, connection events, ping failure handling)
    - Confirm all proptests still pass after fix (no regressions)

  - [x] 3.4 Full build verification
    - Build the full release binary: `cargo build -p exo_pyo3_bindings --release` (from rust/ directory)
    - Verify no warnings or errors related to the changes
    - _Requirements: 2.1, 2.5_

- [x] 4. Integration test — deploy to cluster and validate 4-node formation
  - Deploy to all 4 gremlin nodes: `bash deploy_cluster.sh`
  - Verify cluster state shows 4 nodes: `curl -s http://10.1.1.12:52415/state | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Nodes: {len(d.get(\"topology\",{}).get(\"nodes\",{}))}')"` — expect "Nodes: 4"
  - Verify no "Stream closed" messages in journalctl logs after 2 minutes: `ssh root@10.1.1.12 "journalctl -u exo --since '2 minutes ago' --no-pager | grep -c 'Stream closed'"` — expect 0
  - Verify gossipsub topic subscriptions visible in logs: `ssh root@10.1.1.12 "journalctl -u exo --since '2 minutes ago' --no-pager | grep 'Subscribed'"` — expect GLOBAL_EVENTS, LOCAL_EVENTS, COMMANDS, ELECTION_MESSAGES, CONNECTION_MESSAGES
  - Verify election completes (master elected): `curl -s http://10.1.1.12:52415/state | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('topology',{}).get('master_id','NO MASTER'))"` — expect a peer ID
  - Verify no connection flap messages: `ssh root@10.1.1.12 "journalctl -u exo --since '2 minutes ago' --no-pager | grep -c 'flap'"` — expect 0
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all Rust tests pass: `cargo test -p networking` (from rust/ directory)
  - Ensure cluster remains stable for 5+ minutes with 4 nodes
  - Ensure no regressions in Python election code behavior
  - Ask the user if questions arise.
