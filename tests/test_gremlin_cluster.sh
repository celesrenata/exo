#!/usr/bin/env bash
# Multi-node cluster validation test for gremlin cluster
# This script validates cluster formation and stability across multiple nodes

set -e

# Configuration
GREMLIN_NODES=("gremlin-1:10.1.1.12" "gremlin-2:10.1.1.13" "gremlin-3:10.1.1.14" "gremlin-4:10.1.1.15")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

test_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((TESTS_PASSED++))
}

test_fail() {
    echo -e "${RED}✗${NC} $1"
    ((TESTS_FAILED++))
}

# Parse node info
get_node_name() {
    echo "$1" | cut -d: -f1
}

get_node_ip() {
    echo "$1" | cut -d: -f2
}

# Test 10.4: Test cluster formation
test_cluster_formation() {
    log_info "Test 10.4: Testing cluster formation"
    
    # Check each node is accessible
    log_info "Checking node accessibility..."
    local accessible_nodes=0
    
    for node_info in "${GREMLIN_NODES[@]}"; do
        local node_name=$(get_node_name "$node_info")
        local node_ip=$(get_node_ip "$node_info")
        
        if curl -s --connect-timeout 5 "http://${node_ip}:52415/health" >/dev/null 2>&1; then
            log_info "  ✓ $node_name ($node_ip) is accessible"
            ((accessible_nodes++))
        else
            log_warn "  ✗ $node_name ($node_ip) is not accessible"
        fi
    done
    
    if [ $accessible_nodes -eq ${#GREMLIN_NODES[@]} ]; then
        test_pass "All nodes are accessible"
    elif [ $accessible_nodes -gt 0 ]; then
        test_warn "Only $accessible_nodes/${#GREMLIN_NODES[@]} nodes are accessible"
    else
        test_fail "No nodes are accessible"
        return 1
    fi
    
    # Check cluster status from first accessible node
    log_info "Checking cluster status..."
    for node_info in "${GREMLIN_NODES[@]}"; do
        local node_ip=$(get_node_ip "$node_info")
        
        if curl -s --connect-timeout 5 "http://${node_ip}:52415/health" >/dev/null 2>&1; then
            # Try to get cluster info
            local cluster_info=$(curl -s --connect-timeout 5 "http://${node_ip}:52415/cluster" 2>/dev/null || echo "")
            
            if [ -n "$cluster_info" ]; then
                local node_count=$(echo "$cluster_info" | python -c "import sys, json; print(len(json.load(sys.stdin).get('nodes', [])))" 2>/dev/null || echo "0")
                
                if [ "$node_count" -gt 1 ]; then
                    test_pass "Cluster formed with $node_count nodes"
                    return 0
                else
                    log_warn "Only $node_count node(s) in cluster"
                fi
            fi
            break
        fi
    done
    
    test_warn "Could not verify cluster formation (API may not expose cluster info)"
}

# Test node discovery
test_node_discovery() {
    log_info "Testing node discovery..."
    
    # Check if nodes can see each other
    local discovery_working=false
    
    for node_info in "${GREMLIN_NODES[@]}"; do
        local node_name=$(get_node_name "$node_info")
        local node_ip=$(get_node_ip "$node_info")
        
        if curl -s --connect-timeout 5 "http://${node_ip}:52415/health" >/dev/null 2>&1; then
            log_info "Checking discovery from $node_name..."
            
            # Check if node can see peers
            local peers=$(ssh "root@${node_ip}" "journalctl -u exo -n 100 | grep -i 'peer\|discover\|connect'" 2>/dev/null || echo "")
            
            if echo "$peers" | grep -q "peer\|discover"; then
                log_info "  $node_name shows peer discovery activity"
                discovery_working=true
            fi
        fi
    done
    
    if [ "$discovery_working" = true ]; then
        test_pass "Nodes are discovering each other"
    else
        test_warn "Cannot confirm node discovery (check logs manually)"
    fi
}

# Test dashboard visibility
test_dashboard() {
    log_info "Testing dashboard visibility..."
    
    for node_info in "${GREMLIN_NODES[@]}"; do
        local node_name=$(get_node_name "$node_info")
        local node_ip=$(get_node_ip "$node_info")
        
        if curl -s --connect-timeout 5 "http://${node_ip}:52415/" >/dev/null 2>&1; then
            test_pass "Dashboard accessible on $node_name"
            
            # Check if dashboard shows node info
            local dashboard_html=$(curl -s --connect-timeout 5 "http://${node_ip}:52415/" 2>/dev/null || echo "")
            
            if echo "$dashboard_html" | grep -q "node\|cluster\|gpu\|intel"; then
                log_info "  Dashboard shows cluster information"
            fi
            
            return 0
        fi
    done
    
    test_fail "Dashboard not accessible on any node"
}

# Test 10.5: Validate cluster stability
test_cluster_stability() {
    log_info "Test 10.5: Validating cluster stability"
    
    local duration_minutes=${1:-60}  # Default 60 minutes
    local check_interval=60  # Check every 60 seconds
    local checks=$((duration_minutes * 60 / check_interval))
    
    log_info "Running stability test for $duration_minutes minutes..."
    log_info "Checking cluster health every $check_interval seconds"
    
    local failures=0
    local disconnections=0
    
    for ((i=1; i<=checks; i++)); do
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local accessible=0
        
        # Check each node
        for node_info in "${GREMLIN_NODES[@]}"; do
            local node_name=$(get_node_name "$node_info")
            local node_ip=$(get_node_ip "$node_info")
            
            if curl -s --connect-timeout 5 "http://${node_ip}:52415/health" >/dev/null 2>&1; then
                ((accessible++))
            else
                log_warn "[$timestamp] $node_name is not responding"
                ((disconnections++))
            fi
        done
        
        if [ $accessible -eq ${#GREMLIN_NODES[@]} ]; then
            echo -ne "\r[$timestamp] Check $i/$checks: All nodes healthy ($accessible/${#GREMLIN_NODES[@]})    "
        else
            echo ""
            log_warn "[$timestamp] Check $i/$checks: Only $accessible/${#GREMLIN_NODES[@]} nodes responding"
            ((failures++))
        fi
        
        # Sleep until next check
        if [ $i -lt $checks ]; then
            sleep $check_interval
        fi
    done
    
    echo ""
    log_info "Stability test complete"
    log_info "  Total checks: $checks"
    log_info "  Failed checks: $failures"
    log_info "  Node disconnections: $disconnections"
    
    local failure_rate=$((failures * 100 / checks))
    
    if [ $failure_rate -eq 0 ]; then
        test_pass "Cluster remained stable (0% failure rate)"
    elif [ $failure_rate -lt 5 ]; then
        test_pass "Cluster mostly stable ($failure_rate% failure rate)"
    elif [ $failure_rate -lt 20 ]; then
        test_warn "Cluster had some instability ($failure_rate% failure rate)"
    else
        test_fail "Cluster was unstable ($failure_rate% failure rate)"
    fi
}

# Test model sharding across nodes
test_model_sharding() {
    log_info "Testing model sharding across cluster..."
    
    # Try to load a model that requires sharding
    local model="meta-llama/Llama-3.2-3B"
    
    log_info "Requesting model: $model"
    
    # Send request to first accessible node
    for node_info in "${GREMLIN_NODES[@]}"; do
        local node_ip=$(get_node_ip "$node_info")
        
        if curl -s --connect-timeout 5 "http://${node_ip}:52415/health" >/dev/null 2>&1; then
            local response=$(curl -s --connect-timeout 10 -X POST "http://${node_ip}:52415/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{
                    \"model\": \"$model\",
                    \"messages\": [{\"role\": \"user\", \"content\": \"test\"}],
                    \"max_tokens\": 1,
                    \"stream\": false
                }" 2>&1)
            
            if echo "$response" | grep -q "downloading\|loading\|choices\|sharding"; then
                test_pass "Model sharding initiated"
                
                # Check if multiple nodes are involved
                log_info "Checking if model is distributed across nodes..."
                sleep 5
                
                local nodes_with_model=0
                for check_node in "${GREMLIN_NODES[@]}"; do
                    local check_ip=$(get_node_ip "$check_node")
                    local check_name=$(get_node_name "$check_node")
                    
                    if ssh "root@${check_ip}" "journalctl -u exo -n 50 | grep -i '$model'" 2>/dev/null | grep -q "shard\|load"; then
                        log_info "  $check_name has model shard"
                        ((nodes_with_model++))
                    fi
                done
                
                if [ $nodes_with_model -gt 1 ]; then
                    test_pass "Model sharded across $nodes_with_model nodes"
                else
                    test_warn "Model sharding not confirmed (check logs manually)"
                fi
                
                return 0
            fi
            break
        fi
    done
    
    test_warn "Could not test model sharding (may require manual verification)"
}

# Test inference across cluster
test_cluster_inference() {
    log_info "Testing inference across cluster..."
    
    local model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Send inference requests to different nodes
    local successful_inferences=0
    
    for node_info in "${GREMLIN_NODES[@]}"; do
        local node_name=$(get_node_name "$node_info")
        local node_ip=$(get_node_ip "$node_info")
        
        if curl -s --connect-timeout 5 "http://${node_ip}:52415/health" >/dev/null 2>&1; then
            log_info "Testing inference on $node_name..."
            
            local response=$(curl -s --connect-timeout 30 -X POST "http://${node_ip}:52415/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{
                    \"model\": \"$model\",
                    \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
                    \"max_tokens\": 5,
                    \"stream\": false
                }" 2>&1)
            
            if echo "$response" | grep -q "choices"; then
                log_info "  ✓ Inference successful on $node_name"
                ((successful_inferences++))
            else
                log_warn "  ✗ Inference failed on $node_name"
            fi
        fi
    done
    
    if [ $successful_inferences -gt 0 ]; then
        test_pass "Inference working on $successful_inferences node(s)"
    else
        test_fail "Inference not working on any node"
    fi
}

# Main test execution
main() {
    log_info "Starting cluster validation tests"
    log_info "Nodes: ${GREMLIN_NODES[*]}"
    log_info "=================================================="
    
    # Test 10.4: Cluster formation
    test_cluster_formation
    test_node_discovery
    test_dashboard
    
    # Test model operations
    test_model_sharding
    test_cluster_inference
    
    # Test 10.5: Stability (default 60 minutes, can be overridden)
    local stability_duration=${1:-60}
    test_cluster_stability "$stability_duration"
    
    # Summary
    echo ""
    log_info "=================================================="
    log_info "Cluster Test Summary"
    log_info "=================================================="
    echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
    echo -e "${RED}Failed:${NC} $TESTS_FAILED"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        log_info "All cluster tests passed! ✓"
        exit 0
    else
        log_error "Some tests failed. Please review the output above."
        exit 1
    fi
}

# Run main with optional duration argument
main "$@"
