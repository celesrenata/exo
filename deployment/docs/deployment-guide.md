# EXO Multinode Race Condition Fix - Deployment Guide

This guide provides step-by-step instructions for deploying the multinode race condition fixes to EXO distributed AI inference systems.

## Overview

The multinode race condition fixes address critical issues in EXO's distributed inference system:

- **"Queue is closed" errors** during runner shutdown
- **ClosedResourceError exceptions** in multiprocessing communication
- **Runner failures with exit code 1** during coordination
- **Deadlocks and race conditions** in multi-node scenarios

### Solution Components

1. **Three-Phase Shutdown Protocol**: Coordinated shutdown across all processes
2. **Resource Manager**: Dependency-aware cleanup with proper ordering
3. **Channel Manager**: Race-condition-free multiprocessing communication
4. **Enhanced Error Handling**: Improved recovery and logging mechanisms

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, or NixOS)
- **Python**: Version 3.13 or higher
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: Minimum 10GB free space for deployment
- **Network**: Stable network connection for multi-node communication

### Software Dependencies

```bash
# Python and package management
python3 >= 3.13
pip or uv package manager

# System utilities
curl
systemctl (for systemd systems)
journalctl (for log management)

# Optional but recommended
git (for version control)
htop (for monitoring)
```

### Access Requirements

- **Root/sudo access** for system service management
- **Network access** to EXO API endpoints (port 52415 by default)
- **File system permissions** to read/write EXO installation directory

## Pre-Deployment Preparation

### 1. Environment Assessment

Run the pre-deployment assessment script:

```bash
cd /path/to/exo
./deployment/monitoring/validate-deployment.sh
```

This will check:
- Current service health
- Existing error patterns
- Resource usage
- System compatibility

### 2. Create Backup

**Critical**: Always create a backup before deployment:

```bash
# Automated backup (recommended)
./deployment/deploy-race-condition-fix.sh --backup-only

# Manual backup
mkdir -p deployment/backups/manual_$(date +%Y%m%d_%H%M%S)
cp -r src/exo/worker/runner/ deployment/backups/manual_$(date +%Y%m%d_%H%M%S)/
cp pyproject.toml deployment/backups/manual_$(date +%Y%m%d_%H%M%S)/
```

### 3. Schedule Maintenance Window

For production systems:

1. **Notify stakeholders** of planned maintenance
2. **Schedule during low-usage periods** (typically off-peak hours)
3. **Prepare rollback plan** in case of issues
4. **Have monitoring tools ready** for post-deployment validation

## Deployment Methods

### Method 1: Automated Deployment (Recommended)

The automated deployment script handles the entire process:

```bash
cd /path/to/exo
./deployment/deploy-race-condition-fix.sh
```

**What the script does:**
1. Validates environment and prerequisites
2. Creates automatic backup of current installation
3. Stops EXO services gracefully
4. Deploys the race condition fixes
5. Runs validation tests
6. Starts services and validates functionality
7. Generates deployment report

**Expected output:**
```
🎉 EXO Race Condition Fixes Deployed Successfully!
📊 Monitor with: journalctl -u exo -f
🌐 Dashboard: http://localhost:52415
📋 Report: deployment/logs/deployment_report_20250102_143000.md

Expected improvements:
  ✅ No more 'Queue is closed' errors
  ✅ No more 'ClosedResourceError' exceptions
  ✅ Graceful multi-node shutdown coordination
  ✅ Better error recovery and logging
```

### Method 2: Manual Deployment

For environments requiring manual control:

#### Step 1: Stop Services

```bash
# Stop systemd service
sudo systemctl stop exo

# Verify no EXO processes are running
pgrep -f exo || echo "No EXO processes found"
```

#### Step 2: Install Updates

```bash
cd /path/to/exo

# Update package with race condition fixes
if command -v uv &> /dev/null; then
    uv sync --all-packages
else
    pip install -e . --force-reinstall
fi
```

#### Step 3: Validate Installation

```bash
# Test import
python -c "import exo; print('EXO import successful')"

# Run unit tests
python -m pytest src/exo/worker/tests/unittests/test_runner/ -v
```

#### Step 4: Start Services

```bash
# Start systemd service
sudo systemctl start exo

# Verify service is active
systemctl is-active exo
```

#### Step 5: Validate Deployment

```bash
# Run validation script
./deployment/monitoring/validate-deployment.sh

# Check for race condition errors
journalctl -u exo --since "10 minutes ago" | grep -E "Queue is closed|ClosedResourceError"
```

### Method 3: NixOS Deployment

For NixOS systems using the flake configuration:

#### Step 1: Update Flake

```bash
cd /path/to/nixos-config

# Update EXO input to latest version with fixes
nix flake update exo

# Or use local development version
# Edit flake.nix to point to local EXO directory
```

#### Step 2: Rebuild System

```bash
# Rebuild and switch
sudo nixos-rebuild switch --flake .#your-hostname

# Or test first
sudo nixos-rebuild test --flake .#your-hostname
```

#### Step 3: Validate NixOS Deployment

```bash
# Check service status
systemctl status exo

# Validate functionality
curl -f http://localhost:52415/health
```

## Post-Deployment Validation

### Immediate Validation (0-15 minutes)

1. **Service Health Check**
   ```bash
   # Check service status
   systemctl status exo
   
   # Verify endpoints respond
   curl -f http://localhost:52415/health
   curl -f http://localhost:52415/state
   curl -f http://localhost:52415/
   ```

2. **Error Log Review**
   ```bash
   # Check for race condition errors
   journalctl -u exo --since "15 minutes ago" | grep -E "Queue is closed|ClosedResourceError|Runner.*exit.*code.*1"
   
   # Should return no results if fixes are working
   ```

3. **Basic Functionality Test**
   ```bash
   # Run basic functionality test
   python test_basic_functionality.py
   
   # Run multinode integration test
   python validate_multinode_integration.py
   ```

### Short-term Validation (15 minutes - 2 hours)

1. **Automated Monitoring**
   ```bash
   # Run comprehensive validation
   ./deployment/monitoring/validate-deployment.sh
   
   # Schedule regular checks
   watch -n 300 './deployment/monitoring/validate-deployment.sh'
   ```

2. **Performance Monitoring**
   ```bash
   # Monitor resource usage
   htop -p $(pgrep -f exo)
   
   # Check response times
   for i in {1..10}; do
     time curl -s http://localhost:52415/health > /dev/null
   done
   ```

3. **Multi-node Testing** (if applicable)
   ```bash
   # Test multi-node instance creation
   curl -X POST http://localhost:52415/instances \
     -H "Content-Type: application/json" \
     -d '{"model": "test-model", "nodes": 2}'
   
   # Monitor coordination logs
   journalctl -u exo -f | grep -i "coordination\|shutdown"
   ```

### Long-term Validation (2+ hours)

1. **Stability Testing**
   - Monitor for 24-48 hours minimum
   - Check for memory leaks or resource accumulation
   - Verify no regression in existing functionality

2. **Load Testing** (if applicable)
   - Run inference workloads
   - Test under various load conditions
   - Verify graceful handling of high concurrency

3. **Error Pattern Analysis**
   ```bash
   # Daily error analysis
   journalctl -u exo --since "24 hours ago" -p err | wc -l
   
   # Should show significant reduction in errors
   ```

## Rollback Procedures

### When to Rollback

Consider rollback if:
- **Critical errors** persist after deployment
- **Service becomes unresponsive** or unstable
- **Performance degradation** is observed
- **New error patterns** emerge that weren't present before

### Automated Rollback

```bash
# Find backup directory (created during deployment)
ls -la deployment/backups/

# Execute rollback
./deployment/rollback-race-condition-fix.sh deployment/backups/20250102_143000
```

### Manual Rollback

If automated rollback fails:

```bash
# Stop services
sudo systemctl stop exo

# Restore files from backup
BACKUP_DIR="deployment/backups/20250102_143000"
cp -r $BACKUP_DIR/src/exo/worker/runner/* src/exo/worker/runner/
cp $BACKUP_DIR/pyproject.toml .

# Reinstall package
pip install -e . --force-reinstall

# Start services
sudo systemctl start exo

# Validate rollback
systemctl status exo
curl -f http://localhost:52415/health
```

## Troubleshooting

### Common Deployment Issues

#### 1. Permission Errors

**Error**: `Permission denied` during file operations

**Solution**:
```bash
# Check file ownership
ls -la src/exo/worker/runner/

# Fix ownership if needed
sudo chown -R $(whoami):$(whoami) src/exo/worker/runner/

# Check directory permissions
chmod 755 src/exo/worker/runner/
```

#### 2. Service Won't Start

**Error**: `systemctl start exo` fails

**Diagnosis**:
```bash
# Check service logs
journalctl -u exo --since "5 minutes ago"

# Check configuration
systemctl cat exo

# Verify binary
which exo
```

**Solution**:
```bash
# Reset service state
sudo systemctl reset-failed exo
sudo systemctl daemon-reload

# Check dependencies
python -c "import exo; print('Import successful')"
```

#### 3. Import Errors

**Error**: `ModuleNotFoundError` or import failures

**Solution**:
```bash
# Reinstall package
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"

# Verify installation
pip show exo
```

#### 4. Test Failures

**Error**: Validation tests fail

**Diagnosis**:
```bash
# Run tests with verbose output
python -m pytest src/exo/worker/tests/unittests/test_runner/ -v -s

# Check specific test
python -m pytest src/exo/worker/tests/unittests/test_runner/test_shutdown_coordinator.py -v
```

**Solution**:
- Review test output for specific failures
- Check if environment meets test requirements
- Consider running tests in isolation

### Emergency Recovery

If deployment causes system instability:

1. **Immediate Actions**
   ```bash
   # Force stop all EXO processes
   sudo pkill -9 -f exo
   
   # Reset systemd state
   sudo systemctl reset-failed exo
   ```

2. **Emergency Rollback**
   ```bash
   # Use most recent backup
   LATEST_BACKUP=$(ls -1t deployment/backups/ | head -1)
   ./deployment/rollback-race-condition-fix.sh deployment/backups/$LATEST_BACKUP
   ```

3. **System Recovery**
   ```bash
   # Clear any locks or temporary files
   sudo rm -f /var/run/exo/*.lock
   sudo rm -f /tmp/exo-*
   
   # Restart system services if needed
   sudo systemctl daemon-reload
   ```

## Best Practices

### Deployment Planning

1. **Test in Staging First**
   - Deploy to staging environment before production
   - Run comprehensive tests in staging
   - Validate performance under load

2. **Gradual Rollout**
   - For multi-node deployments, update nodes incrementally
   - Monitor each node after update before proceeding
   - Keep some nodes on previous version for comparison

3. **Documentation**
   - Document any customizations or configuration changes
   - Keep deployment logs for future reference
   - Update operational procedures as needed

### Monitoring and Maintenance

1. **Regular Health Checks**
   ```bash
   # Schedule daily validation
   echo "0 6 * * * /path/to/exo/deployment/monitoring/validate-deployment.sh" | crontab -
   ```

2. **Log Rotation**
   ```bash
   # Configure logrotate for EXO logs
   sudo tee /etc/logrotate.d/exo << EOF
   /var/log/exo/*.log {
       daily
       rotate 30
       compress
       delaycompress
       missingok
       notifempty
       create 644 exo exo
   }
   EOF
   ```

3. **Backup Retention**
   ```bash
   # Clean old backups (keep last 10)
   cd deployment/backups
   ls -1t | tail -n +11 | xargs -r rm -rf
   ```

## Support and Documentation

### Additional Resources

- **Operational Procedures**: `deployment/docs/operational-procedures.md`
- **Architecture Documentation**: `docs/architecture.md`
- **API Documentation**: `src/exo/master/api.py`
- **Test Documentation**: `src/exo/worker/tests/README.md`

### Getting Help

1. **Check Logs First**
   ```bash
   journalctl -u exo --since "1 hour ago" -p err
   ```

2. **Run Diagnostics**
   ```bash
   ./deployment/monitoring/validate-deployment.sh
   ```

3. **Collect Support Information**
   ```bash
   # Generate support bundle
   mkdir -p /tmp/exo-support
   journalctl -u exo --since "24 hours ago" > /tmp/exo-support/service.log
   systemctl status exo > /tmp/exo-support/status.txt
   curl -s http://localhost:52415/state > /tmp/exo-support/api-state.json
   ```

### Contact Information

- **GitHub Issues**: [EXO Repository Issues](https://github.com/exo-explore/exo/issues)
- **Documentation**: [EXO Documentation](https://github.com/exo-explore/exo/docs)
- **Community**: [EXO Discussions](https://github.com/exo-explore/exo/discussions)

---

*This deployment guide should be reviewed and updated with each release to ensure accuracy and completeness.*