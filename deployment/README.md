# EXO Multinode Race Condition Fix - Deployment System

This directory contains comprehensive deployment and rollback procedures for the EXO multinode race condition fixes.

## Overview

The deployment system provides automated, reliable deployment and rollback capabilities for the race condition fixes that eliminate:

- **"Queue is closed" errors** during runner shutdown
- **ClosedResourceError exceptions** in multiprocessing communication  
- **Runner failures with exit code 1** during coordination
- **Deadlocks and race conditions** in multi-node scenarios

## Directory Structure

```
deployment/
├── README.md                           # This file
├── deploy-race-condition-fix.sh        # Main deployment script
├── rollback-race-condition-fix.sh      # Rollback script
├── config/
│   └── deployment.conf                 # Deployment configuration
├── docs/
│   ├── deployment-guide.md             # Comprehensive deployment guide
│   └── operational-procedures.md       # Operations manual
├── monitoring/
│   └── validate-deployment.sh          # Deployment validation script
├── logs/                               # Deployment logs (created automatically)
└── backups/                            # Backup storage (created automatically)
```

## Quick Start

### 1. Deploy Race Condition Fixes

```bash
# Run automated deployment
./deployment/deploy-race-condition-fix.sh
```

### 2. Validate Deployment

```bash
# Run validation checks
./deployment/monitoring/validate-deployment.sh
```

### 3. Rollback if Needed

```bash
# List available backups
ls -la deployment/backups/

# Rollback to specific backup
./deployment/rollback-race-condition-fix.sh deployment/backups/20250102_143000
```

## Scripts Overview

### deploy-race-condition-fix.sh

**Purpose**: Automated deployment of race condition fixes

**Features**:
- Environment validation and prerequisite checking
- Automatic backup creation with manifest
- Graceful service shutdown and startup
- Comprehensive validation testing
- Detailed logging and reporting
- Automatic rollback on failure

**Usage**:
```bash
./deployment/deploy-race-condition-fix.sh [options]

Options:
  --backup-only    Create backup without deploying
  --skip-tests     Skip validation tests (not recommended)
  --dry-run        Show what would be done without executing
```

**Output**:
- Deployment logs in `deployment/logs/deploy_YYYYMMDD_HHMMSS.log`
- Backup created in `deployment/backups/YYYYMMDD_HHMMSS/`
- Deployment report in `deployment/logs/deployment_report_YYYYMMDD_HHMMSS.md`

### rollback-race-condition-fix.sh

**Purpose**: Rollback to previous version in case of issues

**Features**:
- Backup validation and integrity checking
- Graceful service management
- File restoration with verification
- Package reinstallation
- Post-rollback validation
- Rollback reporting

**Usage**:
```bash
./deployment/rollback-race-condition-fix.sh <backup_directory>

Example:
./deployment/rollback-race-condition-fix.sh deployment/backups/20250102_143000
```

**Output**:
- Rollback logs in `deployment/logs/rollback_YYYYMMDD_HHMMSS.log`
- Rollback report in `deployment/logs/rollback_report_YYYYMMDD_HHMMSS.md`

### validate-deployment.sh

**Purpose**: Comprehensive deployment validation and monitoring

**Features**:
- Service health checking
- Race condition error detection
- Resource usage monitoring
- Performance metrics collection
- Multi-node functionality testing
- Detailed validation reporting

**Usage**:
```bash
./deployment/monitoring/validate-deployment.sh [options]

Options:
  --continuous     Run continuous monitoring
  --report-only    Generate report without checks
```

**Output**:
- Validation logs in `deployment/logs/validation_YYYYMMDD_HHMMSS.log`
- Validation report in `deployment/logs/validation_report_YYYYMMDD_HHMMSS.md`

## Configuration

### deployment.conf

The main configuration file contains all deployment parameters:

```bash
# Key settings
DEPLOYMENT_TIMEOUT=300
SERVICE_STOP_TIMEOUT=30
VALIDATION_TIMEOUT=120
BACKUP_RETENTION_DAYS=30

# Race condition fix settings
SHUTDOWN_TIMEOUT=30
RESOURCE_CLEANUP_TIMEOUT=10
CHANNEL_DRAIN_TIMEOUT=5
```

### Local Overrides

Create `deployment/config/local.conf` to override default settings:

```bash
# Example local overrides
DEPLOYMENT_TIMEOUT=600  # Longer timeout for slow systems
VERBOSE_LOGGING=true    # Enable detailed logging
SKIP_TESTS=false        # Always run tests
```

## Deployment Workflow

### 1. Pre-Deployment

```bash
# Check current system status
systemctl status exo
curl -f http://localhost:52415/health

# Review deployment configuration
cat deployment/config/deployment.conf

# Ensure adequate resources
df -h
free -h
```

### 2. Deployment Execution

```bash
# Run deployment with logging
./deployment/deploy-race-condition-fix.sh 2>&1 | tee deployment.log

# Monitor progress
tail -f deployment/logs/deploy_*.log
```

### 3. Post-Deployment Validation

```bash
# Immediate validation
./deployment/monitoring/validate-deployment.sh

# Check for race condition errors
journalctl -u exo --since "15 minutes ago" | grep -E "Queue is closed|ClosedResourceError"

# Monitor service health
watch systemctl status exo
```

### 4. Long-term Monitoring

```bash
# Schedule regular validation
echo "0 */6 * * * /path/to/exo/deployment/monitoring/validate-deployment.sh" | crontab -

# Monitor logs for patterns
journalctl -u exo -f | grep -E "ERROR|WARNING|shutdown|coordination"
```

## Rollback Scenarios

### Scenario 1: Deployment Failure

If deployment fails during execution:

```bash
# Automatic rollback is triggered
# Check rollback logs
tail -f deployment/logs/rollback_*.log

# Validate rollback success
./deployment/monitoring/validate-deployment.sh
```

### Scenario 2: Post-Deployment Issues

If issues are discovered after deployment:

```bash
# Find latest backup
ls -1t deployment/backups/ | head -1

# Execute manual rollback
./deployment/rollback-race-condition-fix.sh deployment/backups/$(ls -1t deployment/backups/ | head -1)
```

### Scenario 3: Emergency Rollback

For critical production issues:

```bash
# Force stop services
sudo pkill -9 -f exo

# Emergency rollback
./deployment/rollback-race-condition-fix.sh deployment/backups/$(ls -1t deployment/backups/ | head -1)

# Validate system recovery
systemctl status exo
curl -f http://localhost:52415/health
```

## Monitoring and Alerting

### Health Checks

The deployment system provides comprehensive health monitoring:

```bash
# Service status
systemctl is-active exo

# API endpoints
curl -f http://localhost:52415/health
curl -f http://localhost:52415/state

# Resource usage
ps -o pid,ppid,cmd,%mem,%cpu --sort=-%mem -C python | grep exo
```

### Error Detection

Automated detection of race condition issues:

```bash
# Check for specific errors
journalctl -u exo --since "1 hour ago" | grep -E "Queue is closed|ClosedResourceError|Runner.*exit.*code.*1"

# Monitor coordination logs
journalctl -u exo -f | grep -i "coordination\|shutdown"
```

### Performance Monitoring

Track system performance after deployment:

```bash
# Response time monitoring
for i in {1..10}; do
  time curl -s http://localhost:52415/health > /dev/null
done

# Memory leak detection
ps -o pid,ppid,cmd,rss --sort=-rss -C python | grep exo
```

## Troubleshooting

### Common Issues

#### 1. Permission Errors

```bash
# Fix file permissions
sudo chown -R $(whoami):$(whoami) deployment/
chmod +x deployment/*.sh deployment/monitoring/*.sh
```

#### 2. Service Won't Start

```bash
# Check service configuration
systemctl cat exo

# Reset service state
sudo systemctl reset-failed exo
sudo systemctl daemon-reload
```

#### 3. Validation Failures

```bash
# Run tests individually
python -m pytest src/exo/worker/tests/unittests/test_runner/ -v

# Check specific components
python -c "from exo.worker.runner.shutdown_coordinator import ShutdownCoordinator; print('Import successful')"
```

### Log Analysis

Key log files to examine:

```bash
# Deployment logs
ls -la deployment/logs/deploy_*.log

# Service logs
journalctl -u exo --since "1 hour ago"

# System logs
dmesg | tail -50
```

### Recovery Procedures

If deployment system itself has issues:

```bash
# Reset deployment directory permissions
sudo chown -R $(whoami):$(whoami) deployment/
find deployment/ -name "*.sh" -exec chmod +x {} \;

# Clear temporary files
rm -f deployment/logs/*.tmp
rm -f deployment/backups/*.tmp

# Validate scripts
bash -n deployment/deploy-race-condition-fix.sh
bash -n deployment/rollback-race-condition-fix.sh
```

## Best Practices

### Before Deployment

1. **Test in staging environment first**
2. **Schedule during maintenance windows**
3. **Notify stakeholders of planned changes**
4. **Ensure adequate system resources**
5. **Review recent system changes**

### During Deployment

1. **Monitor deployment progress closely**
2. **Keep rollback plan ready**
3. **Have emergency contacts available**
4. **Document any issues encountered**

### After Deployment

1. **Run comprehensive validation**
2. **Monitor for 24-48 hours minimum**
3. **Check performance metrics**
4. **Update documentation as needed**
5. **Schedule follow-up reviews**

## Security Considerations

### File Permissions

```bash
# Secure deployment scripts
chmod 750 deployment/*.sh
chmod 750 deployment/monitoring/*.sh

# Secure configuration files
chmod 640 deployment/config/*.conf

# Secure log files
chmod 640 deployment/logs/*.log
```

### Service Security

```bash
# Run service as dedicated user
sudo useradd -r -s /bin/false exo

# Set proper ownership
sudo chown -R exo:exo /opt/exo
```

### Network Security

```bash
# Restrict API access
ufw allow from 192.168.1.0/24 to any port 52415

# Monitor network connections
netstat -tlnp | grep exo
```

## Support and Documentation

### Additional Resources

- **Deployment Guide**: `deployment/docs/deployment-guide.md`
- **Operational Procedures**: `deployment/docs/operational-procedures.md`
- **Architecture Documentation**: `docs/architecture.md`
- **API Documentation**: `src/exo/master/api.py`

### Getting Help

1. **Check deployment logs first**
2. **Run validation script for diagnostics**
3. **Review troubleshooting section**
4. **Collect system information for support**

### Contact Information

- **GitHub Issues**: [EXO Repository](https://github.com/exo-explore/exo/issues)
- **Documentation**: [EXO Docs](https://github.com/exo-explore/exo/docs)
- **Community**: [EXO Discussions](https://github.com/exo-explore/exo/discussions)

---

*This deployment system is designed to provide reliable, automated deployment and rollback capabilities for the EXO multinode race condition fixes. Regular testing and validation ensure production readiness.*