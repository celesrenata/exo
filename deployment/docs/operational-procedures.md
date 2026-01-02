# EXO Multinode Race Condition Fix - Operational Procedures

This document provides comprehensive operational procedures for system administrators managing the EXO distributed AI inference system with the multinode race condition fixes.

## Overview

The multinode race condition fixes implement a three-phase shutdown protocol with enhanced resource management to eliminate "Queue is closed" and "ClosedResourceError" exceptions during multi-node operations.

### Key Components
- **Shutdown Coordinator**: Orchestrates graceful shutdown across all runner processes
- **Resource Manager**: Manages lifecycle of multiprocessing resources with proper ordering
- **Channel Manager**: Provides race-condition-free multiprocessing communication
- **Enhanced Runner Supervisor**: Coordinates runner lifecycle with improved error handling

## Deployment Procedures

### Pre-Deployment Checklist

1. **Environment Validation**
   ```bash
   # Check Python version (requires 3.13+)
   python3 --version
   
   # Verify EXO installation
   which exo
   
   # Check system resources
   free -h
   df -h
   ```

2. **Backup Current Installation**
   ```bash
   # Create backup directory
   mkdir -p /opt/exo/backups/$(date +%Y%m%d_%H%M%S)
   
   # Backup configuration
   cp -r /etc/exo/ /opt/exo/backups/$(date +%Y%m%d_%H%M%S)/
   
   # Backup service files
   cp /etc/systemd/system/exo.service /opt/exo/backups/$(date +%Y%m%d_%H%M%S)/
   ```

3. **Service Health Check**
   ```bash
   # Check current service status
   systemctl status exo
   
   # Verify endpoints are responding
   curl -f http://localhost:52415/health
   curl -f http://localhost:52415/state
   ```

### Deployment Process

1. **Run Deployment Script**
   ```bash
   cd /path/to/exo
   ./deployment/deploy-race-condition-fix.sh
   ```

2. **Monitor Deployment Progress**
   ```bash
   # Follow deployment logs
   tail -f deployment/logs/deploy_*.log
   
   # Monitor service status
   watch systemctl status exo
   ```

3. **Post-Deployment Validation**
   ```bash
   # Run validation script
   ./deployment/monitoring/validate-deployment.sh
   
   # Check for race condition errors
   journalctl -u exo --since "10 minutes ago" | grep -E "Queue is closed|ClosedResourceError"
   ```

## Rollback Procedures

### Emergency Rollback

If critical issues are detected during or after deployment:

1. **Immediate Rollback**
   ```bash
   # Find latest backup
   ls -la deployment/backups/
   
   # Execute rollback (replace with actual backup directory)
   ./deployment/rollback-race-condition-fix.sh deployment/backups/20250102_143000
   ```

2. **Verify Rollback Success**
   ```bash
   # Check service status
   systemctl status exo
   
   # Verify endpoints
   curl -f http://localhost:52415/health
   
   # Monitor logs for stability
   journalctl -u exo -f
   ```

### Planned Rollback

For planned rollbacks during maintenance windows:

1. **Schedule Maintenance Window**
   - Notify users of planned downtime
   - Ensure no critical inference jobs are running

2. **Execute Rollback**
   ```bash
   # Stop service gracefully
   systemctl stop exo
   
   # Run rollback script
   ./deployment/rollback-race-condition-fix.sh <backup_directory>
   
   # Validate rollback
   ./deployment/monitoring/validate-deployment.sh
   ```

## Monitoring and Maintenance

### Daily Monitoring Tasks

1. **Service Health Check**
   ```bash
   # Check service status
   systemctl status exo
   
   # Verify endpoints
   curl -f http://localhost:52415/health
   curl -f http://localhost:52415/state
   ```

2. **Error Log Review**
   ```bash
   # Check for race condition errors (last 24 hours)
   journalctl -u exo --since "24 hours ago" | grep -E "Queue is closed|ClosedResourceError|Runner.*exit.*code.*1"
   
   # Review general error patterns
   journalctl -u exo --since "24 hours ago" -p err
   ```

3. **Resource Usage Monitoring**
   ```bash
   # Check memory usage
   ps -o pid,ppid,cmd,%mem --sort=-%mem -C python | grep exo
   
   # Check CPU usage
   ps -o pid,ppid,cmd,%cpu --sort=-%cpu -C python | grep exo
   
   # Check file descriptors
   ls -1 /proc/$(pgrep -f "exo" | head -1)/fd | wc -l
   ```

### Weekly Monitoring Tasks

1. **Performance Analysis**
   ```bash
   # Run comprehensive validation
   ./deployment/monitoring/validate-deployment.sh
   
   # Analyze response times
   for i in {1..10}; do
     time curl -s http://localhost:52415/health > /dev/null
   done
   ```

2. **Log Rotation and Cleanup**
   ```bash
   # Rotate EXO logs
   journalctl --vacuum-time=30d
   
   # Clean old deployment logs
   find deployment/logs/ -name "*.log" -mtime +30 -delete
   ```

3. **Backup Verification**
   ```bash
   # Verify backup integrity
   ls -la deployment/backups/
   
   # Test backup restoration (in test environment)
   ./deployment/rollback-race-condition-fix.sh deployment/backups/latest
   ```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. "Queue is closed" Errors

**Symptoms:**
- Errors in logs: `ValueError: Queue is closed`
- Runner processes failing with exit code 1

**Diagnosis:**
```bash
# Check for recent occurrences
journalctl -u exo --since "1 hour ago" | grep "Queue is closed"

# Check shutdown coordination logs
journalctl -u exo --since "1 hour ago" | grep -i "shutdown\|coordination"
```

**Resolution:**
1. Verify the race condition fixes are properly deployed
2. Check if the three-phase shutdown protocol is active
3. Monitor resource cleanup order in logs

#### 2. ClosedResourceError Exceptions

**Symptoms:**
- Errors in logs: `ClosedResourceError`
- Multiprocessing communication failures

**Diagnosis:**
```bash
# Check resource management logs
journalctl -u exo --since "1 hour ago" | grep -i "resource\|cleanup"

# Verify channel manager status
journalctl -u exo --since "1 hour ago" | grep -i "channel"
```

**Resolution:**
1. Restart EXO service to reset resource state
2. Check for resource leaks using monitoring tools
3. Verify proper dependency ordering in cleanup

#### 3. High Memory Usage

**Symptoms:**
- Memory usage > 80%
- System becoming unresponsive

**Diagnosis:**
```bash
# Check memory usage by process
ps -o pid,ppid,cmd,%mem,rss --sort=-rss -C python | grep exo

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python -m exo
```

**Resolution:**
1. Restart EXO service to free memory
2. Check for resource cleanup issues
3. Monitor memory usage over time

#### 4. Service Won't Start

**Symptoms:**
- `systemctl start exo` fails
- Service shows failed status

**Diagnosis:**
```bash
# Check service logs
journalctl -u exo --since "10 minutes ago"

# Check configuration
systemctl cat exo

# Verify binary exists
ls -la $(which exo)
```

**Resolution:**
1. Check configuration file syntax
2. Verify all dependencies are installed
3. Check file permissions and ownership

### Emergency Procedures

#### Complete System Recovery

If EXO is completely unresponsive:

1. **Force Stop All Processes**
   ```bash
   # Kill all EXO processes
   pkill -9 -f exo
   
   # Clean up any remaining resources
   ipcs -q | grep $(id -u exo) | awk '{print $2}' | xargs -r ipcrm -q
   ```

2. **Reset Service State**
   ```bash
   # Reset systemd service
   systemctl reset-failed exo
   systemctl daemon-reload
   
   # Clear any locks
   rm -f /var/run/exo/*.lock
   ```

3. **Restore from Backup**
   ```bash
   # Find most recent stable backup
   ls -la deployment/backups/ | grep -v rollback
   
   # Execute rollback
   ./deployment/rollback-race-condition-fix.sh <backup_directory>
   ```

## Configuration Management

### Service Configuration

The EXO service configuration includes race condition fix parameters:

```ini
[Unit]
Description=EXO distributed AI inference system
After=network.target

[Service]
Type=simple
User=exo
Group=exo
ExecStart=/usr/local/bin/exo --verbose
Restart=always
RestartSec=10

# Race condition fix environment variables
Environment="EXO_SHUTDOWN_TIMEOUT=30"
Environment="EXO_RESOURCE_CLEANUP_TIMEOUT=10"
Environment="EXO_CHANNEL_DRAIN_TIMEOUT=5"

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
```

### Environment Variables

Key environment variables for race condition fixes:

- `EXO_SHUTDOWN_TIMEOUT`: Maximum time for graceful shutdown (default: 30s)
- `EXO_RESOURCE_CLEANUP_TIMEOUT`: Timeout for resource cleanup (default: 10s)
- `EXO_CHANNEL_DRAIN_TIMEOUT`: Timeout for channel draining (default: 5s)
- `EXO_DEBUG_COORDINATION`: Enable debug logging for coordination (default: false)

## Performance Tuning

### Optimization Settings

For high-performance deployments:

```bash
# Increase file descriptor limits
echo "exo soft nofile 65536" >> /etc/security/limits.conf
echo "exo hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "vm.max_map_count = 262144" >> /etc/sysctl.conf

# Apply changes
sysctl -p
```

### Resource Allocation

Recommended resource allocation:

- **Memory**: Minimum 8GB, recommended 16GB+ for large models
- **CPU**: Minimum 4 cores, recommended 8+ cores for multi-node
- **Storage**: SSD recommended, minimum 100GB free space
- **Network**: Gigabit Ethernet for multi-node communication

## Security Considerations

### Access Control

1. **Service User**
   ```bash
   # Create dedicated service user
   useradd -r -s /bin/false exo
   
   # Set proper ownership
   chown -R exo:exo /opt/exo
   chmod 750 /opt/exo
   ```

2. **File Permissions**
   ```bash
   # Secure configuration files
   chmod 640 /etc/exo/*.conf
   chown root:exo /etc/exo/*.conf
   
   # Secure log files
   chmod 640 /var/log/exo/*.log
   chown exo:exo /var/log/exo/*.log
   ```

3. **Network Security**
   ```bash
   # Configure firewall (adjust ports as needed)
   ufw allow 52415/tcp  # EXO API/Dashboard
   ufw allow from 192.168.1.0/24 to any port 52415  # Restrict to local network
   ```

## Compliance and Auditing

### Audit Logging

Enable comprehensive audit logging:

```bash
# Configure rsyslog for EXO
echo "if \$programname == 'exo' then /var/log/exo/audit.log" >> /etc/rsyslog.d/50-exo.conf
systemctl restart rsyslog
```

### Compliance Checks

Regular compliance verification:

```bash
# Check service configuration
systemctl show exo

# Verify file permissions
find /opt/exo -type f -exec ls -la {} \;

# Check network configuration
netstat -tlnp | grep exo
```

## Support and Escalation

### Log Collection for Support

When contacting support, collect these logs:

```bash
# Create support bundle
mkdir -p /tmp/exo-support-$(date +%Y%m%d_%H%M%S)
cd /tmp/exo-support-$(date +%Y%m%d_%H%M%S)

# Collect system information
uname -a > system-info.txt
free -h > memory-info.txt
df -h > disk-info.txt

# Collect EXO logs
journalctl -u exo --since "24 hours ago" > exo-service.log
cp /var/log/exo/*.log .

# Collect configuration
cp -r /etc/exo/ .
systemctl cat exo > service-config.txt

# Create archive
cd ..
tar -czf exo-support-$(date +%Y%m%d_%H%M%S).tar.gz exo-support-$(date +%Y%m%d_%H%M%S)/
```

### Escalation Contacts

- **Level 1**: System Administrator (routine issues)
- **Level 2**: EXO Development Team (race condition issues)
- **Level 3**: Infrastructure Team (system-level issues)

### Emergency Contacts

For critical production issues:
- On-call Engineer: [contact information]
- Development Team Lead: [contact information]
- Infrastructure Manager: [contact information]

---

*This document should be reviewed and updated regularly to reflect changes in the EXO system and operational procedures.*