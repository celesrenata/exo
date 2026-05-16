"""
Cluster Monitoring for Gremlin Cluster Tests

This module provides comprehensive monitoring capabilities for cluster tests,
collecting metrics from Prometheus and InfluxDB2 for:
- Intel GPU metrics (utilization, memory, temperature)
- CPU metrics (utilization, load, temperature)
- Memory metrics (usage, swap)
- Networking metrics (throughput, latency, errors)

Metrics are collected every 15 seconds during tests and compiled into
detailed reports to help identify performance bottlenecks.

Requirements addressed:
- 9.1: System monitoring and metrics collection
- 9.2: Performance metrics tracking
- 9.3: GPU utilization and memory monitoring
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

# Try to import loguru, fall back to stdlib logging if not available
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Try to import Prometheus and InfluxDB clients
try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available. Install with: pip install prometheus_client")

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    logger.warning("influxdb_client not available. Install with: pip install influxdb-client")

# Default cluster URLs (for local development with kubectl port-forward)
# When using kubectl port-forward, use localhost:
#   kubectl port-forward -n prometheus-service svc/prometheus-kube-prometheus-prometheus 9090:9090
#   kubectl port-forward -n influxdb-service svc/influxdb-influxdb2 8086:80
# NOTE: The port-forward maps local 8086 to cluster port 80 (plain HTTP), NOT HTTPS
DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:9090"
DEFAULT_INFLUXDB_URL = "http://127.0.0.1:8086"
DEFAULT_INFLUXDB_ORG = "exo"
DEFAULT_INFLUXDB_BUCKET = "exo_metrics"
# Default InfluxDB token for local development.
# NOTE: This is a password from old test notes, NOT a valid InfluxDB v2 API token.
# For actual InfluxDB v2, fetch the admin-token from the kubernetes secret:
#   kubectl get secret -n influxdb-service influxdb-influxdb2-auth -o jsonpath='{.data.admin-token}' | base64 -d
# Do not commit the decoded admin-token; provide it via INFLUXDB_TOKEN.
DEFAULT_INFLUXDB_TOKEN = "PSCh4ng3me!"


def _is_local_influxdb_url(url: str) -> bool:
    """
    Check if the InfluxDB URL is a local connection (127.0.0.1 or localhost on port 8086).
    
    Args:
        url: The InfluxDB URL to check.
        
    Returns:
        True if the URL is local, False otherwise.
    """
    if not url:
        return False
    
    import re
    # Match http:// or https:// with 127.0.0.1 or localhost on port 8086
    pattern = r"^https?://(127\.0\.0\.1|localhost):8086"
    return bool(re.match(pattern, url, re.IGNORECASE))


def _normalize_influxdb_url_for_port_forward(url: str) -> str:
    """
    Normalize InfluxDB URL for local port-forward usage.
    
    When using kubectl port-forward for local development, the command:
        kubectl port-forward -n influxdb-service svc/influxdb-influxdb2 8086:80
    maps local port 8086 to cluster port 80 (plain HTTP), NOT HTTPS.
    
    This function detects when a user mistakenly uses https:// with localhost/127.0.0.1:8086
    and converts it to http:// to prevent SSL errors.
    
    Args:
        url: The InfluxDB URL to normalize.
        
    Returns:
        The normalized URL (http:// for local port-forward, unchanged for remote URLs).
        
    Examples:
        >>> _normalize_influxdb_url_for_port_forward("https://127.0.0.1:8086")
        'http://127.0.0.1:8086'
        >>> _normalize_influxdb_url_for_port_forward("https://localhost:8086")
        'http://localhost:8086'
        >>> _normalize_influxdb_url_for_port_forward("https://us-east-1-1.aws.cloud2.influxdata.com")
        'https://us-east-1-1.aws.cloud2.influxdata.com'
        >>> _normalize_influxdb_url_for_port_forward("http://127.0.0.1:8086")
        'http://127.0.0.1:8086'
    """
    if not url:
        return url
    
    # Only normalize localhost/127.0.0.1 URLs on port 8086
    import re
    pattern = r"^https://(127\.0\.0\.1|localhost):8086"
    if re.match(pattern, url, re.IGNORECASE):
        # Convert https:// to http:// for local port-forward
        normalized = re.sub(r"^https://", "http://", url, flags=re.IGNORECASE)
        logger.warning(
            f"INFLUXDB_URL uses https:// with localhost port-forward. "
            f"Converting to http:// (kubectl port-forward 8086:80 uses plain HTTP). "
            f"URL: {url} -> {normalized}"
        )
        return normalized
    
    # Return unchanged for remote URLs (e.g., Influx Cloud)
    return url


def _mask_token(token: str) -> str:
    """
    Mask a token for logging, showing only first 4 and last 4 characters.
    
    Args:
        token: The token to mask.
        
    Returns:
        Masked token string.
    """
    if not token or len(token) < 8:
        return "****"
    return f"{token[:4]}****{token[-4:]}"


def _get_influxdb_token_for_local(url: str, token: str) -> str:
    """
    Get the InfluxDB token for local connections.
    
    For local InfluxDB (127.0.0.1:8086 or localhost:8086), the actual API token
    is stored in the Kubernetes secret `influxdb-influxdb2-auth` under key `admin-token`.
    The default token 'PSCh4ng3me!' is a password from old test notes, NOT a valid
    InfluxDB v2 API token.
    
    This function returns the provided token unchanged if it's non-empty.
    If the token is empty, it returns an empty string (not a password).
    
    To fetch the real admin token from the cluster:
        kubectl get secret -n influxdb-service influxdb-influxdb2-auth -o jsonpath='{.data.admin-token}' | base64 -d
    
    Args:
        url: The InfluxDB URL.
        token: The InfluxDB token.
        
    Returns:
        The token to use (unchanged if non-empty, empty string otherwise).
    """
    if not _is_local_influxdb_url(url):
        # For remote/cloud URLs, keep the provided token
        return token
    
    # For local connections, keep any non-empty token provided by the user
    # Do NOT replace with DEFAULT_INFLUXDB_TOKEN (which is a password, not an API token)
    if token:
        return token
    
    # Empty token - do not replace with password; let the user know how to fetch the real token
    return ""


class MetricSource(Enum):
    """Sources of metrics data."""
    PROMETHEUS = "prometheus"
    INFLUXDB = "influxdb"
    LOCAL = "local"  # Collected directly from the system


@dataclass
class GPUMetrics:
    """GPU metrics for a single device."""
    timestamp: float
    device_id: int
    device_name: str
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_utilization_percent: float = 0.0
    temperature_celsius: float = 0.0
    power_watts: float = 0.0
    clock_speed_mhz: float = 0.0
    fan_speed_percent: float = 0.0
    pcie_throughput_mb_s: float = 0.0


@dataclass
class CPUMetrics:
    """CPU metrics."""
    timestamp: float
    cpu_percent: float = 0.0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0
    context_switches: int = 0
    interrupts: int = 0
    soft_interrupts: int = 0
    process_count: int = 0
    thread_count: int = 0


@dataclass
class MemoryMetrics:
    """Memory metrics."""
    timestamp: float
    total_mb: float = 0.0
    available_mb: float = 0.0
    used_mb: float = 0.0
    free_mb: float = 0.0
    utilization_percent: float = 0.0
    swap_total_mb: float = 0.0
    swap_used_mb: float = 0.0
    swap_free_mb: float = 0.0
    swap_utilization_percent: float = 0.0
    buffers_mb: float = 0.0
    cached_mb: float = 0.0


@dataclass
class NetworkMetrics:
    """Network interface metrics."""
    timestamp: float
    interface_name: str
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    errors_sent: int = 0
    errors_received: int = 0
    dropped_sent: int = 0
    dropped_received: int = 0
    speed_mbps: int = 0
    duplex: str = "unknown"
    link_up: bool = False


@dataclass
class ClusterMetricsSnapshot:
    """Complete metrics snapshot for a cluster node."""
    timestamp: float
    node_id: str
    node_ip: str
    gpu_metrics: list[GPUMetrics] = field(default_factory=list)
    cpu_metrics: CPUMetrics | None = None
    memory_metrics: MemoryMetrics | None = None
    network_metrics: list[NetworkMetrics] = field(default_factory=list)
    inference_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsAggregates:
    """Aggregated metrics statistics."""
    gpu_utilization_min: float = 0.0
    gpu_utilization_max: float = 0.0
    gpu_utilization_avg: float = 0.0
    gpu_utilization_p95: float = 0.0
    gpu_memory_used_mb_min: float = 0.0
    gpu_memory_used_mb_max: float = 0.0
    gpu_memory_used_mb_avg: float = 0.0
    cpu_utilization_min: float = 0.0
    cpu_utilization_max: float = 0.0
    cpu_utilization_avg: float = 0.0
    memory_utilization_min: float = 0.0
    memory_utilization_max: float = 0.0
    memory_utilization_avg: float = 0.0
    network_throughput_mbps_min: float = 0.0
    network_throughput_mbps_max: float = 0.0
    network_throughput_mbps_avg: float = 0.0
    inference_tps_min: float = 0.0
    inference_tps_max: float = 0.0
    inference_tps_avg: float = 0.0
    inference_latency_ms_min: float = 0.0
    inference_latency_ms_max: float = 0.0
    inference_latency_ms_avg: float = 0.0


class ClusterMonitor:
    """
    Monitors cluster metrics during tests.
    
    Collects metrics from Prometheus and InfluxDB2, aggregates them,
    and provides detailed reports to identify performance bottlenecks.
    
    Args:
        prometheus_url: URL for Prometheus server (e.g., http://localhost:9090)
        influxdb_url: URL for InfluxDB2 server (e.g., https://us-east-1-1.aws.cloud2.influxdata.com)
        influxdb_token: InfluxDB2 authentication token
        influxdb_org: InfluxDB2 organization name
        influxdb_bucket: InfluxDB2 bucket name
        collection_interval: Interval between metric collections in seconds (default: 15)
    """
    
    def __init__(
        self,
        prometheus_url: str | None = None,
        influxdb_url: str | None = None,
        influxdb_token: str | None = None,
        influxdb_org: str | None = None,
        influxdb_bucket: str | None = None,
        collection_interval: float = 15.0,
    ):
        self.prometheus_url = prometheus_url or os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
        # Normalize InfluxDB URL to handle local port-forward (https -> http for localhost:8086)
        self.influxdb_url = _normalize_influxdb_url_for_port_forward(
            influxdb_url or os.environ.get("INFLUXDB_URL", "http://localhost:8086")
        )
        # Get token for local connections (never replaces non-empty user-provided tokens)
        raw_token = influxdb_token or os.environ.get("INFLUXDB_TOKEN", "")
        self.influxdb_token = _get_influxdb_token_for_local(self.influxdb_url, raw_token)
        self.influxdb_org = influxdb_org or os.environ.get("INFLUXDB_ORG", "exo")
        self.influxdb_bucket = influxdb_bucket or os.environ.get("INFLUXDB_BUCKET", "exo_metrics")
        self.collection_interval = collection_interval
        
        self._client: httpx.AsyncClient | None = None
        self._influxdb_client: Any | None = None
        self._influxdb_write_api = None
        self._metrics_history: list[ClusterMetricsSnapshot] = []
        self._running = False
        self._collection_task: asyncio.Task | None = None
        self._influxdb_unauthorized = False  # Track if InfluxDB auth has failed
        self._prometheus_unreachable = False  # Track if Prometheus is unreachable
        
        # Metrics for Prometheus
        self._gpu_utilization_gauge = None
        self._cpu_utilization_gauge = None
        self._memory_utilization_gauge = None
        self._inference_tps_gauge = None
        
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self) -> None:
        """Set up Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._gpu_utilization_gauge = Gauge(
            "exo_gpu_utilization_percent",
            "GPU utilization percentage",
            ["node_id", "device_id", "device_name"]
        )
        self._cpu_utilization_gauge = Gauge(
            "exo_cpu_utilization_percent",
            "CPU utilization percentage",
            ["node_id"]
        )
        self._memory_utilization_gauge = Gauge(
            "exo_memory_utilization_percent",
            "Memory utilization percentage",
            ["node_id"]
        )
        self._inference_tps_gauge = Gauge(
            "exo_inference_tokens_per_second",
            "Inference throughput in tokens per second",
            ["node_id", "model_id"]
        )
    
    async def __aenter__(self) -> ClusterMonitor:
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
    
    async def _validate_influxdb_auth(self) -> bool:
        """
        Validate InfluxDB authentication with a lightweight health check.
        
        Returns:
            True if authentication succeeds, False otherwise.
        """
        if not self._influxdb_client or not INFLUXDB_AVAILABLE:
            return False
        
        try:
            # Perform a lightweight health check using the buckets API
            buckets = self._influxdb_client.buckets_api().find_buckets()
            # Check if buckets object is truthy (has items) without using len()
            if buckets:
                # If we get a response (even empty), auth is valid
                return True
        except Exception as e:
            # Check if it's an unauthorized error
            error_str = str(e)
            if "unauthorized" in error_str.lower() or "401" in error_str:
                logger.error(
                    f"InfluxDB authentication failed (401 Unauthorized) for {self.influxdb_url}. "
                    f"Token: {_mask_token(self.influxdb_token)}, Org: {self.influxdb_org}, "
                    f"Bucket: {self.influxdb_bucket}. "
                    f"Local InfluxDB admin-token can be fetched with: "
                    f"kubectl get secret -n influxdb-service influxdb-influxdb2-auth -o jsonpath='{{.data.admin-token}}' | base64 -d"
                )
                return False
            # Connection refused or other network errors mean the service is unreachable
            if "connection refused" in error_str.lower() or "errno 61" in error_str.lower() or "errno 111" in error_str.lower():
                logger.warning(
                    f"InfluxDB is unreachable at {self.influxdb_url}: {e}. "
                    f"If running locally, start port-forward: "
                    f"kubectl port-forward -n influxdb-service svc/influxdb-influxdb2 8086:80"
                )
                return False
            # Other errors may be transient, log but don't disable
            logger.warning(f"InfluxDB health check failed (non-auth): {e}")
            return True  # Don't disable on non-auth errors
        
        return True
    
    async def start(self) -> None:
        """Start the metrics collector."""
        if self._running:
            return
        
        self._running = True
        self._client = httpx.AsyncClient(timeout=30.0)
        
        # Setup InfluxDB client if credentials provided and client is available
        if self.influxdb_token and INFLUXDB_AVAILABLE:
            try:
                self._influxdb_client = InfluxDBClient(
                    url=self.influxdb_url,
                    token=self.influxdb_token,
                    org=self.influxdb_org,
                )
                self._influxdb_write_api = self._influxdb_client.write_api(
                    write_options=SYNCHRONOUS
                )
                # Validate authentication
                if await self._validate_influxdb_auth():
                    logger.info(f"Connected to InfluxDB at {self.influxdb_url}")
                else:
                    # Auth failed, disable InfluxDB writes
                    self._influxdb_write_api = None
                    self._influxdb_client = None
                    self._influxdb_unauthorized = True
            except Exception as e:
                error_str = str(e)
                if "unauthorized" in error_str.lower() or "401" in error_str:
                    logger.error(
                        f"InfluxDB authentication failed (401 Unauthorized) for {self.influxdb_url}. "
                        f"Token: {_mask_token(self.influxdb_token)}, Org: {self.influxdb_org}, "
                        f"Bucket: {self.influxdb_bucket}. "
                        f"Local InfluxDB admin-token can be fetched with: "
                        f"kubectl get secret -n influxdb-service influxdb-influxdb2-auth -o jsonpath='{{.data.admin-token}}' | base64 -d"
                    )
                    self._influxdb_unauthorized = True
                else:
                    logger.warning(f"Failed to connect to InfluxDB: {e}")
        elif self.influxdb_token and not INFLUXDB_AVAILABLE:
            logger.warning(
                "InfluxDB client package unavailable; skipping InfluxDB writes. "
                "Install influxdb-client: pip install influxdb-client"
            )
        
        # Pre-check Prometheus connectivity
        try:
            check_client = httpx.AsyncClient(timeout=5.0)
            try:
                resp = await check_client.get(f"{self.prometheus_url}/-/healthy", timeout=5.0)
                if resp.status_code == 200:
                    logger.info(f"Prometheus reachable at {self.prometheus_url}")
                else:
                    logger.warning(
                        f"Prometheus returned status {resp.status_code} at {self.prometheus_url}. "
                        f"Metrics collection will skip Prometheus queries."
                    )
                    self._prometheus_unreachable = True
            except Exception as e:
                error_str = str(e)
                if "connection refused" in error_str.lower() or "errno 61" in error_str.lower() or "errno 111" in error_str.lower():
                    logger.warning(
                        f"Prometheus is unreachable at {self.prometheus_url}. "
                        f"Metrics collection will skip Prometheus queries. "
                        f"If running locally, start port-forward: "
                        f"kubectl port-forward -n prometheus-service svc/prometheus-kube-prometheus-prometheus 9090:9090"
                    )
                else:
                    logger.warning(f"Prometheus connectivity check failed: {e}")
                self._prometheus_unreachable = True
            finally:
                await check_client.aclose()
        except Exception:
            self._prometheus_unreachable = True

        # Start metrics collection loop
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info(f"Started metrics collection (interval: {self.collection_interval}s)")
    
    async def stop(self) -> None:
        """Stop the metrics collector."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        if self._client:
            await self._client.aclose()
        
        if self._influxdb_client and INFLUXDB_AVAILABLE:
            self._influxdb_client.close()
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                snapshot = await self.collect_snapshot()
                if snapshot:
                    self._metrics_history.append(snapshot)
                    await self._publish_metrics(snapshot)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            await asyncio.sleep(self.collection_interval)
    
    async def collect_snapshot(self) -> ClusterMetricsSnapshot | None:
        """
        Collect a complete metrics snapshot.
        
        Returns:
            ClusterMetricsSnapshot with all available metrics, or None if collection failed.
        """
        # Skip Prometheus queries if unreachable
        if self._prometheus_unreachable:
            gpu_metrics: list[GPUMetrics] = []
            cpu_metrics: list[CPUMetrics] = []
            memory_metrics: list[MemoryMetrics] = []
            network_metrics: list[NetworkMetrics] = []
        else:
            # Query Prometheus for GPU metrics
            gpu_metrics = await self._query_gpu_metrics()
            
            # Query Prometheus for CPU metrics
            cpu_metrics = await self._query_cpu_metrics()
            
            # Query Prometheus for memory metrics
            memory_metrics = await self._query_memory_metrics()
            
            # Query Prometheus for network metrics
            network_metrics = await self._query_network_metrics()
        
        # Query InfluxDB for inference metrics (skip if unauthorized or unreachable)
        if self._influxdb_unauthorized or not self._influxdb_client:
            inference_metrics: dict[str, Any] = {}
        else:
            inference_metrics = await self._query_inference_metrics()
        
        if not gpu_metrics and not cpu_metrics and not memory_metrics and not network_metrics:
            return None
        
        # Create snapshot with collected metrics
        snapshot = ClusterMetricsSnapshot(
            timestamp=time.time(),
            node_id="unknown",
            node_ip="unknown",
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics[0] if cpu_metrics else None,
            memory_metrics=memory_metrics[0] if memory_metrics else None,
            network_metrics=network_metrics,
            inference_metrics=inference_metrics,
        )
        
        return snapshot
    
    async def _query_gpu_metrics(self) -> list[GPUMetrics]:
        """Query Prometheus for GPU metrics."""
        metrics = []
        
        # Query Intel GPU metrics from dcgm-exporter
        gpu_queries = [
            ("dcgm_gpu_utilization", "GPU utilization percent"),
            ("dcgm_memory_used", "Memory used in bytes"),
            ("dcgm_memory_total", "Total memory in bytes"),
            ("dcgm_temperature", "Temperature in Celsius"),
            ("dcgm_power", "Power usage in watts"),
        ]
        
        for metric_name, description in gpu_queries:
            try:
                url = f"{self.prometheus_url}/api/v1/query"
                params = {"query": metric_name}
                resp = await self._client.get(url, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("status") == "success":
                    results = data.get("data", {}).get("result", [])
                    for result in results:
                        metric = result.get("metric", {})
                        value = result.get("value", [])
                        
                        if len(value) >= 2:
                            timestamp = float(value[0])
                            value_float = float(value[1])
                            
                            device_id = int(metric.get("device", "0"))
                            device_name = metric.get("instance", f"GPU {device_id}")
                            
                            # Map metric name to attribute
                            attr = ""
                            if "utilization" in metric_name:
                                attr = "utilization_percent"
                            elif "memory_used" in metric_name:
                                attr = "memory_used_mb"
                                value_float = value_float / (1024 * 1024)
                            elif "memory_total" in metric_name:
                                attr = "memory_total_mb"
                                value_float = value_float / (1024 * 1024)
                            elif "temperature" in metric_name:
                                attr = "temperature_celsius"
                            elif "power" in metric_name:
                                attr = "power_watts"
                            
                            # Find or create GPU metric for this device
                            gpu = next((g for g in metrics if g.device_id == device_id), None)
                            if gpu is None:
                                gpu = GPUMetrics(
                                    timestamp=timestamp,
                                    device_id=device_id,
                                    device_name=device_name,
                                )
                                metrics.append(gpu)
                            
                            # Set the attribute
                            setattr(gpu, attr, value_float)
            except Exception as e:
                logger.debug(f"Failed to query {metric_name}: {e}")
        
        return metrics
    
    async def _query_cpu_metrics(self) -> list[CPUMetrics]:
        """Query Prometheus for CPU metrics."""
        metrics = []
        
        cpu_queries = [
            ("node_cpu_seconds_total", "CPU time in seconds"),
            ("node_load1", "1-minute load average"),
            ("node_load5", "5-minute load average"),
            ("node_load15", "15-minute load average"),
        ]
        
        for metric_name, description in cpu_queries:
            try:
                url = f"{self.prometheus_url}/api/v1/query"
                params = {"query": metric_name}
                resp = await self._client.get(url, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("status") == "success":
                    results = data.get("data", {}).get("result", [])
                    for result in results:
                        metric = result.get("metric", {})
                        value = result.get("value", [])
                        
                        if len(value) >= 2:
                            timestamp = float(value[0])
                            value_float = float(value[1])
                            
                            instance = metric.get("instance", "unknown")
                            
                            # Find or create CPU metric for this instance
                            cpu = next((c for c in metrics if c.timestamp == timestamp), None)
                            if cpu is None:
                                cpu = CPUMetrics(timestamp=timestamp)
                                metrics.append(cpu)
                            
                            # Set the attribute
                            if "load1" in metric_name:
                                cpu.load_average_1m = value_float
                            elif "load5" in metric_name:
                                cpu.load_average_5m = value_float
                            elif "load15" in metric_name:
                                cpu.load_average_15m = value_float
            except Exception as e:
                logger.debug(f"Failed to query {metric_name}: {e}")
        
        return metrics
    
    async def _query_memory_metrics(self) -> list[MemoryMetrics]:
        """Query Prometheus for memory metrics."""
        metrics = []
        
        memory_queries = [
            ("node_memory_MemTotal_bytes", "Total memory in bytes"),
            ("node_memory_MemAvailable_bytes", "Available memory in bytes"),
            ("node_memory_MemFree_bytes", "Free memory in bytes"),
            ("node_memory_SwapTotal_bytes", "Total swap in bytes"),
            ("node_memory_SwapFree_bytes", "Free swap in bytes"),
        ]
        
        for metric_name, description in memory_queries:
            try:
                url = f"{self.prometheus_url}/api/v1/query"
                params = {"query": metric_name}
                resp = await self._client.get(url, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("status") == "success":
                    results = data.get("data", {}).get("result", [])
                    for result in results:
                        metric = result.get("metric", {})
                        value = result.get("value", [])
                        
                        if len(value) >= 2:
                            timestamp = float(value[0])
                            value_float = float(value[1])
                            
                            instance = metric.get("instance", "unknown")
                            
                            # Find or create memory metric for this instance
                            mem = next((m for m in metrics if m.timestamp == timestamp), None)
                            if mem is None:
                                mem = MemoryMetrics(timestamp=timestamp)
                                metrics.append(mem)
                            
                            # Set the attribute
                            if "MemTotal" in metric_name:
                                mem.total_mb = value_float / (1024 * 1024)
                            elif "MemAvailable" in metric_name:
                                mem.available_mb = value_float / (1024 * 1024)
                            elif "MemFree" in metric_name:
                                mem.free_mb = value_float / (1024 * 1024)
                            elif "SwapTotal" in metric_name:
                                mem.swap_total_mb = value_float / (1024 * 1024)
                            elif "SwapFree" in metric_name:
                                mem.swap_free_mb = value_float / (1024 * 1024)
            except Exception as e:
                logger.debug(f"Failed to query {metric_name}: {e}")
        
        # Calculate utilization
        for mem in metrics:
            if mem.total_mb > 0:
                mem.used_mb = mem.total_mb - mem.available_mb
                mem.utilization_percent = (mem.used_mb / mem.total_mb) * 100
            if mem.swap_total_mb > 0:
                mem.swap_used_mb = mem.swap_total_mb - mem.swap_free_mb
                mem.swap_utilization_percent = (mem.swap_used_mb / mem.swap_total_mb) * 100
        
        return metrics
    
    async def _query_network_metrics(self) -> list[NetworkMetrics]:
        """Query Prometheus for network metrics."""
        metrics = []
        
        network_queries = [
            ("node_network_receive_bytes_total", "Bytes received"),
            ("node_network_transmit_bytes_total", "Bytes transmitted"),
            ("node_network_receive_packets_total", "Packets received"),
            ("node_network_transmit_packets_total", "Packets transmitted"),
            ("node_network_receive_errs_total", "Receive errors"),
            ("node_network_transmit_errs_total", "Transmit errors"),
            ("node_network_receive_drop_total", "Receive drops"),
            ("node_network_transmit_drop_total", "Transmit drops"),
        ]
        
        for metric_name, description in network_queries:
            try:
                url = f"{self.prometheus_url}/api/v1/query"
                params = {"query": metric_name}
                resp = await self._client.get(url, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("status") == "success":
                    results = data.get("data", {}).get("result", [])
                    for result in results:
                        metric = result.get("metric", {})
                        value = result.get("value", [])
                        
                        if len(value) >= 2:
                            timestamp = float(value[0])
                            value_float = float(value[1])
                            
                            device = metric.get("device", "unknown")
                            instance = metric.get("instance", "unknown")
                            interface_name = f"{instance}:{device}"
                            
                            # Find or create network metric for this interface
                            net = next((n for n in metrics if n.interface_name == interface_name), None)
                            if net is None:
                                net = NetworkMetrics(timestamp=timestamp, interface_name=interface_name)
                                metrics.append(net)
                            
                            # Set the attribute
                            if "receive_bytes" in metric_name:
                                net.bytes_received = int(value_float)
                            elif "transmit_bytes" in metric_name:
                                net.bytes_sent = int(value_float)
                            elif "receive_packets" in metric_name:
                                net.packets_received = int(value_float)
                            elif "transmit_packets" in metric_name:
                                net.packets_sent = int(value_float)
                            elif "receive_errs" in metric_name:
                                net.errors_received = int(value_float)
                            elif "transmit_errs" in metric_name:
                                net.errors_sent = int(value_float)
                            elif "receive_drop" in metric_name:
                                net.dropped_received = int(value_float)
                            elif "transmit_drop" in metric_name:
                                net.dropped_sent = int(value_float)
            except Exception as e:
                logger.debug(f"Failed to query {metric_name}: {e}")
        
        return metrics
    
    async def _query_inference_metrics(self) -> dict[str, Any]:
        """Query InfluxDB for inference metrics."""
        metrics = {}
        
        if not self._influxdb_client:
            return metrics
        
        # Check if InfluxDB auth has failed previously
        if self._influxdb_unauthorized:
            return metrics
        
        try:
            # Query recent inference metrics from InfluxDB
            query = f'''
                from(bucket: "{self.influxdb_bucket}")
                    |> range(start: -5m)
                    |> filter(fn: (r) => r._measurement == "cluster_metrics")
                    |> last()
            '''
            
            # Use asyncio.to_thread for synchronous InfluxDB query
            import asyncio
            result = await asyncio.to_thread(self._influxdb_client.query_api().query, query=query)
            
            for table in result:
                for record in table.records:
                    if record.get_field() == "inference_tps":
                        metrics["tokens_per_second"] = record.get_value()
                    elif record.get_field() == "inference_latency_ms":
                        metrics["latency_ms"] = record.get_value()
        except Exception as e:
            error_str = str(e)
            if "unauthorized" in error_str.lower() or "401" in error_str:
                # Suppress repeated unauthorized errors
                if not self._influxdb_unauthorized:
                    self._influxdb_unauthorized = True
                    logger.error(
                        f"InfluxDB query failed (401 Unauthorized) for {self.influxdb_url}. "
                        f"Token: {_mask_token(self.influxdb_token)}, Org: {self.influxdb_org}, "
                        f"Bucket: {self.influxdb_bucket}. "
                        f"Local InfluxDB default token is '{DEFAULT_INFLUXDB_TOKEN}'. "
                        f"Disabling InfluxDB writes/queries for this monitor instance."
                    )
            else:
                logger.debug(f"Failed to query InfluxDB for inference metrics: {e}")
        
        return metrics
    
    async def _publish_metrics(self, snapshot: ClusterMetricsSnapshot) -> None:
        """Publish metrics to Prometheus and InfluxDB."""
        # Publish to Prometheus
        if PROMETHEUS_AVAILABLE and self._gpu_utilization_gauge:
            for gpu in snapshot.gpu_metrics:
                self._gpu_utilization_gauge.labels(
                    node_id=snapshot.node_id,
                    device_id=gpu.device_id,
                    device_name=gpu.device_name,
                ).set(gpu.utilization_percent)
        
        if PROMETHEUS_AVAILABLE and self._cpu_utilization_gauge and snapshot.cpu_metrics:
            self._cpu_utilization_gauge.labels(
                node_id=snapshot.node_id,
            ).set(snapshot.cpu_metrics.cpu_percent)
        
        if PROMETHEUS_AVAILABLE and self._memory_utilization_gauge and snapshot.memory_metrics:
            self._memory_utilization_gauge.labels(
                node_id=snapshot.node_id,
            ).set(snapshot.memory_metrics.utilization_percent)
        
        # Publish to InfluxDB
        if self._influxdb_write_api:
            try:
                point = Point("cluster_metrics") \
                    .tag("node_id", snapshot.node_id) \
                    .tag("node_ip", snapshot.node_ip) \
                    .field("timestamp", snapshot.timestamp)
                
                # Add CPU metrics
                if snapshot.cpu_metrics:
                    point.field("cpu_utilization_percent", snapshot.cpu_metrics.cpu_percent)
                    point.field("load_average_1m", snapshot.cpu_metrics.load_average_1m)
                
                # Add memory metrics
                if snapshot.memory_metrics:
                    point.field("memory_utilization_percent", snapshot.memory_metrics.utilization_percent)
                    point.field("memory_used_mb", snapshot.memory_metrics.used_mb)
                
                # Add GPU metrics
                for gpu in snapshot.gpu_metrics:
                    point.field(f"gpu_{gpu.device_id}_utilization", gpu.utilization_percent)
                    point.field(f"gpu_{gpu.device_id}_memory_used_mb", gpu.memory_used_mb)
                
                # Add inference metrics
                for metric_name, metric_value in snapshot.inference_metrics.items():
                    point.field(metric_name, metric_value)
                
                self._influxdb_write_api.write(
                    bucket=self.influxdb_bucket,
                    record=point
                )
            except Exception as e:
                error_str = str(e)
                if "unauthorized" in error_str.lower() or "401" in error_str:
                    # Suppress repeated unauthorized errors
                    if not self._influxdb_unauthorized:
                        self._influxdb_unauthorized = True
                        logger.error(
                            f"InfluxDB write failed (401 Unauthorized) for {self.influxdb_url}. "
                            f"Token: {_mask_token(self.influxdb_token)}, Org: {self.influxdb_org}, "
                            f"Bucket: {self.influxdb_bucket}. "
                            f"Local InfluxDB default token is '{DEFAULT_INFLUXDB_TOKEN}'. "
                            f"Disabling InfluxDB writes/queries for this monitor instance."
                        )
                else:
                    logger.error(f"Error writing to InfluxDB: {e}")
    
    def get_metrics_history(self) -> list[ClusterMetricsSnapshot]:
        """Get all collected metrics snapshots."""
        return self._metrics_history.copy()
    
    def calculate_aggregates(self) -> MetricsAggregates:
        """
        Calculate aggregate statistics from collected metrics.
        
        Returns:
            MetricsAggregates with min/max/avg/p95 values for all metrics.
        """
        aggregates = MetricsAggregates()
        
        if not self._metrics_history:
            return aggregates
        
        # GPU metrics
        gpu_utils = []
        gpu_memories = []
        for snapshot in self._metrics_history:
            for gpu in snapshot.gpu_metrics:
                gpu_utils.append(gpu.utilization_percent)
                gpu_memories.append(gpu.memory_used_mb)
        
        if gpu_utils:
            aggregates.gpu_utilization_min = min(gpu_utils)
            aggregates.gpu_utilization_max = max(gpu_utils)
            aggregates.gpu_utilization_avg = statistics.mean(gpu_utils)
            aggregates.gpu_utilization_p95 = statistics.quantiles(gpu_utils, n=20)[18] if len(gpu_utils) > 1 else gpu_utils[0]
        
        if gpu_memories:
            aggregates.gpu_memory_used_mb_min = min(gpu_memories)
            aggregates.gpu_memory_used_mb_max = max(gpu_memories)
            aggregates.gpu_memory_used_mb_avg = statistics.mean(gpu_memories)
        
        # CPU metrics
        cpu_utils = [s.cpu_metrics.cpu_percent for s in self._metrics_history if s.cpu_metrics]
        if cpu_utils:
            aggregates.cpu_utilization_min = min(cpu_utils)
            aggregates.cpu_utilization_max = max(cpu_utils)
            aggregates.cpu_utilization_avg = statistics.mean(cpu_utils)
        
        # Memory metrics
        mem_utils = [s.memory_metrics.utilization_percent for s in self._metrics_history if s.memory_metrics]
        if mem_utils:
            aggregates.memory_utilization_min = min(mem_utils)
            aggregates.memory_utilization_max = max(mem_utils)
            aggregates.memory_utilization_avg = statistics.mean(mem_utils)
        
        # Network metrics
        network_throughputs = []
        for snapshot in self._metrics_history:
            for net in snapshot.network_metrics:
                # Calculate throughput based on bytes sent/received
                # This is a simplified calculation
                throughput = (net.bytes_sent + net.bytes_received) / (1024 * 1024)  # MB
                network_throughputs.append(throughput)
        
        if network_throughputs:
            aggregates.network_throughput_mbps_min = min(network_throughputs)
            aggregates.network_throughput_mbps_max = max(network_throughputs)
            aggregates.network_throughput_mbps_avg = statistics.mean(network_throughputs)
        
        # Inference metrics
        tps_values = []
        latency_values = []
        for snapshot in self._metrics_history:
            if "tokens_per_second" in snapshot.inference_metrics:
                tps_values.append(snapshot.inference_metrics["tokens_per_second"])
            if "latency_ms" in snapshot.inference_metrics:
                latency_values.append(snapshot.inference_metrics["latency_ms"])
        
        if tps_values:
            aggregates.inference_tps_min = min(tps_values)
            aggregates.inference_tps_max = max(tps_values)
            aggregates.inference_tps_avg = statistics.mean(tps_values)
        
        if latency_values:
            aggregates.inference_latency_ms_min = min(latency_values)
            aggregates.inference_latency_ms_max = max(latency_values)
            aggregates.inference_latency_ms_avg = statistics.mean(latency_values)
        
        return aggregates
    
    def generate_report(self) -> str:
        """
        Generate a human-readable report of collected metrics.
        
        Returns:
            Multi-line string with metrics report.
        """
        aggregates = self.calculate_aggregates()
        lines = [
            "=" * 70,
            "CLUSTER METRICS REPORT",
            "=" * 70,
            "",
            f"Total snapshots collected: {len(self._metrics_history)}",
            f"Collection duration: {self._get_duration_str()}",
            "",
            "--- GPU Metrics ---",
            f"  Utilization:  {aggregates.gpu_utilization_min:.1f}% - {aggregates.gpu_utilization_max:.1f}% (avg: {aggregates.gpu_utilization_avg:.1f}%, p95: {aggregates.gpu_utilization_p95:.1f}%)",
            f"  Memory Used:  {aggregates.gpu_memory_used_mb_min:.1f} MB - {aggregates.gpu_memory_used_mb_max:.1f} MB (avg: {aggregates.gpu_memory_used_mb_avg:.1f} MB)",
            "",
            "--- CPU Metrics ---",
            f"  Utilization:  {aggregates.cpu_utilization_min:.1f}% - {aggregates.cpu_utilization_max:.1f}% (avg: {aggregates.cpu_utilization_avg:.1f}%)",
            "",
            "--- Memory Metrics ---",
            f"  Utilization:  {aggregates.memory_utilization_min:.1f}% - {aggregates.memory_utilization_max:.1f}% (avg: {aggregates.memory_utilization_avg:.1f}%)",
            "",
            "--- Network Metrics ---",
            f"  Throughput:   {aggregates.network_throughput_mbps_min:.2f} MB/s - {aggregates.network_throughput_mbps_max:.2f} MB/s (avg: {aggregates.network_throughput_mbps_avg:.2f} MB/s)",
            "",
            "--- Inference Metrics ---",
            f"  TPS:          {aggregates.inference_tps_min:.2f} - {aggregates.inference_tps_max:.2f} (avg: {aggregates.inference_tps_avg:.2f})",
            f"  Latency:      {aggregates.inference_latency_ms_min:.1f} ms - {aggregates.inference_latency_ms_max:.1f} ms (avg: {aggregates.inference_latency_ms_avg:.1f} ms)",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)
    
    def _get_duration_str(self) -> str:
        """Get duration string from first to last metric collection."""
        if not self._metrics_history:
            return "N/A"
        
        first_ts = self._metrics_history[0].timestamp
        last_ts = self._metrics_history[-1].timestamp
        duration = last_ts - first_ts
        
        if duration < 60:
            return f"{duration:.1f} seconds"
        elif duration < 3600:
            return f"{duration / 60:.1f} minutes"
        else:
            return f"{duration / 3600:.1f} hours"
    
    def save_to_file(self, filepath: str) -> None:
        """Save metrics history to a JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics_history": [
                {
                    "timestamp": s.timestamp,
                    "node_id": s.node_id,
                    "node_ip": s.node_ip,
                    "gpu_metrics": [
                        {
                            "timestamp": g.timestamp,
                            "device_id": g.device_id,
                            "device_name": g.device_name,
                            "utilization_percent": g.utilization_percent,
                            "memory_used_mb": g.memory_used_mb,
                            "memory_total_mb": g.memory_total_mb,
                            "memory_utilization_percent": g.memory_utilization_percent,
                            "temperature_celsius": g.temperature_celsius,
                            "power_watts": g.power_watts,
                            "clock_speed_mhz": g.clock_speed_mhz,
                            "fan_speed_percent": g.fan_speed_percent,
                            "pcie_throughput_mb_s": g.pcie_throughput_mb_s,
                        }
                        for g in s.gpu_metrics
                    ],
                    "cpu_metrics": {
                        "timestamp": s.cpu_metrics.timestamp,
                        "cpu_percent": s.cpu_metrics.cpu_percent,
                        "load_average_1m": s.cpu_metrics.load_average_1m,
                        "load_average_5m": s.cpu_metrics.load_average_5m,
                        "load_average_15m": s.cpu_metrics.load_average_15m,
                        "context_switches": s.cpu_metrics.context_switches,
                        "interrupts": s.cpu_metrics.interrupts,
                        "soft_interrupts": s.cpu_metrics.soft_interrupts,
                        "process_count": s.cpu_metrics.process_count,
                        "thread_count": s.cpu_metrics.thread_count,
                    } if s.cpu_metrics else None,
                    "memory_metrics": {
                        "timestamp": s.memory_metrics.timestamp,
                        "total_mb": s.memory_metrics.total_mb,
                        "available_mb": s.memory_metrics.available_mb,
                        "used_mb": s.memory_metrics.used_mb,
                        "free_mb": s.memory_metrics.free_mb,
                        "utilization_percent": s.memory_metrics.utilization_percent,
                        "swap_total_mb": s.memory_metrics.swap_total_mb,
                        "swap_used_mb": s.memory_metrics.swap_used_mb,
                        "swap_free_mb": s.memory_metrics.swap_free_mb,
                        "swap_utilization_percent": s.memory_metrics.swap_utilization_percent,
                        "buffers_mb": s.memory_metrics.buffers_mb,
                        "cached_mb": s.memory_metrics.cached_mb,
                    } if s.memory_metrics else None,
                    "network_metrics": [
                        {
                            "timestamp": n.timestamp,
                            "interface_name": n.interface_name,
                            "bytes_sent": n.bytes_sent,
                            "bytes_received": n.bytes_received,
                            "packets_sent": n.packets_sent,
                            "packets_received": n.packets_received,
                            "errors_sent": n.errors_sent,
                            "errors_received": n.errors_received,
                            "dropped_sent": n.dropped_sent,
                            "dropped_received": n.dropped_received,
                            "speed_mbps": n.speed_mbps,
                            "duplex": n.duplex,
                            "link_up": n.link_up,
                        }
                        for n in s.network_metrics
                    ],
                    "inference_metrics": s.inference_metrics,
                }
                for s in self._metrics_history
            ],
            "aggregates": {
                "gpu_utilization_min": self.calculate_aggregates().gpu_utilization_min,
                "gpu_utilization_max": self.calculate_aggregates().gpu_utilization_max,
                "gpu_utilization_avg": self.calculate_aggregates().gpu_utilization_avg,
                "gpu_utilization_p95": self.calculate_aggregates().gpu_utilization_p95,
                "gpu_memory_used_mb_min": self.calculate_aggregates().gpu_memory_used_mb_min,
                "gpu_memory_used_mb_max": self.calculate_aggregates().gpu_memory_used_mb_max,
                "gpu_memory_used_mb_avg": self.calculate_aggregates().gpu_memory_used_mb_avg,
                "cpu_utilization_min": self.calculate_aggregates().cpu_utilization_min,
                "cpu_utilization_max": self.calculate_aggregates().cpu_utilization_max,
                "cpu_utilization_avg": self.calculate_aggregates().cpu_utilization_avg,
                "memory_utilization_min": self.calculate_aggregates().memory_utilization_min,
                "memory_utilization_max": self.calculate_aggregates().memory_utilization_max,
                "memory_utilization_avg": self.calculate_aggregates().memory_utilization_avg,
                "network_throughput_mbps_min": self.calculate_aggregates().network_throughput_mbps_min,
                "network_throughput_mbps_max": self.calculate_aggregates().network_throughput_mbps_max,
                "network_throughput_mbps_avg": self.calculate_aggregates().network_throughput_mbps_avg,
                "inference_tps_min": self.calculate_aggregates().inference_tps_min,
                "inference_tps_max": self.calculate_aggregates().inference_tps_max,
                "inference_tps_avg": self.calculate_aggregates().inference_tps_avg,
                "inference_latency_ms_min": self.calculate_aggregates().inference_latency_ms_min,
                "inference_latency_ms_max": self.calculate_aggregates().inference_latency_ms_max,
                "inference_latency_ms_avg": self.calculate_aggregates().inference_latency_ms_avg,
            },
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")


def create_monitor_from_env() -> ClusterMonitor:
    """
    Create a ClusterMonitor from environment variables.
    
    Environment variables:
    - PROMETHEUS_URL: URL for Prometheus server (default: internal cluster URL)
    - INFLUXDB_URL: URL for InfluxDB2 server (default: internal cluster URL)
      NOTE: For local development with kubectl port-forward, use http://127.0.0.1:8086
      (NOT https:// - port-forward maps to cluster port 80 which is plain HTTP).
      The URL will be automatically normalized if https://127.0.0.1:8086 is provided.
    - INFLUXDB_TOKEN: InfluxDB2 authentication token
    - INFLUXDB_ORG: InfluxDB2 organization name (default: "exo")
    - INFLUXDB_BUCKET: InfluxDB2 bucket name (default: "exo_metrics")
    - METRICS_COLLECTION_INTERVAL: Interval between collections (default: 15)
    
    Returns:
        Configured ClusterMonitor instance.
    """
    collection_interval = float(os.environ.get("METRICS_COLLECTION_INTERVAL", "15"))
    
    # Use default cluster URLs if not specified
    prometheus_url = os.environ.get("PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL)
    # Normalize InfluxDB URL to handle local port-forward (https -> http for localhost:8086)
    influxdb_url = _normalize_influxdb_url_for_port_forward(
        os.environ.get("INFLUXDB_URL", DEFAULT_INFLUXDB_URL)
    )
    influxdb_org = os.environ.get("INFLUXDB_ORG", DEFAULT_INFLUXDB_ORG)
    influxdb_bucket = os.environ.get("INFLUXDB_BUCKET", DEFAULT_INFLUXDB_BUCKET)
    # Use default token if not specified (matches test notes: PSCh4ng3me!)
    influxdb_token = os.environ.get("INFLUXDB_TOKEN", DEFAULT_INFLUXDB_TOKEN)
    
    return ClusterMonitor(
        prometheus_url=prometheus_url,
        influxdb_url=influxdb_url,
        influxdb_token=influxdb_token,
        influxdb_org=influxdb_org,
        influxdb_bucket=influxdb_bucket,
        collection_interval=collection_interval,
    )


# ============================================================================
# Helper functions for integration with gremlin_cluster_test.py
# ============================================================================

async def run_monitoring_during_test(
    monitor: ClusterMonitor,
    node_ids: list[str],
    node_ips: list[str],
    duration_seconds: float,
) -> ClusterMetricsSnapshot | None:
    """
    Run monitoring during a test and return the final snapshot.
    
    Args:
        monitor: ClusterMonitor instance
        node_ids: List of node IDs to monitor
        node_ips: List of node IPs corresponding to node_ids
        duration_seconds: Duration to run monitoring
        
    Returns:
        Final metrics snapshot, or None if monitoring failed.
    """
    if not monitor._running:
        await monitor.start()
    
    # Collect metrics for the specified duration
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        try:
            snapshot = await monitor.collect_snapshot()
            if snapshot:
                # Update node info
                snapshot.node_id = node_ids[0]  # Use first node for now
                snapshot.node_ip = node_ips[0]
        except Exception as e:
            logger.error(f"Error during test monitoring: {e}")
        
        await asyncio.sleep(min(monitor.collection_interval, duration_seconds - (time.time() - start_time)))
    
    return monitor._metrics_history[-1] if monitor._metrics_history else None


def print_bottleneck_analysis(metrics: MetricsAggregates, test_results: dict[str, Any]) -> None:
    """
    Print bottleneck analysis based on collected metrics and test results.
    
    Args:
        metrics: Aggregated metrics
        test_results: Test results dictionary
    """
    lines = [
        "",
        "=" * 70,
        "BOTTLENECK ANALYSIS",
        "=" * 70,
    ]
    
    # Analyze GPU bottlenecks
    if metrics.gpu_utilization_avg < 50:
        lines.append("")
        lines.append("⚠️  GPU Bottleneck: Low GPU Utilization")
        lines.append(f"   GPU utilization is only {metrics.gpu_utilization_avg:.1f}% on average.")
        lines.append("   This suggests the GPU is underutilized, which could be caused by:")
        lines.append("   - Insufficient batch size")
        lines.append("   - Data loading bottleneck")
        lines.append("   - CPU bottleneck limiting GPU throughput")
        lines.append("   - Network latency in distributed inference")
    
    if metrics.gpu_memory_used_mb_avg > 0:
        lines.append("")
        lines.append(f"GPU Memory Usage: {metrics.gpu_memory_used_mb_avg:.1f} MB average")
        if metrics.gpu_memory_used_mb_max > 0:
            lines.append(f"  Peak: {metrics.gpu_memory_used_mb_max:.1f} MB")
    
    # Analyze CPU bottlenecks
    if metrics.cpu_utilization_avg > 80:
        lines.append("")
        lines.append("⚠️  CPU Bottleneck: High CPU Utilization")
        lines.append(f"   CPU utilization is {metrics.cpu_utilization_avg:.1f}% on average.")
        lines.append("   This could indicate:")
        lines.append("   - CPU-bound preprocessing")
        lines.append("   - Data loading bottleneck")
        lines.append("   - Inefficient parallelization")
    
    # Analyze memory bottlenecks
    if metrics.memory_utilization_avg > 80:
        lines.append("")
        lines.append("⚠️  Memory Bottleneck: High Memory Utilization")
        lines.append(f"   Memory utilization is {metrics.memory_utilization_avg:.1f}% on average.")
        lines.append("   This could indicate:")
        lines.append("   - Insufficient RAM")
        lines.append("   - Memory leaks")
        lines.append("   - Large model not fitting in memory")
    
    # Analyze network bottlenecks
    if metrics.network_throughput_mbps_avg > 0:
        lines.append("")
        lines.append(f"Network Throughput: {metrics.network_throughput_mbps_avg:.2f} MB/s average")
        if metrics.network_throughput_mbps_max > 0:
            lines.append(f"  Peak: {metrics.network_throughput_mbps_max:.2f} MB/s")
    
    # Analyze inference performance
    if metrics.inference_tps_avg > 0:
        lines.append("")
        lines.append(f"Inference Performance: {metrics.inference_tps_avg:.2f} tokens/s average")
        if metrics.inference_latency_ms_avg > 0:
            lines.append(f"  Average Latency: {metrics.inference_latency_ms_avg:.1f} ms")
    
    # Test results correlation
    if "basic" in test_results:
        basic = test_results["basic"]
        if hasattr(basic, "tokens_per_second") and basic.tokens_per_second > 0:
            lines.append("")
            lines.append(f"Test Results: {basic.tokens_per_second:.2f} tokens/s")
    
    lines.append("")
    lines.append("=" * 70)
    
    print("\n".join(lines))


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster Monitoring Tool")
    parser.add_argument("--prometheus-url", help="Prometheus URL")
    parser.add_argument("--influxdb-url", help="InfluxDB URL")
    parser.add_argument("--influxdb-token", help="InfluxDB token")
    parser.add_argument("--influxdb-org", help="InfluxDB organization")
    parser.add_argument("--influxdb-bucket", help="InfluxDB bucket")
    parser.add_argument("--interval", type=float, default=15.0, help="Collection interval in seconds")
    parser.add_argument("--output", help="Output file for metrics")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration to run in seconds")
    
    args = parser.parse_args()
    
    async def main():
        monitor = ClusterMonitor(
            prometheus_url=args.prometheus_url,
            influxdb_url=args.influxdb_url,
            influxdb_token=args.influxdb_token,
            influxdb_org=args.influxdb_org,
            influxdb_bucket=args.influxdb_bucket,
            collection_interval=args.interval,
        )
        
        async with monitor:
            print(f"Monitoring for {args.duration} seconds...")
            await asyncio.sleep(args.duration)
            
            print(monitor.generate_report())
            
            if args.output:
                monitor.save_to_file(args.output)
    
    asyncio.run(main())
