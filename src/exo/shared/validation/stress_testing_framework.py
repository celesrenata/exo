"""
Stress testing framework for long-running distributed inference stability validation.
"""

import asyncio
import random
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from exo.utils.pydantic_ext import CamelCaseModel
from ..types.validation import (
    FailureMode, RecoveryAction, ValidationStatus, 
    EnhancedTokenChunk, DeviceHealth
)
from ..types.models import ModelId


class StressTestType(str, Enum):
    """Types of stress tests."""
    LONG_RUNNING_STABILITY = "long_running_stability"
    HIGH_THROUGHPUT = "high_throughput"
    FAILURE_INJECTION = "failure_injection"
    MEMORY_PRESSURE = "memory_pressure"
    NETWORK_DISRUPTION = "network_disruption"
    CONCURRENT_REQUESTS = "concurrent_requests"


class FailureInjectionType(str, Enum):
    """Types of failures to inject during testing."""
    NETWORK_DELAY = "network_delay"
    PACKET_LOSS = "packet_loss"
    DEVICE_DISCONNECT = "device_disconnect"
    MEMORY_CORRUPTION = "memory_corruption"
    TIMEOUT_SIMULATION = "timeout_simulation"
    ENCODING_CORRUPTION = "encoding_corruption"
    SYNCHRONIZATION_FAILURE = "synchronization_failure"


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during stress testing."""
    timestamp: datetime
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    network_io: Dict[str, float]  # bytes/sec
    disk_io: Dict[str, float]  # bytes/sec
    tokens_per_second: float
    latency_ms: float
    error_count: int
    active_connections: int


@dataclass
class StressTestConfig:
    """Configuration for stress tests."""
    test_type: StressTestType
    duration: timedelta
    model_id: ModelId
    concurrent_requests: int = 1
    request_interval: float = 1.0  # seconds
    failure_injection_rate: float = 0.0  # 0.0 to 1.0
    failure_types: List[FailureInjectionType] = field(default_factory=list)
    max_tokens_per_request: int = 100
    monitoring_interval: float = 5.0  # seconds
    recovery_timeout: float = 30.0  # seconds
    custom_prompts: List[str] = field(default_factory=list)


@dataclass
class StressTestResult:
    """Result of a stress test run."""
    test_config: StressTestConfig
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float
    peak_memory_usage: float
    peak_cpu_usage: float
    errors_by_type: Dict[str, int]
    performance_timeline: List[PerformanceMetrics]
    recovery_events: List[Dict[str, Any]]
    stability_score: float  # 0.0 to 1.0
    details: str
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def duration_seconds(self) -> float:
        """Get test duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()


class FailureInjector:
    """Injects various types of failures for testing recovery mechanisms."""
    
    def __init__(self):
        self.active_failures: Dict[str, Any] = {}
        self.injection_history: List[Dict[str, Any]] = []
    
    async def inject_failure(self, 
                           failure_type: FailureInjectionType,
                           duration: float = 5.0,
                           severity: float = 0.5) -> bool:
        """
        Inject a specific type of failure.
        
        Args:
            failure_type: Type of failure to inject
            duration: How long the failure should last
            severity: Severity of the failure (0.0 to 1.0)
            
        Returns:
            True if injection was successful
        """
        failure_id = f"{failure_type}_{int(time.time())}"
        
        try:
            if failure_type == FailureInjectionType.NETWORK_DELAY:
                await self._inject_network_delay(failure_id, duration, severity)
            elif failure_type == FailureInjectionType.PACKET_LOSS:
                await self._inject_packet_loss(failure_id, duration, severity)
            elif failure_type == FailureInjectionType.MEMORY_CORRUPTION:
                await self._inject_memory_corruption(failure_id, duration, severity)
            elif failure_type == FailureInjectionType.TIMEOUT_SIMULATION:
                await self._inject_timeout_simulation(failure_id, duration, severity)
            elif failure_type == FailureInjectionType.ENCODING_CORRUPTION:
                await self._inject_encoding_corruption(failure_id, duration, severity)
            else:
                return False
            
            # Record injection
            self.injection_history.append({
                "failure_id": failure_id,
                "type": failure_type,
                "timestamp": datetime.now(),
                "duration": duration,
                "severity": severity,
                "status": "injected"
            })
            
            return True
            
        except Exception as e:
            print(f"Failed to inject {failure_type}: {e}")
            return False
    
    async def _inject_network_delay(self, failure_id: str, duration: float, severity: float):
        """Simulate network delays."""
        delay_ms = int(severity * 1000)  # Up to 1 second delay
        self.active_failures[failure_id] = {
            "type": "network_delay",
            "delay_ms": delay_ms,
            "start_time": time.time()
        }
        
        # Schedule cleanup
        asyncio.create_task(self._cleanup_failure(failure_id, duration))
    
    async def _inject_packet_loss(self, failure_id: str, duration: float, severity: float):
        """Simulate packet loss."""
        loss_rate = severity  # 0.0 to 1.0
        self.active_failures[failure_id] = {
            "type": "packet_loss",
            "loss_rate": loss_rate,
            "start_time": time.time()
        }
        
        asyncio.create_task(self._cleanup_failure(failure_id, duration))
    
    async def _inject_memory_corruption(self, failure_id: str, duration: float, severity: float):
        """Simulate memory corruption."""
        corruption_rate = severity
        self.active_failures[failure_id] = {
            "type": "memory_corruption",
            "corruption_rate": corruption_rate,
            "start_time": time.time()
        }
        
        asyncio.create_task(self._cleanup_failure(failure_id, duration))
    
    async def _inject_timeout_simulation(self, failure_id: str, duration: float, severity: float):
        """Simulate timeout conditions."""
        timeout_factor = 1.0 + severity * 10  # Up to 11x normal timeout
        self.active_failures[failure_id] = {
            "type": "timeout_simulation",
            "timeout_factor": timeout_factor,
            "start_time": time.time()
        }
        
        asyncio.create_task(self._cleanup_failure(failure_id, duration))
    
    async def _inject_encoding_corruption(self, failure_id: str, duration: float, severity: float):
        """Simulate encoding corruption."""
        corruption_probability = severity
        self.active_failures[failure_id] = {
            "type": "encoding_corruption",
            "corruption_probability": corruption_probability,
            "start_time": time.time()
        }
        
        asyncio.create_task(self._cleanup_failure(failure_id, duration))
    
    async def _cleanup_failure(self, failure_id: str, duration: float):
        """Clean up injected failure after duration."""
        await asyncio.sleep(duration)
        if failure_id in self.active_failures:
            del self.active_failures[failure_id]
            
            # Update history
            for record in self.injection_history:
                if record["failure_id"] == failure_id:
                    record["status"] = "cleaned_up"
                    record["end_time"] = datetime.now()
                    break
    
    def is_failure_active(self, failure_type: FailureInjectionType) -> bool:
        """Check if a specific type of failure is currently active."""
        return any(
            failure["type"] == failure_type.value 
            for failure in self.active_failures.values()
        )
    
    def get_active_failures(self) -> List[Dict[str, Any]]:
        """Get list of currently active failures."""
        return list(self.active_failures.values())


class PerformanceMonitor:
    """Monitors system performance during stress testing."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv
        }
        
        # Disk I/O
        disk_io_counters = psutil.disk_io_counters()
        disk_io = {
            "read_bytes": disk_io_counters.read_bytes if disk_io_counters else 0,
            "write_bytes": disk_io_counters.write_bytes if disk_io_counters else 0
        }
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            network_io=network_io,
            disk_io=disk_io,
            tokens_per_second=0.0,  # Will be updated by test runner
            latency_ms=0.0,  # Will be updated by test runner
            error_count=0,  # Will be updated by test runner
            active_connections=0  # Will be updated by test runner
        )
    
    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak values from monitoring history."""
        if not self.metrics_history:
            return {}
        
        return {
            "peak_cpu": max(m.cpu_usage for m in self.metrics_history),
            "peak_memory": max(m.memory_usage for m in self.metrics_history),
            "peak_latency": max(m.latency_ms for m in self.metrics_history),
            "max_tokens_per_second": max(m.tokens_per_second for m in self.metrics_history)
        }


class StressTestingFramework:
    """
    Comprehensive stress testing framework for distributed inference.
    
    Provides long-running stability tests, performance monitoring,
    and automated failure injection capabilities.
    """
    
    def __init__(self, 
                 distributed_runner=None,
                 validation_callback: Optional[Callable] = None):
        """
        Initialize the stress testing framework.
        
        Args:
            distributed_runner: Runner for distributed inference
            validation_callback: Optional callback for custom validation
        """
        self.distributed_runner = distributed_runner
        self.validation_callback = validation_callback
        self.failure_injector = FailureInjector()
        self.performance_monitor = PerformanceMonitor()
        self.test_history: List[StressTestResult] = []
        self.active_test: Optional[StressTestConfig] = None
    
    async def run_stress_test(self, config: StressTestConfig) -> StressTestResult:
        """
        Run a comprehensive stress test.
        
        Args:
            config: Test configuration
            
        Returns:
            StressTestResult with detailed metrics
        """
        self.active_test = config
        start_time = datetime.now()
        
        # Initialize counters
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        latencies = []
        errors_by_type = {}
        recovery_events = []
        
        # Start performance monitoring
        await self.performance_monitor.start_monitoring()
        
        try:
            # Run test based on type
            if config.test_type == StressTestType.LONG_RUNNING_STABILITY:
                result_data = await self._run_long_running_test(config)
            elif config.test_type == StressTestType.HIGH_THROUGHPUT:
                result_data = await self._run_high_throughput_test(config)
            elif config.test_type == StressTestType.FAILURE_INJECTION:
                result_data = await self._run_failure_injection_test(config)
            elif config.test_type == StressTestType.CONCURRENT_REQUESTS:
                result_data = await self._run_concurrent_requests_test(config)
            else:
                raise ValueError(f"Unsupported test type: {config.test_type}")
            
            # Extract results
            total_requests = result_data["total_requests"]
            successful_requests = result_data["successful_requests"]
            failed_requests = result_data["failed_requests"]
            latencies = result_data["latencies"]
            errors_by_type = result_data["errors_by_type"]
            recovery_events = result_data["recovery_events"]
            
        finally:
            # Stop monitoring
            await self.performance_monitor.stop_monitoring()
            self.active_test = None
        
        end_time = datetime.now()
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        peak_metrics = self.performance_monitor.get_peak_metrics()
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(
            successful_requests, total_requests, errors_by_type, peak_metrics
        )
        
        # Create result
        result = StressTestResult(
            test_config=config,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_latency=avg_latency,
            peak_memory_usage=peak_metrics.get("peak_memory", 0.0),
            peak_cpu_usage=peak_metrics.get("peak_cpu", 0.0),
            errors_by_type=errors_by_type,
            performance_timeline=self.performance_monitor.metrics_history.copy(),
            recovery_events=recovery_events,
            stability_score=stability_score,
            details=f"Stress test completed: {config.test_type.value}"
        )
        
        self.test_history.append(result)
        return result
    
    async def _run_long_running_test(self, config: StressTestConfig) -> Dict[str, Any]:
        """Run long-running stability test."""
        end_time = datetime.now() + config.duration
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        latencies = []
        errors_by_type = {}
        recovery_events = []
        
        prompts = config.custom_prompts or [
            "Tell me a story about artificial intelligence.",
            "Explain quantum computing in simple terms.",
            "What are the benefits of distributed computing?",
            "Describe the future of machine learning."
        ]
        
        while datetime.now() < end_time:
            prompt = random.choice(prompts)
            
            try:
                start_request = time.time()
                
                # Simulate distributed inference
                output = await self._simulate_inference(
                    prompt, config.model_id, config.max_tokens_per_request
                )
                
                end_request = time.time()
                latency = (end_request - start_request) * 1000  # ms
                
                latencies.append(latency)
                successful_requests += 1
                
                # Update performance metrics
                if self.performance_monitor.metrics_history:
                    latest_metrics = self.performance_monitor.metrics_history[-1]
                    latest_metrics.tokens_per_second = len(output.split()) / (latency / 1000)
                    latest_metrics.latency_ms = latency
                
            except Exception as e:
                failed_requests += 1
                error_type = type(e).__name__
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
                
                # Record recovery event
                recovery_events.append({
                    "timestamp": datetime.now(),
                    "error_type": error_type,
                    "error_message": str(e),
                    "recovery_attempted": True
                })
            
            total_requests += 1
            
            # Wait before next request
            await asyncio.sleep(config.request_interval)
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "latencies": latencies,
            "errors_by_type": errors_by_type,
            "recovery_events": recovery_events
        }
    
    async def _run_high_throughput_test(self, config: StressTestConfig) -> Dict[str, Any]:
        """Run high throughput test."""
        # Similar to long-running but with minimal intervals
        modified_config = StressTestConfig(
            test_type=config.test_type,
            duration=config.duration,
            model_id=config.model_id,
            concurrent_requests=config.concurrent_requests,
            request_interval=0.1,  # Very short interval
            max_tokens_per_request=config.max_tokens_per_request,
            custom_prompts=config.custom_prompts
        )
        
        return await self._run_long_running_test(modified_config)
    
    async def _run_failure_injection_test(self, config: StressTestConfig) -> Dict[str, Any]:
        """Run test with failure injection."""
        # Start base test
        base_task = asyncio.create_task(self._run_long_running_test(config))
        
        # Inject failures periodically
        failure_task = asyncio.create_task(
            self._inject_failures_periodically(config)
        )
        
        try:
            result_data = await base_task
            return result_data
        finally:
            failure_task.cancel()
            try:
                await failure_task
            except asyncio.CancelledError:
                pass
    
    async def _run_concurrent_requests_test(self, config: StressTestConfig) -> Dict[str, Any]:
        """Run test with concurrent requests."""
        tasks = []
        
        # Create multiple concurrent test tasks
        for i in range(config.concurrent_requests):
            task_config = StressTestConfig(
                test_type=StressTestType.LONG_RUNNING_STABILITY,
                duration=config.duration,
                model_id=config.model_id,
                concurrent_requests=1,
                request_interval=config.request_interval,
                max_tokens_per_request=config.max_tokens_per_request,
                custom_prompts=config.custom_prompts
            )
            
            task = asyncio.create_task(self._run_long_running_test(task_config))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        latencies = []
        errors_by_type = {}
        recovery_events = []
        
        for result in results:
            if isinstance(result, dict):
                total_requests += result["total_requests"]
                successful_requests += result["successful_requests"]
                failed_requests += result["failed_requests"]
                latencies.extend(result["latencies"])
                
                for error_type, count in result["errors_by_type"].items():
                    errors_by_type[error_type] = errors_by_type.get(error_type, 0) + count
                
                recovery_events.extend(result["recovery_events"])
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "latencies": latencies,
            "errors_by_type": errors_by_type,
            "recovery_events": recovery_events
        }
    
    async def _inject_failures_periodically(self, config: StressTestConfig):
        """Inject failures periodically during test."""
        end_time = datetime.now() + config.duration
        
        while datetime.now() < end_time:
            if config.failure_injection_rate > 0 and config.failure_types:
                if random.random() < config.failure_injection_rate:
                    failure_type = random.choice(config.failure_types)
                    duration = random.uniform(1.0, 10.0)  # 1-10 seconds
                    severity = random.uniform(0.1, 0.8)   # Low to high severity
                    
                    await self.failure_injector.inject_failure(
                        failure_type, duration, severity
                    )
            
            await asyncio.sleep(5.0)  # Check every 5 seconds
    
    async def _simulate_inference(self, 
                                prompt: str, 
                                model_id: ModelId,
                                max_tokens: int) -> str:
        """Simulate distributed inference."""
        if self.distributed_runner:
            # TODO: Integrate with actual distributed runner
            pass
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 2.0))
        
        # Check for active failures and simulate their effects
        active_failures = self.failure_injector.get_active_failures()
        
        for failure in active_failures:
            if failure["type"] == "encoding_corruption":
                if random.random() < failure["corruption_probability"]:
                    raise ValueError("Simulated encoding corruption")
            elif failure["type"] == "timeout_simulation":
                timeout_delay = 5.0 * failure["timeout_factor"]
                await asyncio.sleep(timeout_delay)
            elif failure["type"] == "memory_corruption":
                if random.random() < failure["corruption_rate"]:
                    raise RuntimeError("Simulated memory corruption")
        
        # Return simulated output
        return f"Generated response for: {prompt[:30]}..."
    
    def _calculate_stability_score(self, 
                                 successful: int, 
                                 total: int,
                                 errors: Dict[str, int],
                                 peak_metrics: Dict[str, float]) -> float:
        """Calculate overall stability score."""
        if total == 0:
            return 0.0
        
        # Base score from success rate
        success_rate = successful / total
        base_score = success_rate
        
        # Penalty for high resource usage
        cpu_penalty = max(0, (peak_metrics.get("peak_cpu", 0) - 80) / 20) * 0.1
        memory_penalty = max(0, (peak_metrics.get("peak_memory", 0) - 80) / 20) * 0.1
        
        # Penalty for errors
        error_penalty = min(0.3, len(errors) * 0.05)
        
        stability_score = base_score - cpu_penalty - memory_penalty - error_penalty
        return max(0.0, min(1.0, stability_score))
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all stress tests."""
        if not self.test_history:
            return {"message": "No stress tests run yet"}
        
        total_tests = len(self.test_history)
        avg_stability = sum(test.stability_score for test in self.test_history) / total_tests
        
        # Test type distribution
        test_types = {}
        for test in self.test_history:
            test_type = test.test_config.test_type.value
            test_types[test_type] = test_types.get(test_type, 0) + 1
        
        # Success rates
        success_rates = [test.success_rate for test in self.test_history]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        return {
            "total_stress_tests": total_tests,
            "average_stability_score": avg_stability,
            "average_success_rate": avg_success_rate,
            "test_type_distribution": test_types,
            "last_test_time": (
                self.test_history[-1].end_time.isoformat() 
                if self.test_history else None
            ),
            "peak_performance": {
                "max_requests_per_test": max(
                    test.total_requests for test in self.test_history
                ),
                "best_stability_score": max(
                    test.stability_score for test in self.test_history
                ),
                "lowest_latency": min(
                    test.average_latency for test in self.test_history
                    if test.average_latency > 0
                )
            }
        }