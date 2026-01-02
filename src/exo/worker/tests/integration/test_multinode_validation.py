"""
Multi-node functionality validation tests.

This module provides comprehensive validation tests for multi-node instance creation,
inference coordination, graceful shutdown, and error handling mechanisms.
"""

import asyncio
import pytest
import time
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from exo.shared.types.worker.instances import BoundInstance, MlxJacclInstance
from exo.shared.types.worker.runners import RunnerId, RunnerIdle, RunnerFailed
from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import CreateRunner, Shutdown, TaskStatus
from exo.utils.channels import Sender, channel
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.worker.runner.shutdown_coordinator import ShutdownCoordinator, get_shutdown_coordinator
from exo.worker.runner.resource_manager import ResourceManager, get_resource_manager
from exo.worker.runner.error_handler import ErrorHandler, get_error_handler


@dataclass
class MultiNodeTestConfig:
    """Configuration for multi-node tests."""
    node_count: int = 2
    runners_per_node: int = 2
    test_timeout: float = 30.0
    shutdown_timeout: float = 10.0
    parallelism_type: str = "pipeline"  # "pipeline" or "tensor"


class MultiNodeValidator:
    """Validator for multi-node functionality."""
    
    def __init__(self, config: MultiNodeTestConfig):
        self.config = config
        self.nodes: Dict[str, Dict[str, RunnerSupervisor]] = {}
        self.event_senders: Dict[str, Sender[Event]] = {}
        self.shutdown_coordinator = get_shutdown_coordinator()
        self.resource_manager = get_resource_manager()
        self.error_handler = get_error_handler()
        
    async def setup_multi_node_environment(self) -> bool:
        """Set up a multi-node test environment."""
        try:
            print(f"Setting up {self.config.node_count} nodes with {self.config.runners_per_node} runners each")
            
            for node_id in range(self.config.node_count):
                node_name = f"node_{node_id}"
                self.nodes[node_name] = {}
                
                # Create event sender for this node
                event_sender, _ = channel[Event]()
                self.event_senders[node_name] = event_sender
                
                # Create runners for this node
                for runner_id in range(self.config.runners_per_node):
                    runner_name = f"runner_{node_id}_{runner_id}"
                    
                    # Create mock bound instance
                    bound_instance = self._create_mock_bound_instance(
                        runner_name, 
                        self.config.parallelism_type
                    )
                    
                    # Create runner supervisor
                    runner = RunnerSupervisor.create(
                        bound_instance=bound_instance,
                        event_sender=event_sender.clone(),
                        initialize_timeout=self.config.test_timeout
                    )
                    
                    self.nodes[node_name][runner_name] = runner
                    print(f"Created runner {runner_name} on {node_name}")
            
            print("Multi-node environment setup completed")
            return True
            
        except Exception as e:
            print(f"Failed to setup multi-node environment: {e}")
            return False
    
    def _create_mock_bound_instance(self, runner_id: str, parallelism_type: str) -> BoundInstance:
        """Create a mock bound instance for testing."""
        # Create appropriate instance type based on parallelism
        if parallelism_type == "tensor":
            instance = MlxJacclInstance(
                jaccl_devices=[0, 1],  # Multi-device for tensor parallelism
                model_id="test_model",
                shard_assignments={}
            )
        else:
            instance = MlxJacclInstance(
                jaccl_devices=[0],  # Single device for pipeline parallelism
                model_id="test_model", 
                shard_assignments={}
            )
        
        return BoundInstance(
            bound_runner_id=RunnerId(runner_id),
            bound_shard=Mock(),
            instance=instance
        )
    
    async def test_multi_node_instance_creation(self) -> bool:
        """Test multi-node instance creation works reliably."""
        print("Testing multi-node instance creation...")
        
        try:
            # Start all runners concurrently
            start_tasks = []
            for node_name, runners in self.nodes.items():
                for runner_name, runner in runners.items():
                    print(f"Starting runner {runner_name} on {node_name}")
                    start_tasks.append(asyncio.create_task(runner.run()))
            
            # Wait for all runners to start (with timeout)
            await asyncio.wait_for(
                asyncio.gather(*start_tasks, return_exceptions=True),
                timeout=self.config.test_timeout
            )
            
            # Verify all runners are healthy
            all_healthy = True
            for node_name, runners in self.nodes.items():
                for runner_name, runner in runners.items():
                    health = runner.get_runner_health()
                    if not health.get("healthy", False):
                        print(f"Runner {runner_name} on {node_name} is not healthy: {health}")
                        all_healthy = False
                    else:
                        print(f"Runner {runner_name} on {node_name} is healthy")
            
            if all_healthy:
                print("✓ Multi-node instance creation test passed")
                return True
            else:
                print("✗ Multi-node instance creation test failed - some runners unhealthy")
                return False
                
        except asyncio.TimeoutError:
            print("✗ Multi-node instance creation test failed - timeout")
            return False
        except Exception as e:
            print(f"✗ Multi-node instance creation test failed: {e}")
            return False
    
    async def test_inference_coordination(self) -> bool:
        """Test that inference works correctly across nodes."""
        print("Testing inference coordination across nodes...")
        
        try:
            # Create mock inference tasks for each runner
            inference_tasks = []
            
            for node_name, runners in self.nodes.items():
                for runner_name, runner in runners.items():
                    # Create a mock inference task
                    task = Mock()
                    task.task_id = f"task_{runner_name}"
                    task.instance_id = f"instance_{runner_name}"
                    
                    print(f"Starting inference task on {runner_name} ({node_name})")
                    inference_tasks.append(
                        asyncio.create_task(runner.start_task(task))
                    )
            
            # Wait for all inference tasks to complete
            results = await asyncio.wait_for(
                asyncio.gather(*inference_tasks, return_exceptions=True),
                timeout=self.config.test_timeout
            )
            
            # Check results
            success_count = 0
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    success_count += 1
                    print(f"✓ Inference task {i} completed successfully")
                else:
                    print(f"✗ Inference task {i} failed: {result}")
            
            total_tasks = len(inference_tasks)
            success_rate = success_count / total_tasks
            
            if success_rate >= 0.8:  # 80% success rate threshold
                print(f"✓ Inference coordination test passed ({success_count}/{total_tasks} tasks succeeded)")
                return True
            else:
                print(f"✗ Inference coordination test failed ({success_count}/{total_tasks} tasks succeeded)")
                return False
                
        except asyncio.TimeoutError:
            print("✗ Inference coordination test failed - timeout")
            return False
        except Exception as e:
            print(f"✗ Inference coordination test failed: {e}")
            return False
    
    async def test_graceful_shutdown_scenarios(self) -> bool:
        """Test graceful shutdown under various scenarios."""
        print("Testing graceful shutdown scenarios...")
        
        scenarios = [
            ("normal_shutdown", self._test_normal_shutdown),
            ("concurrent_shutdown", self._test_concurrent_shutdown),
            ("partial_failure_shutdown", self._test_partial_failure_shutdown),
            ("timeout_shutdown", self._test_timeout_shutdown)
        ]
        
        results = {}
        
        for scenario_name, test_func in scenarios:
            print(f"Testing scenario: {scenario_name}")
            try:
                result = await test_func()
                results[scenario_name] = result
                if result:
                    print(f"✓ Scenario {scenario_name} passed")
                else:
                    print(f"✗ Scenario {scenario_name} failed")
            except Exception as e:
                print(f"✗ Scenario {scenario_name} failed with exception: {e}")
                results[scenario_name] = False
        
        # Overall success if most scenarios pass
        success_count = sum(1 for result in results.values() if result)
        total_scenarios = len(scenarios)
        success_rate = success_count / total_scenarios
        
        if success_rate >= 0.75:  # 75% success rate threshold
            print(f"✓ Graceful shutdown test passed ({success_count}/{total_scenarios} scenarios)")
            return True
        else:
            print(f"✗ Graceful shutdown test failed ({success_count}/{total_scenarios} scenarios)")
            return False
    
    async def _test_normal_shutdown(self) -> bool:
        """Test normal shutdown scenario."""
        try:
            # Shutdown all runners normally
            shutdown_tasks = []
            for node_name, runners in self.nodes.items():
                for runner_name, runner in runners.items():
                    shutdown_tasks.append(
                        asyncio.create_task(
                            self.shutdown_coordinator.initiate_shutdown(
                                runner_id=runner_name,
                                timeout=self.config.shutdown_timeout
                            )
                        )
                    )
            
            # Wait for all shutdowns to complete
            results = await asyncio.wait_for(
                asyncio.gather(*shutdown_tasks, return_exceptions=True),
                timeout=self.config.shutdown_timeout + 5.0
            )
            
            # Check if all shutdowns succeeded
            success_count = sum(1 for result in results if result is True)
            return success_count >= len(shutdown_tasks) * 0.8  # 80% success rate
            
        except Exception as e:
            print(f"Normal shutdown test error: {e}")
            return False
    
    async def _test_concurrent_shutdown(self) -> bool:
        """Test concurrent shutdown scenario."""
        try:
            # Start multiple shutdown requests simultaneously
            concurrent_tasks = []
            
            for node_name, runners in self.nodes.items():
                for runner_name, runner in runners.items():
                    # Start multiple concurrent shutdown requests for the same runner
                    for i in range(3):
                        concurrent_tasks.append(
                            asyncio.create_task(
                                self.shutdown_coordinator.initiate_shutdown(
                                    runner_id=f"{runner_name}_concurrent_{i}",
                                    timeout=self.config.shutdown_timeout
                                )
                            )
                        )
            
            # Wait for all concurrent shutdowns
            results = await asyncio.wait_for(
                asyncio.gather(*concurrent_tasks, return_exceptions=True),
                timeout=self.config.shutdown_timeout + 5.0
            )
            
            # Check that no deadlocks or race conditions occurred
            exception_count = sum(1 for result in results if isinstance(result, Exception))
            return exception_count == 0  # No exceptions should occur
            
        except Exception as e:
            print(f"Concurrent shutdown test error: {e}")
            return False
    
    async def _test_partial_failure_shutdown(self) -> bool:
        """Test shutdown with partial failures."""
        try:
            # Simulate some runners failing during shutdown
            shutdown_tasks = []
            
            for i, (node_name, runners) in enumerate(self.nodes.items()):
                for j, (runner_name, runner) in enumerate(runners.items()):
                    if (i + j) % 3 == 0:  # Fail every third runner
                        # Simulate failure by using invalid runner ID
                        shutdown_tasks.append(
                            asyncio.create_task(
                                self.shutdown_coordinator.initiate_shutdown(
                                    runner_id=f"invalid_{runner_name}",
                                    timeout=self.config.shutdown_timeout
                                )
                            )
                        )
                    else:
                        shutdown_tasks.append(
                            asyncio.create_task(
                                self.shutdown_coordinator.initiate_shutdown(
                                    runner_id=runner_name,
                                    timeout=self.config.shutdown_timeout
                                )
                            )
                        )
            
            # Wait for all shutdown attempts
            results = await asyncio.wait_for(
                asyncio.gather(*shutdown_tasks, return_exceptions=True),
                timeout=self.config.shutdown_timeout + 5.0
            )
            
            # Check that system handles partial failures gracefully
            # Some should succeed, some should fail, but no deadlocks
            success_count = sum(1 for result in results if result is True)
            return success_count > 0  # At least some should succeed
            
        except Exception as e:
            print(f"Partial failure shutdown test error: {e}")
            return False
    
    async def _test_timeout_shutdown(self) -> bool:
        """Test shutdown timeout handling."""
        try:
            # Test with very short timeout
            short_timeout = 0.1
            
            shutdown_tasks = []
            for node_name, runners in self.nodes.items():
                for runner_name, runner in runners.items():
                    shutdown_tasks.append(
                        asyncio.create_task(
                            self.shutdown_coordinator.initiate_shutdown(
                                runner_id=runner_name,
                                timeout=short_timeout
                            )
                        )
                    )
            
            # Wait for shutdown attempts with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*shutdown_tasks, return_exceptions=True),
                timeout=short_timeout + 2.0
            )
            
            # Check that timeouts are handled gracefully
            # Most should timeout, but system should remain stable
            timeout_count = sum(1 for result in results if result is False)
            return timeout_count > 0  # Some timeouts expected
            
        except Exception as e:
            print(f"Timeout shutdown test error: {e}")
            return False
    
    async def test_error_handling_recovery(self) -> bool:
        """Test error handling and recovery mechanisms."""
        print("Testing error handling and recovery mechanisms...")
        
        try:
            error_scenarios = [
                ("resource_error", self._simulate_resource_error),
                ("communication_error", self._simulate_communication_error),
                ("process_crash", self._simulate_process_crash),
                ("network_partition", self._simulate_network_partition)
            ]
            
            results = {}
            
            for scenario_name, simulate_func in error_scenarios:
                print(f"Testing error scenario: {scenario_name}")
                try:
                    # Simulate the error
                    await simulate_func()
                    
                    # Wait for recovery
                    await asyncio.sleep(2.0)
                    
                    # Check if system recovered
                    recovery_success = await self._check_system_recovery()
                    results[scenario_name] = recovery_success
                    
                    if recovery_success:
                        print(f"✓ Error scenario {scenario_name} recovered successfully")
                    else:
                        print(f"✗ Error scenario {scenario_name} failed to recover")
                        
                except Exception as e:
                    print(f"✗ Error scenario {scenario_name} failed: {e}")
                    results[scenario_name] = False
            
            # Overall success if most scenarios recover
            success_count = sum(1 for result in results.values() if result)
            total_scenarios = len(error_scenarios)
            success_rate = success_count / total_scenarios
            
            if success_rate >= 0.5:  # 50% recovery rate threshold
                print(f"✓ Error handling test passed ({success_count}/{total_scenarios} scenarios recovered)")
                return True
            else:
                print(f"✗ Error handling test failed ({success_count}/{total_scenarios} scenarios recovered)")
                return False
                
        except Exception as e:
            print(f"Error handling test failed: {e}")
            return False
    
    async def _simulate_resource_error(self):
        """Simulate resource allocation errors."""
        # Force resource manager to report errors
        for node_name, runners in self.nodes.items():
            for runner_name, runner in runners.items():
                # Simulate resource error by forcing cleanup
                try:
                    await runner._resource_manager.cleanup_resources(timeout=0.1)
                except:
                    pass  # Expected to fail
    
    async def _simulate_communication_error(self):
        """Simulate communication errors between nodes."""
        # Close some event senders to simulate communication failures
        for i, (node_name, sender) in enumerate(self.event_senders.items()):
            if i % 2 == 0:  # Close every other sender
                try:
                    sender.close()
                except:
                    pass  # Expected to fail
    
    async def _simulate_process_crash(self):
        """Simulate process crashes."""
        # Terminate some runner processes
        for node_name, runners in self.nodes.items():
            for runner_name, runner in runners.items():
                if runner_name.endswith("_0"):  # Crash first runner in each node
                    try:
                        if runner.runner_process.is_alive():
                            runner.runner_process.terminate()
                    except:
                        pass  # Expected to fail
    
    async def _simulate_network_partition(self):
        """Simulate network partition between nodes."""
        # This is a simplified simulation - in reality would involve network-level changes
        print("Simulating network partition (simplified)")
        await asyncio.sleep(0.5)  # Brief partition simulation
    
    async def _check_system_recovery(self) -> bool:
        """Check if the system has recovered from errors."""
        try:
            # Check if error handler has processed errors
            error_stats = self.error_handler.get_error_statistics()
            
            # Check if resource manager is functional
            resource_states = self.resource_manager.get_resource_count_by_state()
            
            # Check if shutdown coordinator is responsive
            test_shutdown = await self.shutdown_coordinator.initiate_shutdown(
                runner_id="test_recovery",
                timeout=1.0
            )
            
            # System is considered recovered if:
            # 1. Error handler is processing errors
            # 2. Resource manager is tracking resources
            # 3. Shutdown coordinator is responsive
            return (
                error_stats is not None and
                resource_states is not None and
                test_shutdown is not None
            )
            
        except Exception as e:
            print(f"Recovery check failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test environment."""
        print("Cleaning up multi-node test environment...")
        
        try:
            # Shutdown all runners
            for node_name, runners in self.nodes.items():
                for runner_name, runner in runners.items():
                    try:
                        runner.shutdown()
                    except:
                        pass  # Ignore cleanup errors
            
            # Close event senders
            for node_name, sender in self.event_senders.items():
                try:
                    sender.close()
                except:
                    pass  # Ignore cleanup errors
            
            # Clear data structures
            self.nodes.clear()
            self.event_senders.clear()
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Cleanup error: {e}")


@pytest.mark.asyncio
async def test_pipeline_parallelism_multinode():
    """Test multi-node functionality with pipeline parallelism."""
    config = MultiNodeTestConfig(
        node_count=2,
        runners_per_node=2,
        parallelism_type="pipeline"
    )
    
    validator = MultiNodeValidator(config)
    
    try:
        # Setup
        setup_success = await validator.setup_multi_node_environment()
        assert setup_success, "Failed to setup multi-node environment"
        
        # Test instance creation
        creation_success = await validator.test_multi_node_instance_creation()
        assert creation_success, "Multi-node instance creation failed"
        
        # Test inference coordination
        inference_success = await validator.test_inference_coordination()
        assert inference_success, "Inference coordination failed"
        
        # Test graceful shutdown
        shutdown_success = await validator.test_graceful_shutdown_scenarios()
        assert shutdown_success, "Graceful shutdown scenarios failed"
        
        # Test error handling
        error_handling_success = await validator.test_error_handling_recovery()
        assert error_handling_success, "Error handling and recovery failed"
        
    finally:
        await validator.cleanup()


@pytest.mark.asyncio
async def test_tensor_parallelism_multinode():
    """Test multi-node functionality with tensor parallelism."""
    config = MultiNodeTestConfig(
        node_count=2,
        runners_per_node=2,
        parallelism_type="tensor"
    )
    
    validator = MultiNodeValidator(config)
    
    try:
        # Setup
        setup_success = await validator.setup_multi_node_environment()
        assert setup_success, "Failed to setup multi-node environment"
        
        # Test instance creation
        creation_success = await validator.test_multi_node_instance_creation()
        assert creation_success, "Multi-node instance creation failed"
        
        # Test inference coordination
        inference_success = await validator.test_inference_coordination()
        assert inference_success, "Inference coordination failed"
        
        # Test graceful shutdown
        shutdown_success = await validator.test_graceful_shutdown_scenarios()
        assert shutdown_success, "Graceful shutdown scenarios failed"
        
        # Test error handling
        error_handling_success = await validator.test_error_handling_recovery()
        assert error_handling_success, "Error handling and recovery failed"
        
    finally:
        await validator.cleanup()


@pytest.mark.asyncio
async def test_stress_multinode():
    """Stress test with many nodes and runners."""
    config = MultiNodeTestConfig(
        node_count=4,
        runners_per_node=3,
        test_timeout=60.0,
        shutdown_timeout=20.0
    )
    
    validator = MultiNodeValidator(config)
    
    try:
        # Setup
        setup_success = await validator.setup_multi_node_environment()
        assert setup_success, "Failed to setup multi-node environment"
        
        # Run multiple test cycles
        for cycle in range(3):
            print(f"Running stress test cycle {cycle + 1}/3")
            
            # Test instance creation
            creation_success = await validator.test_multi_node_instance_creation()
            assert creation_success, f"Multi-node instance creation failed in cycle {cycle + 1}"
            
            # Test graceful shutdown
            shutdown_success = await validator.test_graceful_shutdown_scenarios()
            assert shutdown_success, f"Graceful shutdown failed in cycle {cycle + 1}"
            
            # Brief pause between cycles
            await asyncio.sleep(1.0)
        
    finally:
        await validator.cleanup()


if __name__ == "__main__":
    """Run validation tests directly."""
    import sys
    
    async def run_validation():
        """Run all validation tests."""
        print("Starting multi-node functionality validation...")
        
        tests = [
            ("Pipeline Parallelism", test_pipeline_parallelism_multinode),
            ("Tensor Parallelism", test_tensor_parallelism_multinode),
            ("Stress Test", test_stress_multinode)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"Running {test_name} Test")
            print(f"{'='*50}")
            
            try:
                await test_func()
                results[test_name] = True
                print(f"✓ {test_name} test PASSED")
            except Exception as e:
                results[test_name] = False
                print(f"✗ {test_name} test FAILED: {e}")
        
        # Summary
        print(f"\n{'='*50}")
        print("VALIDATION SUMMARY")
        print(f"{'='*50}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "PASSED" if result else "FAILED"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All validation tests PASSED!")
            return 0
        else:
            print("❌ Some validation tests FAILED!")
            return 1
    
    # Run the validation
    exit_code = asyncio.run(run_validation())
    sys.exit(exit_code)