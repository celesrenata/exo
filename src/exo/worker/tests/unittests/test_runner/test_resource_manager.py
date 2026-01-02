"""
Unit tests for ResourceManager.

Tests resource lifecycle tracking, dependency-aware cleanup ordering,
timeout handling, and error recovery mechanisms.
"""

import asyncio
import weakref
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.worker.runner.resource_manager import (
    CleanupResult,
    ResourceHandle,
    ResourceManager,
    ResourceState,
    ResourceType,
    get_resource_manager,
    reset_resource_manager,
)


class MockResource:
    """Mock resource for testing."""
    
    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.closed = False
        self.cleanup_called = False
    
    def close(self):
        """Mock close method."""
        if self.should_fail:
            raise RuntimeError(f"Failed to close {self.name}")
        self.closed = True
        self.cleanup_called = True
    
    async def async_close(self):
        """Mock async close method."""
        if self.should_fail:
            raise RuntimeError(f"Failed to async close {self.name}")
        self.closed = True
        self.cleanup_called = True


class TestResourceManager:
    """Test cases for ResourceManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a ResourceManager instance for testing."""
        return ResourceManager()
    
    @pytest.fixture(autouse=True)
    def reset_global_manager(self):
        """Reset global manager after each test."""
        yield
        reset_resource_manager()
    
    def test_initialization(self, manager):
        """Test ResourceManager initialization."""
        assert len(manager._resources) == 0
        assert len(manager._resource_refs) == 0
        assert manager._cleanup_in_progress is False
    
    def test_register_resource_sync_cleanup(self, manager):
        """Test registering a resource with synchronous cleanup."""
        resource = MockResource("test_resource")
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=10,
            cleanup_func=resource.close,
            resource_id="test_resource_1"
        )
        
        assert handle.id == "test_resource_1"
        assert handle.resource_type == ResourceType.CUSTOM
        assert handle.cleanup_order == 10
        assert handle.state == ResourceState.ACTIVE
        assert handle.cleanup_func == resource.close
        
        # Verify resource is tracked
        assert "test_resource_1" in manager._resources
        assert "test_resource_1" in manager._resource_refs
    
    def test_register_resource_async_cleanup(self, manager):
        """Test registering a resource with asynchronous cleanup."""
        resource = MockResource("test_resource")
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=10,
            async_cleanup_func=resource.async_close,
            resource_id="test_resource_1"
        )
        
        assert handle.async_cleanup_func == resource.async_close
        assert handle.cleanup_func is None
    
    def test_register_resource_auto_id(self, manager):
        """Test registering a resource with auto-generated ID."""
        resource = MockResource("test_resource")
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.QUEUE,
            cleanup_order=10,
            cleanup_func=resource.close
        )
        
        assert handle.id.startswith("queue_")
        assert len(handle.id) > len("queue_")
    
    def test_register_resource_with_dependencies(self, manager):
        """Test registering a resource with dependencies."""
        resource = MockResource("test_resource")
        dependencies = {"dep1", "dep2"}
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=10,
            cleanup_func=resource.close,
            dependencies=dependencies
        )
        
        assert handle.dependencies == dependencies
    
    def test_register_resource_validation_error(self, manager):
        """Test validation error when no cleanup function provided."""
        resource = MockResource("test_resource")
        
        with pytest.raises(ValueError, match="Either cleanup_func or async_cleanup_func must be provided"):
            manager.register_resource(
                resource=resource,
                resource_type=ResourceType.CUSTOM,
                cleanup_order=10
            )
    
    @pytest.mark.asyncio
    async def test_cleanup_single_resource_sync(self, manager):
        """Test cleanup of a single resource with sync cleanup function."""
        resource = MockResource("test_resource")
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=10,
            cleanup_func=resource.close,
            resource_id="test_resource_1"
        )
        
        success = await manager.cleanup_single_resource_by_id("test_resource_1")
        
        assert success is True
        assert resource.cleanup_called is True
        assert handle.state == ResourceState.CLOSED
    
    @pytest.mark.asyncio
    async def test_cleanup_single_resource_async(self, manager):
        """Test cleanup of a single resource with async cleanup function."""
        resource = MockResource("test_resource")
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=10,
            async_cleanup_func=resource.async_close,
            resource_id="test_resource_1"
        )
        
        success = await manager.cleanup_single_resource_by_id("test_resource_1")
        
        assert success is True
        assert resource.cleanup_called is True
        assert handle.state == ResourceState.CLOSED
    
    @pytest.mark.asyncio
    async def test_cleanup_single_resource_failure(self, manager):
        """Test cleanup failure handling."""
        resource = MockResource("test_resource", should_fail=True)
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=10,
            cleanup_func=resource.close,
            resource_id="test_resource_1"
        )
        
        success = await manager.cleanup_single_resource_by_id("test_resource_1")
        
        assert success is False
        assert handle.state == ResourceState.ERROR
    
    @pytest.mark.asyncio
    async def test_cleanup_single_resource_timeout(self, manager):
        """Test cleanup timeout handling."""
        resource = MockResource("test_resource")
        
        async def slow_cleanup():
            await asyncio.sleep(2.0)
            resource.cleanup_called = True
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=10,
            async_cleanup_func=slow_cleanup,
            resource_id="test_resource_1"
        )
        
        success = await manager.cleanup_single_resource_by_id("test_resource_1", timeout=0.1)
        
        assert success is False
        assert handle.state == ResourceState.ERROR
    
    @pytest.mark.asyncio
    async def test_cleanup_resources_simple_order(self, manager):
        """Test cleanup of multiple resources in simple order."""
        resources = []
        
        # Register resources in reverse cleanup order
        for i in range(3):
            resource = MockResource(f"resource_{i}")
            resources.append(resource)
            
            manager.register_resource(
                resource=resource,
                resource_type=ResourceType.CUSTOM,
                cleanup_order=2 - i,  # 2, 1, 0
                cleanup_func=resource.close,
                resource_id=f"resource_{i}"
            )
        
        result = await manager.cleanup_resources()
        
        assert result.success is True
        assert len(result.cleaned_resources) == 3
        assert len(result.failed_resources) == 0
        
        # Verify all resources were cleaned up
        for resource in resources:
            assert resource.cleanup_called is True
    
    @pytest.mark.asyncio
    async def test_cleanup_resources_with_dependencies(self, manager):
        """Test cleanup with dependency ordering."""
        resources = []
        
        # Create resources with dependencies
        # resource_0 depends on nothing (cleanup order 0)
        # resource_1 depends on resource_0 (cleanup order 1)  
        # resource_2 depends on resource_1 (cleanup order 2)
        
        for i in range(3):
            resource = MockResource(f"resource_{i}")
            resources.append(resource)
            
            dependencies = set()
            if i > 0:
                dependencies.add(f"resource_{i-1}")
            
            manager.register_resource(
                resource=resource,
                resource_type=ResourceType.CUSTOM,
                cleanup_order=i,
                cleanup_func=resource.close,
                dependencies=dependencies,
                resource_id=f"resource_{i}"
            )
        
        result = await manager.cleanup_resources()
        
        assert result.success is True
        assert len(result.cleaned_resources) == 3
        
        # Verify cleanup order (should be reverse dependency order)
        expected_order = ["resource_2", "resource_1", "resource_0"]
        assert result.cleaned_resources == expected_order
    
    @pytest.mark.asyncio
    async def test_cleanup_resources_partial_failure(self, manager):
        """Test cleanup with partial failures."""
        resources = []
        
        for i in range(3):
            # Make middle resource fail
            should_fail = (i == 1)
            resource = MockResource(f"resource_{i}", should_fail=should_fail)
            resources.append(resource)
            
            manager.register_resource(
                resource=resource,
                resource_type=ResourceType.CUSTOM,
                cleanup_order=i,
                cleanup_func=resource.close,
                resource_id=f"resource_{i}"
            )
        
        result = await manager.cleanup_resources()
        
        assert result.success is False
        assert len(result.cleaned_resources) == 2
        assert len(result.failed_resources) == 1
        assert "resource_1" in result.failed_resources
        assert len(result.errors) == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_resources_timeout(self, manager):
        """Test cleanup with overall timeout."""
        resource = MockResource("slow_resource")
        
        async def very_slow_cleanup():
            await asyncio.sleep(5.0)
            resource.cleanup_called = True
        
        manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            async_cleanup_func=very_slow_cleanup,
            resource_id="slow_resource"
        )
        
        result = await manager.cleanup_resources(timeout=0.1)
        
        assert result.success is False
        assert len(result.cleaned_resources) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_concurrent_calls(self, manager):
        """Test that concurrent cleanup calls are handled properly."""
        resource = MockResource("test_resource")
        
        manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            cleanup_func=resource.close,
            resource_id="test_resource"
        )
        
        # Start two concurrent cleanup operations
        task1 = asyncio.create_task(manager.cleanup_resources())
        task2 = asyncio.create_task(manager.cleanup_resources())
        
        result1, result2 = await asyncio.gather(task1, task2)
        
        # First should succeed, second should fail due to cleanup in progress
        assert (result1.success and not result2.success) or (not result1.success and result2.success)
    
    def test_calculate_cleanup_order_simple(self, manager):
        """Test cleanup order calculation without dependencies."""
        # Register resources with different cleanup orders
        for i in range(3):
            resource = MockResource(f"resource_{i}")
            manager.register_resource(
                resource=resource,
                resource_type=ResourceType.CUSTOM,
                cleanup_order=2 - i,  # 2, 1, 0
                cleanup_func=resource.close,
                resource_id=f"resource_{i}"
            )
        
        order = manager._calculate_cleanup_order()
        
        # Should be ordered by cleanup_order (ascending)
        expected = ["resource_2", "resource_1", "resource_0"]
        assert order == expected
    
    def test_calculate_cleanup_order_with_dependencies(self, manager):
        """Test cleanup order calculation with dependencies."""
        # Create dependency chain: A -> B -> C
        resources = ["A", "B", "C"]
        
        for i, name in enumerate(resources):
            resource = MockResource(name)
            dependencies = set()
            if i > 0:
                dependencies.add(resources[i-1])
            
            manager.register_resource(
                resource=resource,
                resource_type=ResourceType.CUSTOM,
                cleanup_order=0,  # Same order, should use dependencies
                cleanup_func=resource.close,
                dependencies=dependencies,
                resource_id=name
            )
        
        order = manager._calculate_cleanup_order()
        
        # Should be reverse dependency order
        expected = ["C", "B", "A"]
        assert order == expected
    
    def test_calculate_cleanup_order_circular_dependencies(self, manager):
        """Test cleanup order with circular dependencies."""
        # Create circular dependency: A -> B -> A
        resource_a = MockResource("A")
        resource_b = MockResource("B")
        
        manager.register_resource(
            resource=resource_a,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            cleanup_func=resource_a.close,
            dependencies={"B"},
            resource_id="A"
        )
        
        manager.register_resource(
            resource=resource_b,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=1,
            cleanup_func=resource_b.close,
            dependencies={"A"},
            resource_id="B"
        )
        
        order = manager._calculate_cleanup_order()
        
        # Should handle circular dependency gracefully
        assert len(order) == 2
        assert "A" in order
        assert "B" in order
    
    def test_is_resource_active(self, manager):
        """Test checking if resource is active."""
        resource = MockResource("test_resource")
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            cleanup_func=resource.close,
            resource_id="test_resource"
        )
        
        assert manager.is_resource_active(handle) is True
        
        handle.state = ResourceState.CLOSED
        assert manager.is_resource_active(handle) is False
    
    def test_get_resource_dependencies(self, manager):
        """Test getting resource dependencies."""
        resource_a = MockResource("A")
        resource_b = MockResource("B")
        
        handle_a = manager.register_resource(
            resource=resource_a,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            cleanup_func=resource_a.close,
            resource_id="A"
        )
        
        handle_b = manager.register_resource(
            resource=resource_b,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=1,
            cleanup_func=resource_b.close,
            dependencies={"A"},
            resource_id="B"
        )
        
        deps = manager.get_resource_dependencies(handle_b)
        assert len(deps) == 1
        assert deps[0] == handle_a
        
        deps = manager.get_resource_dependencies(handle_a)
        assert len(deps) == 0
    
    def test_get_resource_state(self, manager):
        """Test getting resource state."""
        resource = MockResource("test_resource")
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            cleanup_func=resource.close,
            resource_id="test_resource"
        )
        
        state = manager.get_resource_state("test_resource")
        assert state == ResourceState.ACTIVE
        
        state = manager.get_resource_state("nonexistent")
        assert state is None
    
    def test_get_all_resources(self, manager):
        """Test getting all resources."""
        resources = []
        
        for i in range(3):
            resource = MockResource(f"resource_{i}")
            resources.append(resource)
            
            manager.register_resource(
                resource=resource,
                resource_type=ResourceType.CUSTOM,
                cleanup_order=i,
                cleanup_func=resource.close,
                resource_id=f"resource_{i}"
            )
        
        all_resources = manager.get_all_resources()
        
        assert len(all_resources) == 3
        for i in range(3):
            assert f"resource_{i}" in all_resources
            assert all_resources[f"resource_{i}"].id == f"resource_{i}"
    
    def test_unregister_resource(self, manager):
        """Test unregistering a resource."""
        resource = MockResource("test_resource")
        
        manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            cleanup_func=resource.close,
            resource_id="test_resource"
        )
        
        assert "test_resource" in manager._resources
        
        success = manager.unregister_resource("test_resource")
        assert success is True
        assert "test_resource" not in manager._resources
        assert "test_resource" not in manager._resource_refs
        
        # Try to unregister again
        success = manager.unregister_resource("test_resource")
        assert success is False
    
    def test_get_resource_count_by_state(self, manager):
        """Test getting resource count by state."""
        resources = []
        
        for i in range(3):
            resource = MockResource(f"resource_{i}")
            resources.append(resource)
            
            handle = manager.register_resource(
                resource=resource,
                resource_type=ResourceType.CUSTOM,
                cleanup_order=i,
                cleanup_func=resource.close,
                resource_id=f"resource_{i}"
            )
            
            # Set different states
            if i == 1:
                handle.state = ResourceState.CLOSED
            elif i == 2:
                handle.state = ResourceState.ERROR
        
        counts = manager.get_resource_count_by_state()
        
        assert counts[ResourceState.ACTIVE] == 1
        assert counts[ResourceState.CLOSED] == 1
        assert counts[ResourceState.ERROR] == 1
        assert counts[ResourceState.DRAINING] == 0
        assert counts[ResourceState.CLOSING] == 0
    
    @pytest.mark.asyncio
    async def test_wait_for_state(self, manager):
        """Test waiting for resource to reach specific state."""
        resource = MockResource("test_resource")
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            cleanup_func=resource.close,
            resource_id="test_resource"
        )
        
        # Start waiting task
        wait_task = asyncio.create_task(
            manager.wait_for_state("test_resource", ResourceState.CLOSED)
        )
        
        # Give it a moment to start waiting
        await asyncio.sleep(0.1)
        
        # Change state
        handle.state = ResourceState.CLOSED
        
        result = await wait_task
        assert result is True
    
    @pytest.mark.asyncio
    async def test_wait_for_state_timeout(self, manager):
        """Test waiting for state with timeout."""
        resource = MockResource("test_resource")
        
        manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            cleanup_func=resource.close,
            resource_id="test_resource"
        )
        
        result = await manager.wait_for_state("test_resource", ResourceState.CLOSED, timeout=0.1)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_try_default_cleanup(self, manager):
        """Test trying default cleanup methods on resources."""
        resource = MockResource("test_resource")
        
        # Test with close method
        await manager._try_default_cleanup(resource, timeout=1.0)
        assert resource.cleanup_called is True
        
        # Test with resource that has no cleanup methods
        class NoCleanupResource:
            pass
        
        no_cleanup_resource = NoCleanupResource()
        
        # Should not raise exception
        await manager._try_default_cleanup(no_cleanup_resource, timeout=1.0)
    
    def test_weak_reference_handling(self, manager):
        """Test that weak references are handled properly."""
        resource = MockResource("test_resource")
        
        handle = manager.register_resource(
            resource=resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=0,
            cleanup_func=resource.close,
            resource_id="test_resource"
        )
        
        # Get the resource back
        retrieved_resource = manager._get_resource("test_resource")
        assert retrieved_resource is resource
        
        # Delete the original reference
        del resource
        
        # Resource might still be available through weak reference
        # (depends on garbage collection timing)
        retrieved_resource = manager._get_resource("test_resource")
        # Don't assert anything specific here as it depends on GC timing
    
    def test_global_manager(self):
        """Test global manager instance management."""
        # Get global instance
        manager1 = get_resource_manager()
        manager2 = get_resource_manager()
        
        # Should be the same instance
        assert manager1 is manager2
        
        # Reset and get new instance
        reset_resource_manager()
        manager3 = get_resource_manager()
        
        # Should be different instance
        assert manager3 is not manager1