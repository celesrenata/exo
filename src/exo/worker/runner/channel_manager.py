"""
Race-condition-free channel management for multiprocessing communication.

This module provides atomic channel operations with graceful draining and state synchronization
to prevent "Queue is closed" and "ClosedResourceError" exceptions during shutdown.
"""

import asyncio
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from loguru import logger

from exo.utils.channels import MpReceiver, MpSender, MpState, mp_channel
from exo.worker.runner.resource_manager import ResourceManager, ResourceType, get_resource_manager
from exo.worker.runner.synchronization import CrossProcessLock, SharedStateManager


class ChannelState(Enum):
    """States in the channel lifecycle."""
    ACTIVE = "active"
    DRAINING = "draining"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class ChannelInfo:
    """Information about a managed channel."""
    name: str
    channel_id: str
    state: ChannelState = ChannelState.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    buffer_size: int = 0
    sender_count: int = 0
    receiver_count: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class DrainResult:
    """Result of a channel draining operation."""
    success: bool
    messages_drained: int = 0
    drain_time: float = 0.0
    error: Optional[Exception] = None


class ManagedChannel:
    """
    A managed multiprocessing channel with atomic operations and state tracking.
    
    This wrapper provides race-condition-free operations on top of the existing
    MpSender/MpReceiver classes.
    """
    
    def __init__(self, name: str, buffer_size: int = 1000, channel_id: Optional[str] = None):
        """
        Initialize a managed channel.
        
        Args:
            name: Human-readable name for the channel
            buffer_size: Maximum buffer size for the channel
            channel_id: Optional custom ID, otherwise generated
        """
        self.name = name
        self.channel_id = channel_id or f"channel_{uuid4().hex[:8]}"
        self.buffer_size = buffer_size
        
        # Create the underlying channel
        self._sender, self._receiver = mp_channel[Any](buffer_size)
        
        # State tracking
        self._state = ChannelState.ACTIVE
        self._lock = mp.Lock()
        self._created_at = datetime.now()
        self._last_activity = datetime.now()
        self._messages_sent = 0
        self._messages_received = 0
        self._sender_count = 1
        self._receiver_count = 1
        
        # Error recovery
        self._error_count = 0
        self._last_error: Optional[Exception] = None
        
        logger.debug(f"Created managed channel {self.name} (id: {self.channel_id})")
    
    def get_sender(self) -> MpSender[Any]:
        """
        Get a sender for this channel.
        
        Returns:
            MpSender instance
            
        Raises:
            RuntimeError: If channel is not in ACTIVE state
        """
        with self._lock:
            if self._state != ChannelState.ACTIVE:
                raise RuntimeError(f"Cannot get sender for channel {self.name} in state {self._state}")
            
            self._sender_count += 1
            self._last_activity = datetime.now()
            
        return self._sender
    
    def get_receiver(self) -> MpReceiver[Any]:
        """
        Get a receiver for this channel.
        
        Returns:
            MpReceiver instance
            
        Raises:
            RuntimeError: If channel is not in ACTIVE state
        """
        with self._lock:
            if self._state != ChannelState.ACTIVE:
                raise RuntimeError(f"Cannot get receiver for channel {self.name} in state {self._state}")
            
            self._receiver_count += 1
            self._last_activity = datetime.now()
            
        return self._receiver
    
    def send_safe(self, item: Any, timeout: float = 5.0) -> bool:
        """
        Send an item with state checking and timeout.
        
        Args:
            item: Item to send
            timeout: Maximum time to wait for send
            
        Returns:
            True if sent successfully, False if failed or timeout
        """
        with self._lock:
            if self._state not in (ChannelState.ACTIVE, ChannelState.DRAINING):
                logger.debug(f"Cannot send to channel {self.name} in state {self._state}")
                return False
        
        try:
            start_time = time.time()
            
            # Try non-blocking send first
            try:
                self._sender.send_nowait(item)
                with self._lock:
                    self._messages_sent += 1
                    self._last_activity = datetime.now()
                return True
            except Exception:
                # Fall back to blocking send with timeout
                pass
            
            # Blocking send with timeout simulation
            while time.time() - start_time < timeout:
                try:
                    self._sender.send_nowait(item)
                    with self._lock:
                        self._messages_sent += 1
                        self._last_activity = datetime.now()
                    return True
                except Exception:
                    time.sleep(0.01)  # Brief pause before retry
                    
                    # Check if state changed during wait
                    with self._lock:
                        if self._state not in (ChannelState.ACTIVE, ChannelState.DRAINING):
                            return False
            
            logger.debug(f"Send timeout for channel {self.name}")
            return False
            
        except Exception as e:
            with self._lock:
                self._error_count += 1
                self._last_error = e
            logger.error(f"Send error for channel {self.name}: {e}")
            return False
    
    def receive_safe(self, timeout: float = 5.0) -> Tuple[bool, Any]:
        """
        Receive an item with state checking and timeout.
        
        Args:
            timeout: Maximum time to wait for receive
            
        Returns:
            Tuple of (success, item). If success is False, item is None.
        """
        with self._lock:
            if self._state == ChannelState.CLOSED:
                return False, None
        
        try:
            start_time = time.time()
            
            # Try non-blocking receive first
            try:
                item = self._receiver.receive_nowait()
                with self._lock:
                    self._messages_received += 1
                    self._last_activity = datetime.now()
                return True, item
            except Exception:
                # Fall back to blocking receive with timeout
                pass
            
            # Blocking receive with timeout simulation
            while time.time() - start_time < timeout:
                try:
                    item = self._receiver.receive_nowait()
                    with self._lock:
                        self._messages_received += 1
                        self._last_activity = datetime.now()
                    return True, item
                except Exception:
                    time.sleep(0.01)  # Brief pause before retry
                    
                    # Check if state changed during wait
                    with self._lock:
                        if self._state == ChannelState.CLOSED:
                            return False, None
            
            logger.debug(f"Receive timeout for channel {self.name}")
            return False, None
            
        except Exception as e:
            with self._lock:
                self._error_count += 1
                self._last_error = e
            logger.error(f"Receive error for channel {self.name}: {e}")
            return False, None
    
    def drain(self, timeout: float = 5.0) -> DrainResult:
        """
        Drain all messages from the channel.
        
        Args:
            timeout: Maximum time to spend draining
            
        Returns:
            DrainResult with details of the draining operation
        """
        start_time = time.time()
        messages_drained = 0
        
        with self._lock:
            if self._state == ChannelState.CLOSED:
                return DrainResult(success=False, error=RuntimeError("Channel already closed"))
            
            # Transition to draining state
            old_state = self._state
            self._state = ChannelState.DRAINING
        
        try:
            logger.debug(f"Draining channel {self.name}")
            
            # Collect all available messages
            while time.time() - start_time < timeout:
                try:
                    item = self._receiver.receive_nowait()
                    messages_drained += 1
                    logger.debug(f"Drained message from channel {self.name}: {type(item)}")
                except Exception:
                    # No more messages available
                    break
            
            drain_time = time.time() - start_time
            
            logger.debug(f"Drained {messages_drained} messages from channel {self.name} in {drain_time:.3f}s")
            
            return DrainResult(
                success=True,
                messages_drained=messages_drained,
                drain_time=drain_time
            )
            
        except Exception as e:
            # Restore previous state on error
            with self._lock:
                self._state = old_state
                self._error_count += 1
                self._last_error = e
            
            logger.error(f"Error draining channel {self.name}: {e}")
            return DrainResult(success=False, error=e)
    
    def close(self, drain_first: bool = True, drain_timeout: float = 5.0) -> bool:
        """
        Close the channel with optional draining.
        
        Args:
            drain_first: Whether to drain messages before closing
            drain_timeout: Timeout for draining operation
            
        Returns:
            True if closed successfully, False if error
        """
        with self._lock:
            if self._state == ChannelState.CLOSED:
                logger.debug(f"Channel {self.name} already closed")
                return True
            
            logger.debug(f"Closing channel {self.name} (drain_first={drain_first})")
            self._state = ChannelState.CLOSING
        
        try:
            # Drain messages if requested
            if drain_first:
                drain_result = self.drain(drain_timeout)
                if not drain_result.success:
                    logger.warning(f"Failed to drain channel {self.name} before closing: {drain_result.error}")
            
            # Close the underlying channel components
            try:
                self._sender.close()
            except Exception as e:
                logger.debug(f"Error closing sender: {e}")
            
            try:
                self._receiver.close()
            except Exception as e:
                logger.debug(f"Error closing receiver: {e}")
            
            # Join threads if available
            try:
                self._sender.join()
            except Exception as e:
                logger.debug(f"Error joining sender: {e}")
            
            try:
                self._receiver.join()
            except Exception as e:
                logger.debug(f"Error joining receiver: {e}")
            
            with self._lock:
                self._state = ChannelState.CLOSED
            
            logger.debug(f"Successfully closed channel {self.name}")
            return True
            
        except Exception as e:
            with self._lock:
                self._state = ChannelState.ERROR
                self._error_count += 1
                self._last_error = e
            
            logger.error(f"Error closing channel {self.name}: {e}")
            return False
    
    def get_info(self) -> ChannelInfo:
        """
        Get information about the channel.
        
        Returns:
            ChannelInfo with current channel state and statistics
        """
        with self._lock:
            return ChannelInfo(
                name=self.name,
                channel_id=self.channel_id,
                state=self._state,
                created_at=self._created_at,
                buffer_size=self.buffer_size,
                sender_count=self._sender_count,
                receiver_count=self._receiver_count,
                messages_sent=self._messages_sent,
                messages_received=self._messages_received,
                last_activity=self._last_activity
            )
    
    def is_active(self) -> bool:
        """Check if the channel is currently active."""
        with self._lock:
            return self._state == ChannelState.ACTIVE
    
    def get_state(self) -> ChannelState:
        """Get the current channel state."""
        with self._lock:
            return self._state


class ChannelManager:
    """
    Manages multiple channels with atomic operations and coordinated shutdown.
    
    This class provides race-condition-free channel management with graceful
    draining and state synchronization across processes.
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        """
        Initialize the channel manager.
        
        Args:
            resource_manager: Optional ResourceManager instance for lifecycle management
        """
        self._channels: Dict[str, ManagedChannel] = {}
        self._lock = asyncio.Lock()
        self._resource_manager = resource_manager or get_resource_manager()
        
        # Cross-process state synchronization
        self._state_manager = SharedStateManager("channel_manager_state")
        self._process_lock = CrossProcessLock("channel_manager")
        
        # Statistics
        self._channels_created = 0
        self._channels_closed = 0
        self._total_messages_sent = 0
        self._total_messages_received = 0
        
        logger.debug("Initialized ChannelManager")
    
    async def create_channel(self, name: str, buffer_size: int = 1000) -> ManagedChannel:
        """
        Create a new managed channel with atomic operations.
        
        Args:
            name: Human-readable name for the channel
            buffer_size: Maximum buffer size for the channel
            
        Returns:
            ManagedChannel instance
            
        Raises:
            ValueError: If channel with same name already exists
        """
        async with self._lock:
            if name in self._channels:
                raise ValueError(f"Channel {name} already exists")
            
            # Create the managed channel
            channel = ManagedChannel(name, buffer_size)
            self._channels[name] = channel
            self._channels_created += 1
            
            # Register with resource manager for lifecycle management
            self._resource_manager.register_resource(
                resource=channel,
                resource_type=ResourceType.CHANNEL,
                cleanup_order=100,  # Channels should be cleaned up after processes but before files
                async_cleanup_func=lambda: self._cleanup_channel(name),
                resource_id=f"channel_{name}"
            )
            
            # Update cross-process state
            await self._update_shared_state()
            
            logger.info(f"Created channel {name} with buffer size {buffer_size}")
            return channel
    
    async def get_channel(self, name: str) -> Optional[ManagedChannel]:
        """
        Get an existing channel by name.
        
        Args:
            name: Name of the channel to retrieve
            
        Returns:
            ManagedChannel if exists, None otherwise
        """
        async with self._lock:
            return self._channels.get(name)
    
    async def close_channel(self, name: str, drain_timeout: float = 5.0) -> bool:
        """
        Close a channel with graceful draining.
        
        Args:
            name: Name of the channel to close
            drain_timeout: Timeout for draining operation
            
        Returns:
            True if closed successfully, False if error or not found
        """
        async with self._lock:
            channel = self._channels.get(name)
            if not channel:
                logger.warning(f"Channel {name} not found for closing")
                return False
            
            logger.info(f"Closing channel {name}")
            
            # Close the channel (this will drain first by default)
            success = channel.close(drain_first=True, drain_timeout=drain_timeout)
            
            if success:
                # Remove from tracking
                del self._channels[name]
                self._channels_closed += 1
                
                # Update statistics
                info = channel.get_info()
                self._total_messages_sent += info.messages_sent
                self._total_messages_received += info.messages_received
                
                # Update cross-process state
                await self._update_shared_state()
                
                logger.info(f"Successfully closed channel {name}")
            else:
                logger.error(f"Failed to close channel {name}")
            
            return success
    
    def is_channel_active(self, name: str) -> bool:
        """
        Check if a channel is currently active.
        
        Args:
            name: Name of the channel to check
            
        Returns:
            True if channel exists and is active, False otherwise
        """
        channel = self._channels.get(name)
        return channel is not None and channel.is_active()
    
    async def drain_channel(self, name: str, timeout: float = 5.0) -> List[Any]:
        """
        Drain all messages from a channel.
        
        Args:
            name: Name of the channel to drain
            timeout: Maximum time to spend draining
            
        Returns:
            List of drained messages, empty list if channel not found or error
        """
        channel = self._channels.get(name)
        if not channel:
            logger.warning(f"Channel {name} not found for draining")
            return []
        
        drain_result = channel.drain(timeout)
        if drain_result.success:
            logger.debug(f"Drained {drain_result.messages_drained} messages from channel {name}")
            return []  # Messages are consumed during draining
        else:
            logger.error(f"Failed to drain channel {name}: {drain_result.error}")
            return []
    
    async def close_all_channels(self, drain_timeout: float = 5.0) -> Dict[str, bool]:
        """
        Close all managed channels.
        
        Args:
            drain_timeout: Timeout for draining each channel
            
        Returns:
            Dictionary mapping channel names to success status
        """
        results = {}
        
        async with self._lock:
            channel_names = list(self._channels.keys())
        
        logger.info(f"Closing {len(channel_names)} channels")
        
        for name in channel_names:
            success = await self.close_channel(name, drain_timeout)
            results[name] = success
        
        return results
    
    def get_channel_info(self, name: str) -> Optional[ChannelInfo]:
        """
        Get information about a channel.
        
        Args:
            name: Name of the channel
            
        Returns:
            ChannelInfo if channel exists, None otherwise
        """
        channel = self._channels.get(name)
        return channel.get_info() if channel else None
    
    def get_all_channel_info(self) -> Dict[str, ChannelInfo]:
        """
        Get information about all channels.
        
        Returns:
            Dictionary mapping channel names to ChannelInfo
        """
        return {name: channel.get_info() for name, channel in self._channels.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get channel manager statistics.
        
        Returns:
            Dictionary with various statistics
        """
        active_channels = len(self._channels)
        total_messages_current = sum(
            info.messages_sent + info.messages_received 
            for info in self.get_all_channel_info().values()
        )
        
        return {
            "active_channels": active_channels,
            "channels_created": self._channels_created,
            "channels_closed": self._channels_closed,
            "total_messages_sent": self._total_messages_sent,
            "total_messages_received": self._total_messages_received,
            "current_messages": total_messages_current
        }
    
    async def _cleanup_channel(self, name: str) -> None:
        """
        Cleanup function for resource manager integration.
        
        Args:
            name: Name of the channel to cleanup
        """
        await self.close_channel(name)
    
    async def _update_shared_state(self) -> None:
        """Update shared state for cross-process coordination."""
        try:
            state_data = {
                "active_channels": list(self._channels.keys()),
                "statistics": self.get_statistics(),
                "last_updated": datetime.now().isoformat()
            }
            
            self._state_manager.set_value("channel_state", state_data)
        except Exception as e:
            logger.debug(f"Failed to update shared state: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all channels.
        
        Returns:
            Dictionary with health check results
        """
        health_info = {
            "healthy_channels": 0,
            "unhealthy_channels": 0,
            "channel_details": {},
            "overall_health": "unknown"
        }
        
        for name, channel in self._channels.items():
            info = channel.get_info()
            is_healthy = (
                info.state == ChannelState.ACTIVE and
                (datetime.now() - info.last_activity).total_seconds() < 300  # 5 minutes
            )
            
            if is_healthy:
                health_info["healthy_channels"] += 1
            else:
                health_info["unhealthy_channels"] += 1
            
            health_info["channel_details"][name] = {
                "state": info.state.value,
                "last_activity": info.last_activity.isoformat(),
                "messages_sent": info.messages_sent,
                "messages_received": info.messages_received,
                "healthy": is_healthy
            }
        
        # Determine overall health
        total_channels = len(self._channels)
        if total_channels == 0:
            health_info["overall_health"] = "no_channels"
        elif health_info["unhealthy_channels"] == 0:
            health_info["overall_health"] = "healthy"
        elif health_info["healthy_channels"] > health_info["unhealthy_channels"]:
            health_info["overall_health"] = "mostly_healthy"
        else:
            health_info["overall_health"] = "unhealthy"
        
        return health_info


# Global instance for cross-module usage
_global_channel_manager: Optional[ChannelManager] = None


def get_channel_manager() -> ChannelManager:
    """Get the global channel manager instance."""
    global _global_channel_manager
    if _global_channel_manager is None:
        _global_channel_manager = ChannelManager()
    return _global_channel_manager


def reset_channel_manager() -> None:
    """Reset the global channel manager (mainly for testing)."""
    global _global_channel_manager
    _global_channel_manager = None