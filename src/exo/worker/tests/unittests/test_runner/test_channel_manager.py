"""
Unit tests for ChannelManager and ManagedChannel.

Tests atomic channel operations, graceful draining, state synchronization,
and error recovery mechanisms.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from exo.worker.runner.channel_manager import (
    ChannelInfo,
    ChannelManager,
    ChannelState,
    ManagedChannel,
    get_channel_manager,
    reset_channel_manager,
)
from exo.worker.runner.resource_manager import ResourceManager


class MockMpSender:
    """Mock MpSender for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.closed = False
        self.messages = []
        self.send_count = 0

    def send_nowait(self, item):
        """Mock non-blocking send."""
        if self.closed:
            raise RuntimeError("Sender is closed")
        if self.should_fail:
            raise RuntimeError("Send failed")

        self.messages.append(item)
        self.send_count += 1

    def close(self):
        """Mock close method."""
        self.closed = True

    def join(self):
        """Mock join method."""


class MockMpReceiver:
    """Mock MpReceiver for testing."""

    def __init__(self, messages=None, should_fail: bool = False):
        self.messages = list(messages or [])
        self.should_fail = should_fail
        self.closed = False
        self.receive_count = 0

    def receive_nowait(self):
        """Mock non-blocking receive."""
        if self.closed:
            raise RuntimeError("Receiver is closed")
        if self.should_fail:
            raise RuntimeError("Receive failed")

        if not self.messages:
            raise RuntimeError("No messages available")

        self.receive_count += 1
        return self.messages.pop(0)

    def close(self):
        """Mock close method."""
        self.closed = True

    def join(self):
        """Mock join method."""


class TestManagedChannel:
    """Test cases for ManagedChannel."""

    @pytest.fixture
    def mock_channel_creation(self):
        """Mock the mp_channel creation."""
        with patch("exo.worker.runner.channel_manager.mp_channel") as mock_mp_channel:
            sender = MockMpSender()
            receiver = MockMpReceiver()
            mock_mp_channel.return_value = (sender, receiver)
            yield sender, receiver

    def test_initialization(self, mock_channel_creation):
        """Test ManagedChannel initialization."""
        sender, receiver = mock_channel_creation

        channel = ManagedChannel("test_channel", buffer_size=100)

        assert channel.name == "test_channel"
        assert channel.buffer_size == 100
        assert channel.channel_id.startswith("channel_")
        assert channel._state == ChannelState.ACTIVE
        assert channel._sender is sender
        assert channel._receiver is receiver

    def test_initialization_with_custom_id(self, mock_channel_creation):
        """Test initialization with custom channel ID."""
        channel = ManagedChannel("test_channel", channel_id="custom_id")

        assert channel.channel_id == "custom_id"

    def test_get_sender_active_channel(self, mock_channel_creation):
        """Test getting sender from active channel."""
        sender, receiver = mock_channel_creation
        channel = ManagedChannel("test_channel")

        result_sender = channel.get_sender()

        assert result_sender is sender
        assert channel._sender_count == 2  # Initial 1 + 1 from get_sender

    def test_get_sender_inactive_channel(self, mock_channel_creation):
        """Test getting sender from inactive channel."""
        channel = ManagedChannel("test_channel")
        channel._state = ChannelState.CLOSED

        with pytest.raises(
            RuntimeError, match="Cannot get sender for channel test_channel in state"
        ):
            channel.get_sender()

    def test_get_receiver_active_channel(self, mock_channel_creation):
        """Test getting receiver from active channel."""
        sender, receiver = mock_channel_creation
        channel = ManagedChannel("test_channel")

        result_receiver = channel.get_receiver()

        assert result_receiver is receiver
        assert channel._receiver_count == 2  # Initial 1 + 1 from get_receiver

    def test_get_receiver_inactive_channel(self, mock_channel_creation):
        """Test getting receiver from inactive channel."""
        channel = ManagedChannel("test_channel")
        channel._state = ChannelState.CLOSED

        with pytest.raises(
            RuntimeError, match="Cannot get receiver for channel test_channel in state"
        ):
            channel.get_receiver()

    def test_send_safe_success(self, mock_channel_creation):
        """Test successful safe send."""
        sender, receiver = mock_channel_creation
        channel = ManagedChannel("test_channel")

        result = channel.send_safe("test_message")

        assert result is True
        assert channel._messages_sent == 1
        assert "test_message" in sender.messages

    def test_send_safe_closed_channel(self, mock_channel_creation):
        """Test safe send on closed channel."""
        channel = ManagedChannel("test_channel")
        channel._state = ChannelState.CLOSED

        result = channel.send_safe("test_message")

        assert result is False
        assert channel._messages_sent == 0

    def test_send_safe_failure(self, mock_channel_creation):
        """Test safe send with sender failure."""
        sender, receiver = mock_channel_creation
        sender.should_fail = True
        channel = ManagedChannel("test_channel")

        result = channel.send_safe("test_message", timeout=0.1)

        assert result is False
        assert channel._error_count > 0
        assert channel._last_error is not None

    def test_receive_safe_success(self, mock_channel_creation):
        """Test successful safe receive."""
        sender, receiver = mock_channel_creation
        receiver.messages = ["test_message"]
        channel = ManagedChannel("test_channel")

        success, message = channel.receive_safe()

        assert success is True
        assert message == "test_message"
        assert channel._messages_received == 1

    def test_receive_safe_closed_channel(self, mock_channel_creation):
        """Test safe receive on closed channel."""
        channel = ManagedChannel("test_channel")
        channel._state = ChannelState.CLOSED

        success, message = channel.receive_safe()

        assert success is False
        assert message is None

    def test_receive_safe_no_messages(self, mock_channel_creation):
        """Test safe receive with no messages available."""
        sender, receiver = mock_channel_creation
        channel = ManagedChannel("test_channel")

        success, message = channel.receive_safe(timeout=0.1)

        assert success is False
        assert message is None

    def test_receive_safe_failure(self, mock_channel_creation):
        """Test safe receive with receiver failure."""
        sender, receiver = mock_channel_creation
        receiver.should_fail = True
        channel = ManagedChannel("test_channel")

        success, message = channel.receive_safe(timeout=0.1)

        assert success is False
        assert message is None
        assert channel._error_count > 0

    def test_drain_success(self, mock_channel_creation):
        """Test successful channel draining."""
        sender, receiver = mock_channel_creation
        receiver.messages = ["msg1", "msg2", "msg3"]
        channel = ManagedChannel("test_channel")

        result = channel.drain()

        assert result.success is True
        assert result.messages_drained == 3
        assert channel._state == ChannelState.DRAINING
        assert len(receiver.messages) == 0

    def test_drain_closed_channel(self, mock_channel_creation):
        """Test draining closed channel."""
        channel = ManagedChannel("test_channel")
        channel._state = ChannelState.CLOSED

        result = channel.drain()

        assert result.success is False
        assert result.error is not None

    def test_drain_with_error(self, mock_channel_creation):
        """Test draining with receiver error."""
        sender, receiver = mock_channel_creation
        receiver.should_fail = True
        channel = ManagedChannel("test_channel")

        result = channel.drain()

        assert result.success is False
        assert result.error is not None
        assert channel._state == ChannelState.ACTIVE  # Should restore state on error

    def test_close_success(self, mock_channel_creation):
        """Test successful channel closure."""
        sender, receiver = mock_channel_creation
        channel = ManagedChannel("test_channel")

        result = channel.close()

        assert result is True
        assert channel._state == ChannelState.CLOSED
        assert sender.closed is True
        assert receiver.closed is True

    def test_close_already_closed(self, mock_channel_creation):
        """Test closing already closed channel."""
        channel = ManagedChannel("test_channel")
        channel._state = ChannelState.CLOSED

        result = channel.close()

        assert result is True

    def test_close_with_draining(self, mock_channel_creation):
        """Test closing with draining first."""
        sender, receiver = mock_channel_creation
        receiver.messages = ["msg1", "msg2"]
        channel = ManagedChannel("test_channel")

        result = channel.close(drain_first=True)

        assert result is True
        assert channel._state == ChannelState.CLOSED
        assert len(receiver.messages) == 0  # Should be drained

    def test_close_with_error(self, mock_channel_creation):
        """Test closing with error during close."""
        sender, receiver = mock_channel_creation

        # Make sender.close() fail
        def failing_close():
            raise RuntimeError("Close failed")

        sender.close = failing_close

        channel = ManagedChannel("test_channel")

        result = channel.close()

        assert result is False
        assert channel._state == ChannelState.ERROR

    def test_get_info(self, mock_channel_creation):
        """Test getting channel information."""
        channel = ManagedChannel("test_channel", buffer_size=500)
        channel._messages_sent = 10
        channel._messages_received = 5

        info = channel.get_info()

        assert isinstance(info, ChannelInfo)
        assert info.name == "test_channel"
        assert info.buffer_size == 500
        assert info.messages_sent == 10
        assert info.messages_received == 5
        assert info.state == ChannelState.ACTIVE

    def test_is_active(self, mock_channel_creation):
        """Test checking if channel is active."""
        channel = ManagedChannel("test_channel")

        assert channel.is_active() is True

        channel._state = ChannelState.CLOSED
        assert channel.is_active() is False

    def test_get_state(self, mock_channel_creation):
        """Test getting channel state."""
        channel = ManagedChannel("test_channel")

        assert channel.get_state() == ChannelState.ACTIVE

        channel._state = ChannelState.DRAINING
        assert channel.get_state() == ChannelState.DRAINING


class TestChannelManager:
    """Test cases for ChannelManager."""

    @pytest.fixture
    def resource_manager(self):
        """Create a ResourceManager for testing."""
        return ResourceManager()

    @pytest.fixture
    def manager(self, resource_manager):
        """Create a ChannelManager instance for testing."""
        with (
            patch("exo.worker.runner.channel_manager.SharedStateManager"),
            patch("exo.worker.runner.channel_manager.CrossProcessLock"),
        ):
            return ChannelManager(resource_manager=resource_manager)

    @pytest.fixture
    def mock_channel_creation(self):
        """Mock the mp_channel creation."""
        with patch("exo.worker.runner.channel_manager.mp_channel") as mock_mp_channel:
            sender = MockMpSender()
            receiver = MockMpReceiver()
            mock_mp_channel.return_value = (sender, receiver)
            yield sender, receiver

    @pytest.fixture(autouse=True)
    def reset_global_manager(self):
        """Reset global manager after each test."""
        yield
        reset_channel_manager()

    def test_initialization(self, resource_manager):
        """Test ChannelManager initialization."""
        with (
            patch("exo.worker.runner.channel_manager.SharedStateManager"),
            patch("exo.worker.runner.channel_manager.CrossProcessLock"),
        ):
            manager = ChannelManager(resource_manager=resource_manager)

            assert len(manager._channels) == 0
            assert manager._resource_manager is resource_manager
            assert manager._channels_created == 0
            assert manager._channels_closed == 0

    @pytest.mark.asyncio
    async def test_create_channel(self, manager, mock_channel_creation):
        """Test creating a new channel."""
        channel = await manager.create_channel("test_channel", buffer_size=200)

        assert isinstance(channel, ManagedChannel)
        assert channel.name == "test_channel"
        assert channel.buffer_size == 200
        assert "test_channel" in manager._channels
        assert manager._channels_created == 1

    @pytest.mark.asyncio
    async def test_create_channel_duplicate_name(self, manager, mock_channel_creation):
        """Test creating channel with duplicate name."""
        await manager.create_channel("test_channel")

        with pytest.raises(ValueError, match="Channel test_channel already exists"):
            await manager.create_channel("test_channel")

    @pytest.mark.asyncio
    async def test_get_channel(self, manager, mock_channel_creation):
        """Test getting an existing channel."""
        created_channel = await manager.create_channel("test_channel")

        retrieved_channel = await manager.get_channel("test_channel")

        assert retrieved_channel is created_channel

    @pytest.mark.asyncio
    async def test_get_channel_nonexistent(self, manager):
        """Test getting non-existent channel."""
        channel = await manager.get_channel("nonexistent")

        assert channel is None

    @pytest.mark.asyncio
    async def test_close_channel(self, manager, mock_channel_creation):
        """Test closing a channel."""
        await manager.create_channel("test_channel")

        result = await manager.close_channel("test_channel")

        assert result is True
        assert "test_channel" not in manager._channels
        assert manager._channels_closed == 1

    @pytest.mark.asyncio
    async def test_close_channel_nonexistent(self, manager):
        """Test closing non-existent channel."""
        result = await manager.close_channel("nonexistent")

        assert result is False

    def test_is_channel_active(self, manager, mock_channel_creation):
        """Test checking if channel is active."""
        # Non-existent channel
        assert manager.is_channel_active("nonexistent") is False

    @pytest.mark.asyncio
    async def test_is_channel_active_existing(self, manager, mock_channel_creation):
        """Test checking if existing channel is active."""
        channel = await manager.create_channel("test_channel")

        assert manager.is_channel_active("test_channel") is True

        # Close the channel
        channel._state = ChannelState.CLOSED
        assert manager.is_channel_active("test_channel") is False

    @pytest.mark.asyncio
    async def test_drain_channel(self, manager, mock_channel_creation):
        """Test draining a channel."""
        sender, receiver = mock_channel_creation
        receiver.messages = ["msg1", "msg2"]

        await manager.create_channel("test_channel")

        messages = await manager.drain_channel("test_channel")

        assert isinstance(messages, list)
        # Messages are consumed during draining, so we get empty list
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_drain_channel_nonexistent(self, manager):
        """Test draining non-existent channel."""
        messages = await manager.drain_channel("nonexistent")

        assert messages == []

    @pytest.mark.asyncio
    async def test_close_all_channels(self, manager, mock_channel_creation):
        """Test closing all channels."""
        await manager.create_channel("channel1")
        await manager.create_channel("channel2")
        await manager.create_channel("channel3")

        results = await manager.close_all_channels()

        assert len(results) == 3
        assert all(results.values())  # All should succeed
        assert len(manager._channels) == 0
        assert manager._channels_closed == 3

    @pytest.mark.asyncio
    async def test_get_channel_info(self, manager, mock_channel_creation):
        """Test getting channel information."""
        await manager.create_channel("test_channel")

        info = manager.get_channel_info("test_channel")

        assert isinstance(info, ChannelInfo)
        assert info.name == "test_channel"

    def test_get_channel_info_nonexistent(self, manager):
        """Test getting info for non-existent channel."""
        info = manager.get_channel_info("nonexistent")

        assert info is None

    @pytest.mark.asyncio
    async def test_get_all_channel_info(self, manager, mock_channel_creation):
        """Test getting information for all channels."""
        await manager.create_channel("channel1")
        await manager.create_channel("channel2")

        all_info = manager.get_all_channel_info()

        assert len(all_info) == 2
        assert "channel1" in all_info
        assert "channel2" in all_info
        assert isinstance(all_info["channel1"], ChannelInfo)
        assert isinstance(all_info["channel2"], ChannelInfo)

    @pytest.mark.asyncio
    async def test_get_statistics(self, manager, mock_channel_creation):
        """Test getting channel manager statistics."""
        await manager.create_channel("channel1")
        await manager.create_channel("channel2")
        await manager.close_channel("channel1")

        stats = manager.get_statistics()

        assert stats["active_channels"] == 1
        assert stats["channels_created"] == 2
        assert stats["channels_closed"] == 1
        assert "total_messages_sent" in stats
        assert "total_messages_received" in stats

    @pytest.mark.asyncio
    async def test_health_check(self, manager, mock_channel_creation):
        """Test health check functionality."""
        # Create some channels
        await manager.create_channel("healthy_channel")
        channel2 = await manager.create_channel("unhealthy_channel")

        # Make one channel unhealthy (old last activity)
        old_time = datetime.now() - timedelta(minutes=10)
        channel2._last_activity = old_time

        health = await manager.health_check()

        assert "healthy_channels" in health
        assert "unhealthy_channels" in health
        assert "channel_details" in health
        assert "overall_health" in health

        # Should have details for both channels
        assert "healthy_channel" in health["channel_details"]
        assert "unhealthy_channel" in health["channel_details"]

    @pytest.mark.asyncio
    async def test_health_check_no_channels(self, manager):
        """Test health check with no channels."""
        health = await manager.health_check()

        assert health["overall_health"] == "no_channels"
        assert health["healthy_channels"] == 0
        assert health["unhealthy_channels"] == 0

    def test_global_manager(self):
        """Test global manager instance management."""
        # Get global instance
        manager1 = get_channel_manager()
        manager2 = get_channel_manager()

        # Should be the same instance
        assert manager1 is manager2

        # Reset and get new instance
        reset_channel_manager()
        manager3 = get_channel_manager()

        # Should be different instance
        assert manager3 is not manager1

    @pytest.mark.asyncio
    async def test_update_shared_state(self, manager, mock_channel_creation):
        """Test updating shared state for cross-process coordination."""
        await manager.create_channel("test_channel")

        # This should not raise an exception
        await manager._update_shared_state()

    @pytest.mark.asyncio
    async def test_cleanup_channel_integration(self, manager, mock_channel_creation):
        """Test cleanup channel method used by resource manager."""
        await manager.create_channel("test_channel")

        # This should close the channel
        await manager._cleanup_channel("test_channel")

        assert "test_channel" not in manager._channels
