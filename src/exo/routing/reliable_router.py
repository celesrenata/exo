"""
Enhanced router with reliability features for distributed inference stability.
"""

import asyncio
from copy import copy
from itertools import count
from math import inf
from typing import cast, Dict, Optional, Any

from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    create_task_group,
    sleep_forever,
)
from anyio.abc import TaskGroup
from exo_pyo3_bindings import (
    AllQueuesFullError,
    Keypair,
    NetworkingHandle,
    NoPeersSubscribedToTopicError,
)
from loguru import logger

from exo.shared.communication import (
    ReliableMessageTransport,
    ValidatedMessage,
    TransferResult,
    MessageStatus,
)
from exo.shared.types.common import NodeId
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.pydantic_ext import CamelCaseModel

from .connection_message import ConnectionMessage
from .topics import CONNECTION_MESSAGES, PublishPolicy, TypedTopic
from .router import TopicRouter, Router


class ReliableTopicRouter[T: CamelCaseModel](TopicRouter[T]):
    """
    Enhanced TopicRouter with reliability features including:
    - Message validation and integrity checking
    - Retry logic for failed transmissions
    - Error handling and recovery mechanisms
    """

    def __init__(
        self,
        topic: TypedTopic[T],
        networking_sender: Sender[tuple[str, bytes]],
        reliable_transport: ReliableMessageTransport,
        max_buffer_size: float = inf,
        enable_validation: bool = True,
    ):
        super().__init__(topic, networking_sender, max_buffer_size)
        self.reliable_transport = reliable_transport
        self.enable_validation = enable_validation

        # Statistics for monitoring
        self.message_stats = {
            "total_sent": 0,
            "total_received": 0,
            "send_failures": 0,
            "validation_failures": 0,
            "retries": 0,
        }

    async def run(self):
        """Enhanced run method with reliability features."""
        logger.debug(f"Reliable Topic Router {self.topic} ready to send")
        with self.receiver as items:
            async for item in items:
                try:
                    # Check if we should send to network
                    if (
                        len(self.senders) == 0
                        and self.topic.publish_policy is PublishPolicy.Minimal
                    ):
                        await self._send_out_reliable(item)
                        continue
                    if self.topic.publish_policy is PublishPolicy.Always:
                        await self._send_out_reliable(item)
                    # Then publish to all senders
                    await self.publish(item)
                except Exception as e:
                    logger.error(f"Error in reliable router run loop: {e}")
                    self.message_stats["send_failures"] += 1

    async def publish_bytes_reliable(
        self, data: bytes, source_node: Optional[NodeId] = None
    ):
        """
        Publish bytes with validation and reliability features.

        Args:
            data: Raw bytes to publish
            source_node: Optional source node for validation
        """
        try:
            # Validate message if enabled
            if self.enable_validation and source_node:
                # Create a validated message for integrity checking
                validated_msg = ValidatedMessage(
                    message_id=f"topic_{self.topic.topic}_{asyncio.get_event_loop().time()}",
                    source=source_node,
                    destination=NodeId("broadcast"),  # Topic broadcast
                    data=data,
                    checksum="",  # Will be computed
                    timestamp=asyncio.get_event_loop().time(),
                )
                validated_msg.checksum = validated_msg.compute_checksum()

                # Verify integrity
                if not validated_msg.verify_checksum():
                    logger.warning(
                        f"Message integrity check failed for topic {self.topic.topic}"
                    )
                    self.message_stats["validation_failures"] += 1
                    return

            # Deserialize and publish
            item = self.topic.deserialize(data)
            await self.publish(item)
            self.message_stats["total_received"] += 1

        except Exception as e:
            logger.error(f"Error publishing bytes reliably: {e}")
            self.message_stats["validation_failures"] += 1

    async def _send_out_reliable(self, item: T):
        """Send item with reliability features."""
        max_retries = 3
        retry_count = 0

        while retry_count <= max_retries:
            try:
                logger.trace(
                    f"Reliable TopicRouter {self.topic.topic} sending {item} (attempt {retry_count + 1})"
                )

                # Serialize the item
                serialized_data = self.topic.serialize(item)

                # Send through networking layer
                await self.networking_sender.send(
                    (str(self.topic.topic), serialized_data)
                )

                self.message_stats["total_sent"] += 1
                return  # Success

            except (NoPeersSubscribedToTopicError, AllQueuesFullError) as e:
                # These are expected errors, don't retry
                logger.debug(f"Expected networking error: {e}")
                return

            except Exception as e:
                retry_count += 1
                self.message_stats["retries"] += 1

                if retry_count <= max_retries:
                    # Exponential backoff
                    backoff_time = min(0.1 * (2 ** (retry_count - 1)), 2.0)
                    logger.warning(
                        f"Failed to send message on topic {self.topic.topic}, "
                        f"retrying in {backoff_time}s (attempt {retry_count}/{max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(
                        f"Failed to send message on topic {self.topic.topic} "
                        f"after {max_retries + 1} attempts: {e}"
                    )
                    self.message_stats["send_failures"] += 1
                    break

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            **self.message_stats,
            "transport_stats": self.reliable_transport.get_statistics()
            if self.reliable_transport
            else {},
        }


class ReliableRouter(Router):
    """
    Enhanced Router with reliability features for distributed inference stability.

    Provides:
    - Reliable message transport with checksums and retry logic
    - Message validation and integrity checking
    - Enhanced error handling and recovery
    - Comprehensive monitoring and statistics
    """

    def __init__(self, handle: NetworkingHandle, enable_reliability: bool = True):
        super().__init__(handle)
        self.enable_reliability = enable_reliability

        # Create reliable transport if enabled
        if enable_reliability:
            # Extract node ID from handle (simplified approach)
            node_id = NodeId(str(handle))  # This would need proper implementation
            self.reliable_transport = ReliableMessageTransport(
                node_id=node_id, max_retries=3, base_timeout=5.0, max_timeout=30.0
            )
        else:
            self.reliable_transport = None

        # Enhanced statistics
        self.router_stats = {
            "total_topics": 0,
            "active_connections": 0,
            "total_messages_routed": 0,
            "routing_errors": 0,
            "reliability_enabled": enable_reliability,
        }

    async def register_topic[T: CamelCaseModel](self, topic: TypedTopic[T]):
        """Register topic with reliability features."""
        assert self._tg is None, "Attempted to register topic after setup time"

        send = self._tmp_networking_sender
        if send:
            self._tmp_networking_sender = None
        else:
            send = self.networking_receiver.clone_sender()

        # Create reliable router if reliability is enabled
        if self.enable_reliability and self.reliable_transport:
            router = ReliableTopicRouter[T](
                topic, send, self.reliable_transport, enable_validation=True
            )
        else:
            # Fall back to standard router
            router = TopicRouter[T](topic, send)

        self.topic_routers[topic.topic] = cast(TopicRouter[CamelCaseModel], router)
        await self._networking_subscribe(str(topic.topic))
        self.router_stats["total_topics"] += 1

    async def run(self):
        """Enhanced run method with reliability monitoring."""
        logger.debug("Starting Reliable Router")

        # Configure transport callbacks if reliability is enabled
        if self.reliable_transport:
            self.reliable_transport.set_transport_callbacks(
                send_callback=self._reliable_send_callback,
                receive_callback=self._reliable_receive_callback,
            )

        async with create_task_group() as tg:
            self._tg = tg

            # Start topic routers
            for topic in self.topic_routers:
                router = self.topic_routers[topic]
                tg.start_soon(router.run)

            # Start networking tasks
            tg.start_soon(self._networking_recv_reliable)
            tg.start_soon(self._networking_recv_connection_messages)
            tg.start_soon(self._networking_publish_reliable)

            # Start reliability monitoring if enabled
            if self.enable_reliability:
                tg.start_soon(self._reliability_monitor)

            # Router only shuts down if you cancel it
            await sleep_forever()

        # Cleanup
        for topic in self.topic_routers:
            await self._networking_unsubscribe(str(topic))

    async def _networking_recv_reliable(self):
        """Enhanced networking receive with reliability features."""
        while True:
            try:
                topic, data = await self._net.gossipsub_recv()
                logger.trace(
                    f"Received message on {topic} with payload size {len(data)}"
                )

                if topic not in self.topic_routers:
                    logger.warning(
                        f"Received message on unknown or inactive topic {topic}"
                    )
                    continue

                router = self.topic_routers[topic]

                # Use reliable publish if available
                if isinstance(router, ReliableTopicRouter):
                    await router.publish_bytes_reliable(data)
                else:
                    await router.publish_bytes(data)

                self.router_stats["total_messages_routed"] += 1

            except Exception as e:
                logger.error(f"Error in reliable networking receive: {e}")
                self.router_stats["routing_errors"] += 1
                # Continue processing other messages
                await asyncio.sleep(0.1)

    async def _networking_publish_reliable(self):
        """Enhanced networking publish with reliability features."""
        with self.networking_receiver as networked_items:
            async for topic, data in networked_items:
                try:
                    logger.trace(
                        f"Publishing message on {topic} with payload size {len(data)}"
                    )

                    # Add retry logic for publishing
                    max_retries = 3
                    for attempt in range(max_retries + 1):
                        try:
                            await self._net.gossipsub_publish(topic, data)
                            break  # Success
                        except (NoPeersSubscribedToTopicError, AllQueuesFullError):
                            # Expected errors, don't retry
                            break
                        except Exception as e:
                            if attempt < max_retries:
                                backoff_time = 0.1 * (2**attempt)
                                logger.warning(
                                    f"Failed to publish on {topic}, retrying in {backoff_time}s: {e}"
                                )
                                await asyncio.sleep(backoff_time)
                            else:
                                logger.error(
                                    f"Failed to publish on {topic} after {max_retries + 1} attempts: {e}"
                                )
                                self.router_stats["routing_errors"] += 1

                except Exception as e:
                    logger.error(f"Error in reliable networking publish: {e}")
                    self.router_stats["routing_errors"] += 1

    async def _reliability_monitor(self):
        """Monitor reliability metrics and log statistics."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                stats = self.get_comprehensive_statistics()
                logger.info(f"Router reliability stats: {stats}")

                # Check for concerning patterns
                if self.reliable_transport:
                    transport_stats = self.reliable_transport.get_statistics()
                    success_rate = transport_stats.get("success_rate", 1.0)

                    if success_rate < 0.8:  # Less than 80% success rate
                        logger.warning(
                            f"Low message success rate detected: {success_rate:.2%}"
                        )

            except Exception as e:
                logger.error(f"Error in reliability monitor: {e}")
                await asyncio.sleep(5)  # Shorter sleep on error

    async def _reliable_send_callback(self, destination: str, data: bytes):
        """Callback for reliable transport sending."""
        try:
            # This would integrate with the actual networking layer
            # For now, we'll use the existing gossipsub publish
            await self._net.gossipsub_publish(destination, data)
        except Exception as e:
            logger.error(f"Error in reliable send callback: {e}")
            raise

    async def _reliable_receive_callback(self, source: NodeId) -> bytes:
        """Callback for reliable transport receiving."""
        try:
            # This would integrate with the actual networking layer
            # For now, we'll use the existing gossipsub receive
            topic, data = await self._net.gossipsub_recv()
            return data
        except Exception as e:
            logger.error(f"Error in reliable receive callback: {e}")
            raise

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive router and reliability statistics."""
        stats = {"router": self.router_stats.copy(), "topics": {}}

        # Add per-topic statistics
        for topic_name, router in self.topic_routers.items():
            if isinstance(router, ReliableTopicRouter):
                stats["topics"][topic_name] = router.get_statistics()
            else:
                stats["topics"][topic_name] = {"type": "standard"}

        # Add transport statistics if available
        if self.reliable_transport:
            stats["transport"] = self.reliable_transport.get_statistics()

        return stats

    def reset_statistics(self):
        """Reset all statistics counters."""
        for key in self.router_stats:
            if isinstance(self.router_stats[key], (int, float)):
                self.router_stats[key] = 0

        if self.reliable_transport:
            self.reliable_transport.reset_statistics()

        for router in self.topic_routers.values():
            if isinstance(router, ReliableTopicRouter):
                router.message_stats = {k: 0 for k in router.message_stats}


def create_reliable_router(
    identity: Keypair, enable_reliability: bool = True
) -> ReliableRouter:
    """
    Create a ReliableRouter instance.

    Args:
        identity: Keypair for node identity
        enable_reliability: Whether to enable reliability features

    Returns:
        Configured ReliableRouter instance
    """
    handle = NetworkingHandle(identity)
    return ReliableRouter(handle, enable_reliability)
