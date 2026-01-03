"""
Reliable message transport layer for distributed inference communication.
"""

import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, Callable, Awaitable
from uuid import uuid4

from loguru import logger
from pydantic import Field

from exo.utils.pydantic_ext import CamelCaseModel
from exo.shared.types.common import NodeId


class MessageStatus(str, Enum):
    """Status of message transmission."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TransferResult(CamelCaseModel):
    """Result of message transfer operation."""
    success: bool
    message_id: str
    status: MessageStatus
    attempts: int
    total_time: float
    error_message: Optional[str] = None
    checksum_verified: bool = False


class ValidatedMessage(CamelCaseModel):
    """Message with validation metadata."""
    message_id: str
    source: NodeId
    destination: NodeId
    data: bytes
    checksum: str
    timestamp: datetime
    retry_count: int = 0
    
    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of the message data."""
        return hashlib.sha256(self.data).hexdigest()
    
    def verify_checksum(self) -> bool:
        """Verify that the stored checksum matches the computed checksum."""
        return self.checksum == self.compute_checksum()


class ReliableMessageTransport:
    """
    Reliable message transport with checksums, retry logic, and timeout handling.
    
    Provides guaranteed delivery semantics for inter-device communication
    with configurable retry policies and integrity validation.
    """
    
    def __init__(
        self,
        node_id: NodeId,
        max_retries: int = 3,
        base_timeout: float = 5.0,
        max_timeout: float = 30.0,
        backoff_multiplier: float = 2.0,
        checksum_validation: bool = True
    ):
        self.node_id = node_id
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.max_timeout = max_timeout
        self.backoff_multiplier = backoff_multiplier
        self.checksum_validation = checksum_validation
        
        # Message tracking
        self.pending_messages: Dict[str, ValidatedMessage] = {}
        self.acknowledgments: Dict[str, bool] = {}
        
        # Statistics
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.total_retries = 0
        self.total_failures = 0
        
        # Transport layer callback (to be set by router)
        self.transport_send_callback: Optional[Callable[[str, bytes], Awaitable[None]]] = None
        self.transport_receive_callback: Optional[Callable[[NodeId], Awaitable[bytes]]] = None
    
    def set_transport_callbacks(
        self,
        send_callback: Callable[[str, bytes], Awaitable[None]],
        receive_callback: Callable[[NodeId], Awaitable[bytes]]
    ):
        """Set the underlying transport layer callbacks."""
        self.transport_send_callback = send_callback
        self.transport_receive_callback = receive_callback
    
    async def send_with_validation(
        self, 
        data: bytes, 
        destination: NodeId,
        timeout: Optional[float] = None
    ) -> TransferResult:
        """
        Send data with validation, retry logic, and acknowledgment.
        
        Args:
            data: Raw bytes to send
            destination: Target node ID
            timeout: Optional timeout override
            
        Returns:
            TransferResult with success status and metadata
        """
        if not self.transport_send_callback:
            raise RuntimeError("Transport send callback not configured")
        
        message_id = str(uuid4())
        message = ValidatedMessage(
            message_id=message_id,
            source=self.node_id,
            destination=destination,
            data=data,
            checksum=hashlib.sha256(data).hexdigest(),
            timestamp=datetime.now()
        )
        
        self.pending_messages[message_id] = message
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                # Calculate timeout for this attempt
                attempt_timeout = min(
                    self.base_timeout * (self.backoff_multiplier ** attempt),
                    self.max_timeout
                )
                if timeout:
                    attempt_timeout = min(attempt_timeout, timeout)
                
                logger.debug(
                    f"Sending message {message_id} to {destination}, "
                    f"attempt {attempt + 1}/{self.max_retries + 1}, "
                    f"timeout: {attempt_timeout}s"
                )
                
                # Send the message
                serialized_message = self._serialize_message(message)
                await asyncio.wait_for(
                    self.transport_send_callback(str(destination), serialized_message),
                    timeout=attempt_timeout
                )
                
                # Wait for acknowledgment
                ack_received = await self._wait_for_acknowledgment(
                    message_id, attempt_timeout
                )
                
                if ack_received:
                    total_time = time.time() - start_time
                    self.total_messages_sent += 1
                    
                    # Clean up
                    self.pending_messages.pop(message_id, None)
                    self.acknowledgments.pop(message_id, None)
                    
                    return TransferResult(
                        success=True,
                        message_id=message_id,
                        status=MessageStatus.ACKNOWLEDGED,
                        attempts=attempt + 1,
                        total_time=total_time,
                        checksum_verified=True
                    )
                
                # No acknowledgment received, will retry
                message.retry_count = attempt + 1
                self.total_retries += 1
                
                if attempt < self.max_retries:
                    # Exponential backoff before retry
                    backoff_time = min(
                        0.1 * (self.backoff_multiplier ** attempt),
                        2.0
                    )
                    await asyncio.sleep(backoff_time)
                
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout sending message {message_id} to {destination}, "
                    f"attempt {attempt + 1}"
                )
                continue
            except Exception as e:
                logger.error(
                    f"Error sending message {message_id} to {destination}: {e}"
                )
                continue
        
        # All retries exhausted
        total_time = time.time() - start_time
        self.total_failures += 1
        
        # Clean up
        self.pending_messages.pop(message_id, None)
        self.acknowledgments.pop(message_id, None)
        
        return TransferResult(
            success=False,
            message_id=message_id,
            status=MessageStatus.FAILED,
            attempts=self.max_retries + 1,
            total_time=total_time,
            error_message="Max retries exceeded"
        )
    
    async def receive_with_validation(self, source: NodeId) -> ValidatedMessage:
        """
        Receive and validate a message from a source node.
        
        Args:
            source: Source node ID
            
        Returns:
            ValidatedMessage with integrity verification
            
        Raises:
            ValueError: If message validation fails
            RuntimeError: If transport not configured
        """
        if not self.transport_receive_callback:
            raise RuntimeError("Transport receive callback not configured")
        
        try:
            # Receive raw data
            raw_data = await self.transport_receive_callback(source)
            
            # Deserialize message
            message = self._deserialize_message(raw_data)
            
            # Validate checksum if enabled
            if self.checksum_validation and not message.verify_checksum():
                raise ValueError(
                    f"Checksum validation failed for message {message.message_id} "
                    f"from {source}"
                )
            
            # Send acknowledgment
            await self._send_acknowledgment(message.message_id, source)
            
            self.total_messages_received += 1
            
            logger.debug(
                f"Successfully received and validated message {message.message_id} "
                f"from {source}"
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to receive message from {source}: {e}")
            raise
    
    async def _wait_for_acknowledgment(
        self, 
        message_id: str, 
        timeout: float
    ) -> bool:
        """Wait for acknowledgment of a sent message."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if message_id in self.acknowledgments:
                return self.acknowledgments[message_id]
            await asyncio.sleep(0.1)  # Check every 100ms
        
        return False
    
    async def _send_acknowledgment(self, message_id: str, destination: NodeId):
        """Send acknowledgment for a received message."""
        if not self.transport_send_callback:
            return
        
        ack_data = {
            "type": "acknowledgment",
            "message_id": message_id,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            import json
            ack_bytes = json.dumps(ack_data).encode('utf-8')
            await self.transport_send_callback(str(destination), ack_bytes)
            logger.debug(f"Sent acknowledgment for message {message_id} to {destination}")
        except Exception as e:
            logger.error(f"Failed to send acknowledgment for {message_id}: {e}")
    
    def handle_acknowledgment(self, message_id: str):
        """Handle received acknowledgment."""
        self.acknowledgments[message_id] = True
        logger.debug(f"Received acknowledgment for message {message_id}")
    
    def _serialize_message(self, message: ValidatedMessage) -> bytes:
        """Serialize a ValidatedMessage to bytes."""
        import json
        
        data = {
            "message_id": message.message_id,
            "source": str(message.source),
            "destination": str(message.destination),
            "data": message.data.hex(),  # Convert bytes to hex string
            "checksum": message.checksum,
            "timestamp": message.timestamp.isoformat(),
            "retry_count": message.retry_count
        }
        
        return json.dumps(data).encode('utf-8')
    
    def _deserialize_message(self, raw_data: bytes) -> ValidatedMessage:
        """Deserialize bytes to a ValidatedMessage."""
        import json
        
        try:
            data = json.loads(raw_data.decode('utf-8'))
            
            return ValidatedMessage(
                message_id=data["message_id"],
                source=NodeId(data["source"]),
                destination=NodeId(data["destination"]),
                data=bytes.fromhex(data["data"]),  # Convert hex string back to bytes
                checksum=data["checksum"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                retry_count=data.get("retry_count", 0)
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to deserialize message: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transport statistics."""
        return {
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "total_retries": self.total_retries,
            "total_failures": self.total_failures,
            "pending_messages": len(self.pending_messages),
            "success_rate": (
                self.total_messages_sent / (self.total_messages_sent + self.total_failures)
                if (self.total_messages_sent + self.total_failures) > 0 else 0.0
            )
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.total_retries = 0
        self.total_failures = 0