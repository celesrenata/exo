"""
Enhanced validation types for distributed inference stability.
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import Field

from exo.utils.pydantic_ext import CamelCaseModel
from .chunks import TokenChunk


class ValidationStatus(str, Enum):
    """Status of token validation."""

    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    CORRUPTED = "corrupted"
    ENCODING_ERROR = "encoding_error"


class CorruptionType(str, Enum):
    """Types of corruption that can be detected."""

    ENCODING_CORRUPTION = "encoding_corruption"
    SEMANTIC_CORRUPTION = "semantic_corruption"
    SEQUENCE_CORRUPTION = "sequence_corruption"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    NONE = "none"


class CorruptionSeverity(str, Enum):
    """Severity levels for corruption."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureMode(str, Enum):
    """Types of failures that can occur in distributed inference."""

    TOKEN_CORRUPTION = "token_corruption"
    ENCODING_ERROR = "encoding_error"
    SYNCHRONIZATION_FAILURE = "synchronization_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    DEVICE_FAILURE = "device_failure"
    PIPELINE_FAILURE = "pipeline_failure"
    MEMORY_CORRUPTION = "memory_corruption"
    TIMEOUT = "timeout"


class RecoveryAction(str, Enum):
    """Actions that can be taken to recover from failures."""

    RETRY = "retry"
    REGENERATE_TOKEN = "regenerate_token"
    RESTART_PIPELINE_STAGE = "restart_pipeline_stage"
    REINITIALIZE_DEVICE = "reinitialize_device"
    FALLBACK_SINGLE_DEVICE = "fallback_single_device"
    SKIP_CORRUPTED_TOKEN = "skip_corrupted_token"
    RESET_SYNCHRONIZATION = "reset_synchronization"
    NO_ACTION = "no_action"


class RecoveryStatus(str, Enum):
    """Status of recovery operations."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    NOT_ATTEMPTED = "not_attempted"


class ValidationError(CamelCaseModel):
    """Details about validation errors."""

    error_type: CorruptionType
    message: str
    severity: CorruptionSeverity
    recoverable: bool = True


class ValidationResult(CamelCaseModel):
    """Result of token validation."""

    is_valid: bool
    error_type: Optional[CorruptionType] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    error_details: Optional[ValidationError] = None
    suggested_action: RecoveryAction = RecoveryAction.NO_ACTION


class RecoveryResult(CamelCaseModel):
    """Result of a recovery operation."""

    status: RecoveryStatus
    action_taken: RecoveryAction
    success: bool
    error_message: Optional[str] = None
    recovery_time: float = 0.0  # Time taken in seconds
    attempts: int = 1
    affected_devices: List[str] = Field(default_factory=list)
    details: str = ""


class DeviceHealth(CamelCaseModel):
    """Health status of a device in the distributed system."""

    device_id: str
    is_healthy: bool
    last_heartbeat: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    recovery_attempts: int = 0
    status: str = "active"  # active, degraded, failed, recovering


class EnhancedTokenChunk(TokenChunk):
    """Enhanced TokenChunk with validation metadata."""

    # Validation metadata
    checksum: str = Field(default="")
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    source_device_rank: int = Field(default=0)
    validation_status: ValidationStatus = Field(default=ValidationStatus.PENDING)
    sequence_position: int = Field(default=0)

    # Optional validation results
    validation_result: Optional[ValidationResult] = None

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of the token text."""
        content = f"{self.text}:{self.token_id}:{self.idx}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def verify_checksum(self) -> bool:
        """Verify that the stored checksum matches the computed checksum."""
        if not self.checksum:
            return False
        return self.checksum == self.compute_checksum()

    def update_checksum(self) -> None:
        """Update the checksum based on current token content."""
        self.checksum = self.compute_checksum()

    def mark_as_valid(self) -> None:
        """Mark the token as valid."""
        self.validation_status = ValidationStatus.VALID
        self.validation_result = ValidationResult(is_valid=True, confidence_score=1.0)

    def mark_as_corrupted(
        self,
        corruption_type: CorruptionType,
        error_message: str,
        severity: CorruptionSeverity,
    ) -> None:
        """Mark the token as corrupted with details."""
        self.validation_status = ValidationStatus.CORRUPTED
        error = ValidationError(
            error_type=corruption_type, message=error_message, severity=severity
        )
        self.validation_result = ValidationResult(
            is_valid=False,
            error_type=corruption_type,
            confidence_score=0.0,
            error_details=error,
        )


class SequenceValidationResult(CamelCaseModel):
    """Result of sequence validation."""

    is_valid: bool
    missing_positions: list[int] = Field(default_factory=list)
    duplicate_positions: list[int] = Field(default_factory=list)
    out_of_order_positions: list[int] = Field(default_factory=list)
    total_tokens: int = 0
    expected_tokens: int = 0


class CorruptionReport(CamelCaseModel):
    """Detailed report of detected corruption."""

    corruption_type: CorruptionType
    affected_range: tuple[int, int]  # Start and end positions
    severity: CorruptionSeverity
    recovery_possible: bool
    details: str
    detected_at: datetime = Field(default_factory=datetime.now)
