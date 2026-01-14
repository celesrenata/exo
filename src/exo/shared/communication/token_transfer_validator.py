"""
Token transfer validation for ensuring data integrity during distributed communication.
"""

import hashlib
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from loguru import logger
from pydantic import Field

from exo.utils.pydantic_ext import CamelCaseModel
from exo.shared.types.validation import (
    EnhancedTokenChunk,
    ValidationResult,
    CorruptionType,
    CorruptionSeverity,
    ValidationError,
)


class TokenTransferMetadata(CamelCaseModel):
    """Metadata for token transfer validation."""

    transfer_id: str
    source_node: str
    destination_node: str
    token_count: int
    total_checksum: str
    transfer_timestamp: datetime
    compression_used: bool = False


class TokenBatch(CamelCaseModel):
    """Batch of tokens with transfer metadata."""

    tokens: List[EnhancedTokenChunk]
    metadata: TokenTransferMetadata
    batch_checksum: str = Field(default="")

    def compute_batch_checksum(self) -> str:
        """Compute checksum for the entire token batch."""
        # Combine all token checksums and metadata
        content_parts = []

        # Add token checksums in order
        for token in sorted(self.tokens, key=lambda t: t.sequence_position):
            content_parts.append(token.checksum or token.compute_checksum())

        # Add metadata
        content_parts.extend(
            [
                self.metadata.transfer_id,
                str(self.metadata.token_count),
                self.metadata.total_checksum,
            ]
        )

        combined_content = ":".join(content_parts)
        return hashlib.sha256(combined_content.encode("utf-8")).hexdigest()

    def verify_batch_checksum(self) -> bool:
        """Verify the batch checksum matches computed value."""
        if not self.batch_checksum:
            return False
        return self.batch_checksum == self.compute_batch_checksum()

    def update_batch_checksum(self) -> None:
        """Update the batch checksum based on current content."""
        self.batch_checksum = self.compute_batch_checksum()


class TokenTransferValidationResult(CamelCaseModel):
    """Result of token transfer validation."""

    is_valid: bool
    corrupted_tokens: List[int] = Field(default_factory=list)  # Token indices
    missing_tokens: List[int] = Field(default_factory=list)  # Expected positions
    checksum_mismatches: List[int] = Field(default_factory=list)
    encoding_errors: List[int] = Field(default_factory=list)
    batch_checksum_valid: bool = True
    total_tokens_expected: int = 0
    total_tokens_received: int = 0
    validation_details: str = ""


class TokenTransferValidator:
    """
    Validates token data integrity during transfers between distributed devices.

    Provides comprehensive validation including:
    - Individual token checksum verification
    - Batch integrity checking
    - Encoding validation
    - Sequence completeness verification
    - Corruption detection and classification
    """

    def __init__(self, enable_detailed_logging: bool = True):
        self.enable_detailed_logging = enable_detailed_logging
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "corruption_detected": 0,
            "checksum_failures": 0,
            "encoding_errors": 0,
        }

    def compute_token_checksum(self, token: EnhancedTokenChunk) -> str:
        """
        Compute checksum for token data.

        Args:
            token: Token to compute checksum for

        Returns:
            SHA-256 checksum string
        """
        return token.compute_checksum()

    def validate_token_transfer(
        self, original: EnhancedTokenChunk, received: EnhancedTokenChunk
    ) -> bool:
        """
        Validate that a received token matches the original.

        Args:
            original: Original token before transfer
            received: Token after transfer

        Returns:
            True if tokens match, False otherwise
        """
        try:
            # Basic field comparison
            if (
                original.text != received.text
                or original.token_id != received.token_id
                or original.idx != received.idx
            ):
                if self.enable_detailed_logging:
                    logger.warning(
                        f"Token content mismatch: original={original.text[:50]}..., "
                        f"received={received.text[:50]}..."
                    )
                return False

            # Checksum validation
            original_checksum = original.checksum or original.compute_checksum()
            received_checksum = received.checksum or received.compute_checksum()

            if original_checksum != received_checksum:
                if self.enable_detailed_logging:
                    logger.warning(
                        f"Token checksum mismatch: original={original_checksum}, "
                        f"received={received_checksum}"
                    )
                self.validation_stats["checksum_failures"] += 1
                return False

            # Encoding validation
            if not self._validate_token_encoding(received):
                self.validation_stats["encoding_errors"] += 1
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating token transfer: {e}")
            return False

    def validate_token_batch(self, batch: TokenBatch) -> TokenTransferValidationResult:
        """
        Validate an entire batch of transferred tokens.

        Args:
            batch: Token batch to validate

        Returns:
            Detailed validation result
        """
        self.validation_stats["total_validations"] += 1

        result = TokenTransferValidationResult(
            is_valid=True,
            total_tokens_expected=batch.metadata.token_count,
            total_tokens_received=len(batch.tokens),
        )

        try:
            # Verify batch checksum
            if not batch.verify_batch_checksum():
                result.batch_checksum_valid = False
                result.is_valid = False
                result.validation_details += "Batch checksum validation failed. "

            # Check token count
            if len(batch.tokens) != batch.metadata.token_count:
                missing_count = batch.metadata.token_count - len(batch.tokens)
                result.validation_details += f"Missing {missing_count} tokens. "
                result.is_valid = False

            # Validate individual tokens
            expected_positions = set(range(batch.metadata.token_count))
            received_positions = set()

            for i, token in enumerate(batch.tokens):
                # Check sequence position
                if hasattr(token, "sequence_position"):
                    received_positions.add(token.sequence_position)

                # Validate token checksum
                if not token.verify_checksum():
                    result.checksum_mismatches.append(i)
                    result.is_valid = False

                # Validate encoding
                if not self._validate_token_encoding(token):
                    result.encoding_errors.append(i)
                    result.is_valid = False

                # Check for corruption indicators
                if self._detect_token_corruption(token):
                    result.corrupted_tokens.append(i)
                    result.is_valid = False

            # Find missing positions
            if hasattr(batch.tokens[0], "sequence_position") if batch.tokens else False:
                missing_positions = expected_positions - received_positions
                result.missing_tokens = list(missing_positions)
                if missing_positions:
                    result.is_valid = False

            # Update statistics
            if result.is_valid:
                self.validation_stats["successful_validations"] += 1
            else:
                self.validation_stats["failed_validations"] += 1
                if result.corrupted_tokens:
                    self.validation_stats["corruption_detected"] += 1

            # Generate detailed validation message
            if not result.is_valid:
                details = []
                if result.corrupted_tokens:
                    details.append(f"Corrupted tokens: {len(result.corrupted_tokens)}")
                if result.checksum_mismatches:
                    details.append(
                        f"Checksum mismatches: {len(result.checksum_mismatches)}"
                    )
                if result.encoding_errors:
                    details.append(f"Encoding errors: {len(result.encoding_errors)}")
                if result.missing_tokens:
                    details.append(f"Missing tokens: {len(result.missing_tokens)}")

                result.validation_details += " ".join(details)

            if self.enable_detailed_logging:
                logger.info(
                    f"Token batch validation: {'PASSED' if result.is_valid else 'FAILED'} "
                    f"({result.total_tokens_received}/{result.total_tokens_expected} tokens)"
                )
                if not result.is_valid:
                    logger.warning(f"Validation issues: {result.validation_details}")

            return result

        except Exception as e:
            logger.error(f"Error validating token batch: {e}")
            result.is_valid = False
            result.validation_details = f"Validation error: {str(e)}"
            self.validation_stats["failed_validations"] += 1
            return result

    def detect_transfer_corruption(
        self, tokens: List[EnhancedTokenChunk]
    ) -> List[ValidationResult]:
        """
        Detect corruption in transferred tokens.

        Args:
            tokens: List of tokens to analyze

        Returns:
            List of validation results for each token
        """
        results = []

        for i, token in enumerate(tokens):
            try:
                # Check encoding
                encoding_valid = self._validate_token_encoding(token)

                # Check for corruption patterns
                corruption_detected = self._detect_token_corruption(token)

                # Check checksum
                checksum_valid = token.verify_checksum()

                if encoding_valid and not corruption_detected and checksum_valid:
                    results.append(
                        ValidationResult(is_valid=True, confidence_score=1.0)
                    )
                else:
                    # Determine primary error type
                    if not encoding_valid:
                        error_type = CorruptionType.ENCODING_CORRUPTION
                        severity = CorruptionSeverity.HIGH
                        message = "Invalid UTF-8 encoding detected"
                    elif not checksum_valid:
                        error_type = CorruptionType.CHECKSUM_MISMATCH
                        severity = CorruptionSeverity.MEDIUM
                        message = "Checksum validation failed"
                    else:
                        error_type = CorruptionType.SEMANTIC_CORRUPTION
                        severity = CorruptionSeverity.MEDIUM
                        message = "Semantic corruption patterns detected"

                    error = ValidationError(
                        error_type=error_type, message=message, severity=severity
                    )

                    results.append(
                        ValidationResult(
                            is_valid=False,
                            error_type=error_type,
                            confidence_score=0.2,
                            error_details=error,
                        )
                    )

            except Exception as e:
                logger.error(f"Error analyzing token {i}: {e}")
                error = ValidationError(
                    error_type=CorruptionType.SEMANTIC_CORRUPTION,
                    message=f"Analysis error: {str(e)}",
                    severity=CorruptionSeverity.HIGH,
                )
                results.append(
                    ValidationResult(
                        is_valid=False,
                        error_type=CorruptionType.SEMANTIC_CORRUPTION,
                        confidence_score=0.0,
                        error_details=error,
                    )
                )

        return results

    def _validate_token_encoding(self, token: EnhancedTokenChunk) -> bool:
        """Validate that token text has proper UTF-8 encoding."""
        try:
            # Try to encode and decode the text
            encoded = token.text.encode("utf-8")
            decoded = encoded.decode("utf-8")
            return decoded == token.text
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False

    def _detect_token_corruption(self, token: EnhancedTokenChunk) -> bool:
        """
        Detect corruption patterns in token text.

        Returns True if corruption is detected.
        """
        text = token.text

        # Check for common corruption patterns
        corruption_indicators = [
            # Excessive special characters
            sum(1 for c in text if not c.isalnum() and not c.isspace())
            / max(len(text), 1)
            > 0.5,
            # Null bytes or control characters
            any(ord(c) < 32 and c not in "\n\r\t" for c in text),
            # Replacement characters (indicates encoding issues)
            "\ufffd" in text,
            # Excessive repetition of single character
            any(text.count(c) > len(text) * 0.8 for c in set(text) if len(text) > 10),
            # Empty or whitespace-only tokens where content expected
            len(text.strip()) == 0
            and hasattr(token, "token_id")
            and token.token_id > 0,
        ]

        return any(corruption_indicators)

    def create_token_batch(
        self,
        tokens: List[EnhancedTokenChunk],
        transfer_id: str,
        source_node: str,
        destination_node: str,
    ) -> TokenBatch:
        """
        Create a validated token batch for transfer.

        Args:
            tokens: List of tokens to batch
            transfer_id: Unique transfer identifier
            source_node: Source node identifier
            destination_node: Destination node identifier

        Returns:
            TokenBatch with computed checksums and metadata
        """
        # Ensure all tokens have checksums
        for token in tokens:
            if not token.checksum:
                token.update_checksum()

        # Compute total checksum
        all_checksums = [token.checksum for token in tokens]
        total_checksum = hashlib.sha256(
            ":".join(all_checksums).encode("utf-8")
        ).hexdigest()

        metadata = TokenTransferMetadata(
            transfer_id=transfer_id,
            source_node=source_node,
            destination_node=destination_node,
            token_count=len(tokens),
            total_checksum=total_checksum,
            transfer_timestamp=datetime.now(),
        )

        batch = TokenBatch(tokens=tokens, metadata=metadata)

        batch.update_batch_checksum()
        return batch

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()

    def reset_statistics(self):
        """Reset validation statistics."""
        for key in self.validation_stats:
            self.validation_stats[key] = 0
