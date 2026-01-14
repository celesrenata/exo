"""
Sequence integrity checker for validating token ordering in distributed streams.
"""

from typing import Dict, List, Optional, Tuple

from ..types.validation import (
    CorruptionSeverity,
    CorruptionType,
    EnhancedTokenChunk,
    SequenceValidationResult,
    ValidationError,
    ValidationResult,
)


class SequenceIntegrityChecker:
    """Validates token sequence integrity for distributed token streams."""

    def __init__(self):
        self.expected_sequences: Dict[str, List[int]] = {}
        self.received_sequences: Dict[str, List[EnhancedTokenChunk]] = {}

    def validate_sequence(
        self, tokens: List[EnhancedTokenChunk], sequence_id: Optional[str] = None
    ) -> SequenceValidationResult:
        """
        Validate token sequence for ordering and completeness.

        Args:
            tokens: List of tokens to validate
            sequence_id: Optional identifier for the sequence

        Returns:
            SequenceValidationResult with detailed validation information
        """
        if not tokens:
            return SequenceValidationResult(
                is_valid=True, total_tokens=0, expected_tokens=0
            )

        # Extract sequence positions
        positions = [token.sequence_position for token in tokens]
        positions.sort()

        # Determine expected range
        min_pos = min(positions)
        max_pos = max(positions)
        expected_positions = set(range(min_pos, max_pos + 1))
        received_positions = set(positions)

        # Find gaps and duplicates
        missing_positions = list(expected_positions - received_positions)
        duplicate_positions = self._find_duplicates(positions)
        out_of_order_positions = self._find_out_of_order(tokens)

        is_valid = (
            len(missing_positions) == 0
            and len(duplicate_positions) == 0
            and len(out_of_order_positions) == 0
        )

        return SequenceValidationResult(
            is_valid=is_valid,
            missing_positions=missing_positions,
            duplicate_positions=duplicate_positions,
            out_of_order_positions=out_of_order_positions,
            total_tokens=len(tokens),
            expected_tokens=len(expected_positions),
        )

    def detect_gaps(self, tokens: List[EnhancedTokenChunk]) -> List[Tuple[int, int]]:
        """
        Detect gaps in token sequences.

        Args:
            tokens: List of tokens to check

        Returns:
            List of (start, end) tuples representing gaps
        """
        if not tokens:
            return []

        positions = sorted([token.sequence_position for token in tokens])
        gaps: List[Tuple[int, int]] = []

        for i in range(len(positions) - 1):
            current_pos = positions[i]
            next_pos = positions[i + 1]

            if next_pos - current_pos > 1:
                gaps.append((current_pos + 1, next_pos - 1))

        return gaps

    def verify_ordering(self, tokens: List[EnhancedTokenChunk]) -> ValidationResult:
        """
        Verify that tokens are in correct order based on sequence positions.

        Args:
            tokens: List of tokens to verify

        Returns:
            ValidationResult indicating if ordering is correct
        """
        if len(tokens) <= 1:
            return ValidationResult(is_valid=True, confidence_score=1.0)

        # Check if sequence positions are in ascending order
        positions = [token.sequence_position for token in tokens]
        sorted_positions = sorted(positions)

        if positions != sorted_positions:
            out_of_order = []
            for i, (actual, expected) in enumerate(zip(positions, sorted_positions)):
                if actual != expected:
                    out_of_order.append(i)

            return ValidationResult(
                is_valid=False,
                error_type=CorruptionType.SEQUENCE_CORRUPTION,
                confidence_score=0.0,
                error_details=ValidationError(
                    error_type=CorruptionType.SEQUENCE_CORRUPTION,
                    message=f"Tokens out of order at positions: {out_of_order}",
                    severity=CorruptionSeverity.MEDIUM,
                ),
            )

        # Check for timestamp consistency (tokens should be generated in order)
        timestamps = [token.generation_timestamp for token in tokens]
        for i in range(len(timestamps) - 1):
            if timestamps[i] > timestamps[i + 1]:
                return ValidationResult(
                    is_valid=False,
                    error_type=CorruptionType.SEQUENCE_CORRUPTION,
                    confidence_score=0.0,
                    error_details=ValidationError(
                        error_type=CorruptionType.SEQUENCE_CORRUPTION,
                        message=f"Timestamp ordering violation at position {i}",
                        severity=CorruptionSeverity.LOW,
                    ),
                )

        return ValidationResult(is_valid=True, confidence_score=1.0)

    def validate_distributed_stream(
        self, token_streams: Dict[int, List[EnhancedTokenChunk]]
    ) -> ValidationResult:
        """
        Validate token streams from multiple devices for consistency.

        Args:
            token_streams: Dictionary mapping device rank to token list

        Returns:
            ValidationResult for the distributed stream
        """
        if not token_streams:
            return ValidationResult(is_valid=True, confidence_score=1.0)

        # Collect all tokens and sort by sequence position
        all_tokens: List[EnhancedTokenChunk] = []
        for device_rank, tokens in token_streams.items():
            for token in tokens:
                # Verify device rank consistency
                if token.source_device_rank != device_rank:
                    return ValidationResult(
                        is_valid=False,
                        error_type=CorruptionType.SEQUENCE_CORRUPTION,
                        confidence_score=0.0,
                        error_details=ValidationError(
                            error_type=CorruptionType.SEQUENCE_CORRUPTION,
                            message=f"Device rank mismatch: token claims rank {token.source_device_rank} but in stream {device_rank}",
                            severity=CorruptionSeverity.HIGH,
                        ),
                    )
                all_tokens.append(token)

        # Sort by sequence position
        all_tokens.sort(key=lambda t: t.sequence_position)

        # Validate the combined sequence
        sequence_result = self.validate_sequence(all_tokens)
        if not sequence_result.is_valid:
            return ValidationResult(
                is_valid=False,
                error_type=CorruptionType.SEQUENCE_CORRUPTION,
                confidence_score=0.0,
                error_details=ValidationError(
                    error_type=CorruptionType.SEQUENCE_CORRUPTION,
                    message=f"Distributed stream validation failed: {sequence_result}",
                    severity=CorruptionSeverity.HIGH,
                ),
            )

        # Validate ordering within each device stream
        for device_rank, tokens in token_streams.items():
            ordering_result = self.verify_ordering(tokens)
            if not ordering_result.is_valid:
                error_msg = "Unknown error"
                if ordering_result.error_details is not None:
                    error_msg = ordering_result.error_details.message

                return ValidationResult(
                    is_valid=False,
                    error_type=CorruptionType.SEQUENCE_CORRUPTION,
                    confidence_score=0.0,
                    error_details=ValidationError(
                        error_type=CorruptionType.SEQUENCE_CORRUPTION,
                        message=f"Device {device_rank} stream ordering invalid: {error_msg}",
                        severity=CorruptionSeverity.MEDIUM,
                    ),
                )

        return ValidationResult(is_valid=True, confidence_score=1.0)

    def register_expected_sequence(
        self, sequence_id: str, expected_positions: List[int]
    ) -> None:
        """
        Register expected sequence positions for validation.

        Args:
            sequence_id: Identifier for the sequence
            expected_positions: List of expected sequence positions
        """
        self.expected_sequences[sequence_id] = sorted(expected_positions)

    def add_received_token(self, sequence_id: str, token: EnhancedTokenChunk) -> None:
        """
        Add a received token to a tracked sequence.

        Args:
            sequence_id: Identifier for the sequence
            token: The received token
        """
        if sequence_id not in self.received_sequences:
            self.received_sequences[sequence_id] = []
        self.received_sequences[sequence_id].append(token)

    def validate_tracked_sequence(self, sequence_id: str) -> ValidationResult:
        """
        Validate a tracked sequence against expected positions.

        Args:
            sequence_id: Identifier for the sequence to validate

        Returns:
            ValidationResult for the tracked sequence
        """
        if sequence_id not in self.expected_sequences:
            return ValidationResult(
                is_valid=False,
                error_type=CorruptionType.SEQUENCE_CORRUPTION,
                confidence_score=0.0,
                error_details=ValidationError(
                    error_type=CorruptionType.SEQUENCE_CORRUPTION,
                    message=f"No expected sequence registered for {sequence_id}",
                    severity=CorruptionSeverity.HIGH,
                ),
            )

        received_tokens = self.received_sequences.get(sequence_id, [])
        expected_positions = set(self.expected_sequences[sequence_id])
        received_positions = set(token.sequence_position for token in received_tokens)

        missing_positions = list(expected_positions - received_positions)
        extra_positions = list(received_positions - expected_positions)

        if missing_positions or extra_positions:
            return ValidationResult(
                is_valid=False,
                error_type=CorruptionType.SEQUENCE_CORRUPTION,
                confidence_score=0.0,
                error_details=ValidationError(
                    error_type=CorruptionType.SEQUENCE_CORRUPTION,
                    message=f"Sequence mismatch - Missing: {missing_positions}, Extra: {extra_positions}",
                    severity=CorruptionSeverity.MEDIUM,
                ),
            )

        return ValidationResult(is_valid=True, confidence_score=1.0)

    def _find_duplicates(self, positions: List[int]) -> List[int]:
        """Find duplicate positions in the list."""
        seen: set[int] = set()
        duplicates: set[int] = set()

        for pos in positions:
            if pos in seen:
                duplicates.add(pos)
            else:
                seen.add(pos)

        return list(duplicates)

    def _find_out_of_order(self, tokens: List[EnhancedTokenChunk]) -> List[int]:
        """Find tokens that are out of order based on their original positions."""
        out_of_order: List[int] = []

        for i in range(len(tokens) - 1):
            current_pos = tokens[i].sequence_position
            next_pos = tokens[i + 1].sequence_position

            if current_pos > next_pos:
                out_of_order.extend([i, i + 1])

        return list(set(out_of_order))  # Remove duplicates
