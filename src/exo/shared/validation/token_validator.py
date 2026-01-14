"""
Token validation system for detecting corruption in distributed inference.
"""

import re
import unicodedata
from typing import List

from ..types.validation import (
    CorruptionReport,
    CorruptionSeverity,
    CorruptionType,
    EnhancedTokenChunk,
    ValidationError,
    ValidationResult,
)


class TokenValidator:
    """Validates tokens for corruption and encoding issues."""

    def __init__(self):
        # Patterns for detecting common corruption types
        self.garbled_patterns = [
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]",  # Control characters
            r"[^\x00-\x7F]{3,}",  # Long sequences of non-ASCII
            r"�+",  # Unicode replacement characters
            r"[\uFFFD\uFFFE\uFFFF]",  # Invalid Unicode code points
        ]

        # Common encoding corruption patterns
        self.encoding_patterns = [
            r"Ã[€-¿]",  # UTF-8 decoded as Latin-1
            r"â€[™œž]",  # Common UTF-8 corruption
            r"Â[€-¿]",  # Another UTF-8/Latin-1 issue
        ]

        # Semantic corruption indicators
        self.semantic_patterns = [
            r"^[^a-zA-Z0-9\s]{5,}$",  # Only special characters
            r"(.)\1{10,}",  # Excessive repetition
            r"[a-zA-Z]{50,}",  # Unreasonably long words
        ]

    def validate_token(self, token: EnhancedTokenChunk) -> ValidationResult:
        """
        Validate a single token for various types of corruption.

        Args:
            token: The token to validate

        Returns:
            ValidationResult with validation status and details
        """
        # Check checksum first
        if token.checksum and not token.verify_checksum():
            return ValidationResult(
                is_valid=False,
                error_type=CorruptionType.CHECKSUM_MISMATCH,
                confidence_score=0.0,
                error_details=ValidationError(
                    error_type=CorruptionType.CHECKSUM_MISMATCH,
                    message="Token checksum verification failed",
                    severity=CorruptionSeverity.HIGH,
                ),
            )

        # Validate encoding
        encoding_result = self._validate_encoding(token.text)
        if not encoding_result.is_valid:
            return encoding_result

        # Validate semantic content
        semantic_result = self._validate_semantic_content(token.text)
        if not semantic_result.is_valid:
            return semantic_result

        # If all checks pass
        return ValidationResult(is_valid=True, confidence_score=1.0)

    def validate_sequence(self, tokens: List[EnhancedTokenChunk]) -> ValidationResult:
        """
        Validate a sequence of tokens for ordering and completeness.

        Args:
            tokens: List of tokens to validate

        Returns:
            ValidationResult for the entire sequence
        """
        if not tokens:
            return ValidationResult(is_valid=True, confidence_score=1.0)

        # Check for sequence position ordering
        positions = [token.sequence_position for token in tokens]
        expected_positions = list(range(min(positions), max(positions) + 1))

        if positions != expected_positions:
            return ValidationResult(
                is_valid=False,
                error_type=CorruptionType.SEQUENCE_CORRUPTION,
                confidence_score=0.0,
                error_details=ValidationError(
                    error_type=CorruptionType.SEQUENCE_CORRUPTION,
                    message=f"Token sequence out of order. Expected: {expected_positions}, Got: {positions}",
                    severity=CorruptionSeverity.MEDIUM,
                ),
            )

        # Validate individual tokens
        for token in tokens:
            token_result = self.validate_token(token)
            if not token_result.is_valid:
                return token_result

        return ValidationResult(is_valid=True, confidence_score=1.0)

    def detect_corruption(self, text: str) -> CorruptionReport:
        """
        Analyze text for corruption and generate a detailed report.

        Args:
            text: The text to analyze

        Returns:
            CorruptionReport with detailed analysis
        """
        # Check for encoding corruption
        for pattern in self.encoding_patterns:
            if re.search(pattern, text):
                return CorruptionReport(
                    corruption_type=CorruptionType.ENCODING_CORRUPTION,
                    affected_range=(0, len(text)),
                    severity=CorruptionSeverity.HIGH,
                    recovery_possible=True,
                    details=f"Encoding corruption detected with pattern: {pattern}",
                )

        # Check for garbled text
        for pattern in self.garbled_patterns:
            match = re.search(pattern, text)
            if match:
                return CorruptionReport(
                    corruption_type=CorruptionType.ENCODING_CORRUPTION,
                    affected_range=(match.start(), match.end()),
                    severity=CorruptionSeverity.CRITICAL,
                    recovery_possible=False,
                    details=f"Garbled text detected: {match.group()}",
                )

        # Check for semantic corruption
        for pattern in self.semantic_patterns:
            match = re.search(pattern, text)
            if match:
                return CorruptionReport(
                    corruption_type=CorruptionType.SEMANTIC_CORRUPTION,
                    affected_range=(match.start(), match.end()),
                    severity=CorruptionSeverity.MEDIUM,
                    recovery_possible=True,
                    details=f"Semantic corruption detected: {match.group()}",
                )

        # No corruption detected
        return CorruptionReport(
            corruption_type=CorruptionType.NONE,
            affected_range=(0, 0),
            severity=CorruptionSeverity.LOW,
            recovery_possible=True,
            details="No corruption detected",
        )

    def _validate_encoding(self, text: str) -> ValidationResult:
        """Validate text encoding for UTF-8 corruption."""
        try:
            # Check if text can be properly encoded/decoded
            text.encode("utf-8").decode("utf-8")

            # Check for invalid Unicode characters
            for char in text:
                if unicodedata.category(char) == "Cn":  # Unassigned category
                    return ValidationResult(
                        is_valid=False,
                        error_type=CorruptionType.ENCODING_CORRUPTION,
                        confidence_score=0.0,
                        error_details=ValidationError(
                            error_type=CorruptionType.ENCODING_CORRUPTION,
                            message=f"Invalid Unicode character detected: {repr(char)}",
                            severity=CorruptionSeverity.HIGH,
                        ),
                    )

            # Check for encoding corruption patterns
            for pattern in self.encoding_patterns:
                if re.search(pattern, text):
                    return ValidationResult(
                        is_valid=False,
                        error_type=CorruptionType.ENCODING_CORRUPTION,
                        confidence_score=0.0,
                        error_details=ValidationError(
                            error_type=CorruptionType.ENCODING_CORRUPTION,
                            message=f"Encoding corruption pattern detected: {pattern}",
                            severity=CorruptionSeverity.HIGH,
                        ),
                    )

            # Check for garbled patterns
            for pattern in self.garbled_patterns:
                if re.search(pattern, text):
                    return ValidationResult(
                        is_valid=False,
                        error_type=CorruptionType.ENCODING_CORRUPTION,
                        confidence_score=0.0,
                        error_details=ValidationError(
                            error_type=CorruptionType.ENCODING_CORRUPTION,
                            message=f"Garbled text pattern detected: {pattern}",
                            severity=CorruptionSeverity.CRITICAL,
                        ),
                    )

            return ValidationResult(is_valid=True, confidence_score=1.0)

        except UnicodeError as e:
            return ValidationResult(
                is_valid=False,
                error_type=CorruptionType.ENCODING_CORRUPTION,
                confidence_score=0.0,
                error_details=ValidationError(
                    error_type=CorruptionType.ENCODING_CORRUPTION,
                    message=f"Unicode encoding error: {str(e)}",
                    severity=CorruptionSeverity.CRITICAL,
                ),
            )

    def _validate_semantic_content(self, text: str) -> ValidationResult:
        """Validate text for semantic corruption using heuristics."""
        # Check for excessive repetition
        for pattern in self.semantic_patterns:
            match = re.search(pattern, text)
            if match:
                return ValidationResult(
                    is_valid=False,
                    error_type=CorruptionType.SEMANTIC_CORRUPTION,
                    confidence_score=0.0,
                    error_details=ValidationError(
                        error_type=CorruptionType.SEMANTIC_CORRUPTION,
                        message=f"Semantic corruption detected: {match.group()}",
                        severity=CorruptionSeverity.MEDIUM,
                    ),
                )

        # Check character distribution (basic heuristic)
        if len(text) > 10:
            unique_chars = len(set(text))
            total_chars = len(text)
            diversity_ratio = unique_chars / total_chars

            # If diversity is too low, might be corrupted
            if diversity_ratio < 0.1:
                return ValidationResult(
                    is_valid=False,
                    error_type=CorruptionType.SEMANTIC_CORRUPTION,
                    confidence_score=0.0,
                    error_details=ValidationError(
                        error_type=CorruptionType.SEMANTIC_CORRUPTION,
                        message=f"Low character diversity: {diversity_ratio:.2f}",
                        severity=CorruptionSeverity.LOW,
                    ),
                )

        return ValidationResult(is_valid=True, confidence_score=1.0)
