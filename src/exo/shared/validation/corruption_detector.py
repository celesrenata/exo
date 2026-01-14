"""
CorruptionDetector for identifying output issues in distributed inference.
"""

import re
import unicodedata
from typing import List, Optional, Tuple
from datetime import datetime

from ..types.validation import (
    CorruptionType,
    CorruptionSeverity,
    CorruptionReport,
    EnhancedTokenChunk,
    ValidationResult,
    ValidationError,
)


class CorruptionDetector:
    """
    Detects various types of corruption in model output including
    encoding issues, garbled text, and semantic inconsistencies.
    """

    def __init__(self):
        # Common patterns that indicate corruption
        self.corruption_patterns = [
            # Repeated characters (e.g., "aaaaaaa", "!!!!!!")
            (
                re.compile(r"(.)\1{5,}"),
                CorruptionType.SEMANTIC_CORRUPTION,
                CorruptionSeverity.MEDIUM,
            ),
            # Random character sequences (high entropy)
            (
                re.compile(r"[^\w\s]{4,}"),
                CorruptionType.SEMANTIC_CORRUPTION,
                CorruptionSeverity.LOW,
            ),
            # Mixed scripts that shouldn't appear together
            (
                re.compile(r"[\u4e00-\u9fff][\u0400-\u04ff]"),
                CorruptionType.ENCODING_CORRUPTION,
                CorruptionSeverity.HIGH,
            ),
            # Control characters in text
            (
                re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"),
                CorruptionType.ENCODING_CORRUPTION,
                CorruptionSeverity.CRITICAL,
            ),
            # Malformed Unicode replacement characters
            (
                re.compile(r"\ufffd{2,}"),
                CorruptionType.ENCODING_CORRUPTION,
                CorruptionSeverity.HIGH,
            ),
            # Excessive whitespace or newlines
            (
                re.compile(r"\s{10,}"),
                CorruptionType.SEMANTIC_CORRUPTION,
                CorruptionSeverity.LOW,
            ),
            # Broken word boundaries (letters without spaces)
            (
                re.compile(r"[a-zA-Z]{50,}"),
                CorruptionType.SEMANTIC_CORRUPTION,
                CorruptionSeverity.MEDIUM,
            ),
        ]

        # Character categories that indicate encoding issues
        self.problematic_categories = {
            "Cc",  # Control characters
            "Cf",  # Format characters (except common ones)
            "Cs",  # Surrogate characters
            "Co",  # Private use characters
        }

    def analyze_output(self, text: str) -> CorruptionReport:
        """
        Analyze text output for various types of corruption.

        Args:
            text: The text to analyze

        Returns:
            CorruptionReport with details about detected corruption
        """
        if not text:
            return CorruptionReport(
                corruption_type=CorruptionType.NONE,
                affected_range=(0, 0),
                severity=CorruptionSeverity.LOW,
                recovery_possible=True,
                details="Empty text input",
            )

        # Check for encoding issues first
        encoding_report = self.detect_encoding_issues(text)
        if encoding_report.corruption_type != CorruptionType.NONE:
            return encoding_report

        # Check for semantic corruption patterns
        semantic_report = self._detect_semantic_corruption(text)
        if semantic_report.corruption_type != CorruptionType.NONE:
            return semantic_report

        # Check for statistical anomalies
        statistical_report = self._detect_statistical_anomalies(text)
        if statistical_report.corruption_type != CorruptionType.NONE:
            return statistical_report

        # No corruption detected
        return CorruptionReport(
            corruption_type=CorruptionType.NONE,
            affected_range=(0, len(text)),
            severity=CorruptionSeverity.LOW,
            recovery_possible=True,
            details="No corruption detected",
        )

    def detect_encoding_issues(self, text: str) -> CorruptionReport:
        """
        Detect encoding-related corruption in text.

        Args:
            text: The text to analyze

        Returns:
            CorruptionReport with encoding issue details
        """
        issues = []

        # Check for invalid Unicode characters
        for i, char in enumerate(text):
            try:
                category = unicodedata.category(char)
                if category in self.problematic_categories:
                    issues.append((i, f"Problematic character category: {category}"))
            except ValueError:
                issues.append((i, "Invalid Unicode character"))

        # Check for replacement characters
        replacement_chars = []
        for match in re.finditer(r"\ufffd", text):
            replacement_chars.append(match.start())

        if replacement_chars:
            issues.extend(
                [(pos, "Unicode replacement character") for pos in replacement_chars]
            )

        # Check for mixed encodings (heuristic)
        try:
            # Try to encode as various encodings to detect mixed encoding issues
            text.encode("utf-8")
            text.encode("ascii", errors="strict")
        except UnicodeEncodeError as e:
            issues.append((e.start, f"Encoding error: {e.reason}"))
        except UnicodeDecodeError as e:
            issues.append((e.start, f"Decoding error: {e.reason}"))

        if issues:
            first_issue = min(issues, key=lambda x: x[0])
            severity = (
                CorruptionSeverity.HIGH
                if len(issues) > 5
                else CorruptionSeverity.MEDIUM
            )

            return CorruptionReport(
                corruption_type=CorruptionType.ENCODING_CORRUPTION,
                affected_range=(first_issue[0], first_issue[0] + 1),
                severity=severity,
                recovery_possible=True,
                details=f"Encoding issues detected: {len(issues)} problems found. First: {first_issue[1]}",
            )

        return CorruptionReport(
            corruption_type=CorruptionType.NONE,
            affected_range=(0, 0),
            severity=CorruptionSeverity.LOW,
            recovery_possible=True,
            details="No encoding issues detected",
        )

    def detect_token_corruption(
        self, tokens: List[EnhancedTokenChunk]
    ) -> List[CorruptionReport]:
        """
        Detect corruption in a sequence of tokens.

        Args:
            tokens: List of tokens to analyze

        Returns:
            List of CorruptionReports for each detected issue
        """
        reports = []

        if not tokens:
            return reports

        # Check individual tokens
        for i, token in enumerate(tokens):
            # Verify checksum if available
            if token.checksum and not token.verify_checksum():
                reports.append(
                    CorruptionReport(
                        corruption_type=CorruptionType.CHECKSUM_MISMATCH,
                        affected_range=(i, i + 1),
                        severity=CorruptionSeverity.HIGH,
                        recovery_possible=True,
                        details=f"Checksum mismatch for token at position {i}",
                    )
                )

            # Analyze token text
            text_report = self.analyze_output(token.text)
            if text_report.corruption_type != CorruptionType.NONE:
                # Adjust range to token position
                reports.append(
                    CorruptionReport(
                        corruption_type=text_report.corruption_type,
                        affected_range=(i, i + 1),
                        severity=text_report.severity,
                        recovery_possible=text_report.recovery_possible,
                        details=f"Token {i}: {text_report.details}",
                    )
                )

        # Check sequence integrity
        sequence_report = self._detect_sequence_corruption(tokens)
        if sequence_report.corruption_type != CorruptionType.NONE:
            reports.append(sequence_report)

        return reports

    def _detect_semantic_corruption(self, text: str) -> CorruptionReport:
        """Detect semantic corruption using pattern matching."""
        for pattern, corruption_type, severity in self.corruption_patterns:
            match = pattern.search(text)
            if match:
                return CorruptionReport(
                    corruption_type=corruption_type,
                    affected_range=(match.start(), match.end()),
                    severity=severity,
                    recovery_possible=True,
                    details=f"Pattern match: {pattern.pattern} at position {match.start()}-{match.end()}",
                )

        return CorruptionReport(
            corruption_type=CorruptionType.NONE,
            affected_range=(0, 0),
            severity=CorruptionSeverity.LOW,
            recovery_possible=True,
            details="No semantic corruption patterns detected",
        )

    def _detect_statistical_anomalies(self, text: str) -> CorruptionReport:
        """Detect corruption using statistical analysis."""
        if len(text) < 10:
            return CorruptionReport(
                corruption_type=CorruptionType.NONE,
                affected_range=(0, 0),
                severity=CorruptionSeverity.LOW,
                recovery_possible=True,
                details="Text too short for statistical analysis",
            )

        # Calculate character entropy
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Very low entropy (repeated characters)
        if len(char_counts) < len(text) * 0.1:  # Less than 10% unique characters
            return CorruptionReport(
                corruption_type=CorruptionType.SEMANTIC_CORRUPTION,
                affected_range=(0, len(text)),
                severity=CorruptionSeverity.MEDIUM,
                recovery_possible=True,
                details=f"Low entropy detected: {len(char_counts)} unique chars in {len(text)} total",
            )

        # Check for excessive non-printable characters
        non_printable = sum(
            1 for char in text if not char.isprintable() and char not in "\n\t\r"
        )
        if non_printable > len(text) * 0.05:  # More than 5% non-printable
            return CorruptionReport(
                corruption_type=CorruptionType.ENCODING_CORRUPTION,
                affected_range=(0, len(text)),
                severity=CorruptionSeverity.HIGH,
                recovery_possible=False,
                details=f"Excessive non-printable characters: {non_printable}/{len(text)}",
            )

        return CorruptionReport(
            corruption_type=CorruptionType.NONE,
            affected_range=(0, 0),
            severity=CorruptionSeverity.LOW,
            recovery_possible=True,
            details="No statistical anomalies detected",
        )

    def _detect_sequence_corruption(
        self, tokens: List[EnhancedTokenChunk]
    ) -> CorruptionReport:
        """Detect corruption in token sequence ordering."""
        if len(tokens) < 2:
            return CorruptionReport(
                corruption_type=CorruptionType.NONE,
                affected_range=(0, 0),
                severity=CorruptionSeverity.LOW,
                recovery_possible=True,
                details="Insufficient tokens for sequence analysis",
            )

        # Check for sequence position consistency
        positions = [
            token.sequence_position for token in tokens if token.sequence_position > 0
        ]
        if positions:
            # Check for gaps or duplicates
            sorted_positions = sorted(positions)
            expected = list(range(min(positions), max(positions) + 1))

            if sorted_positions != expected:
                return CorruptionReport(
                    corruption_type=CorruptionType.SEQUENCE_CORRUPTION,
                    affected_range=(0, len(tokens)),
                    severity=CorruptionSeverity.MEDIUM,
                    recovery_possible=True,
                    details=f"Sequence position mismatch: expected {expected}, got {sorted_positions}",
                )

        # Check for timestamp ordering (should be generally increasing)
        timestamps = [token.generation_timestamp for token in tokens]
        if len(timestamps) > 1:
            out_of_order = 0
            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i - 1]:
                    out_of_order += 1

            if out_of_order > len(timestamps) * 0.2:  # More than 20% out of order
                return CorruptionReport(
                    corruption_type=CorruptionType.SEQUENCE_CORRUPTION,
                    affected_range=(0, len(tokens)),
                    severity=CorruptionSeverity.LOW,
                    recovery_possible=True,
                    details=f"Timestamp ordering issues: {out_of_order} out of {len(timestamps)} tokens",
                )

        return CorruptionReport(
            corruption_type=CorruptionType.NONE,
            affected_range=(0, 0),
            severity=CorruptionSeverity.LOW,
            recovery_possible=True,
            details="No sequence corruption detected",
        )
