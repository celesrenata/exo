"""
Distributed inference validation framework for comparing outputs and testing stability.
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from exo.utils.pydantic_ext import CamelCaseModel
from ..types.validation import (
    ValidationResult,
    CorruptionType,
    CorruptionSeverity,
    EnhancedTokenChunk,
    ValidationStatus,
)
from ..types.chunks import TokenChunk
from ..types.models import ModelId


class TestMode(str, Enum):
    """Test execution modes."""

    SINGLE_DEVICE = "single_device"
    DISTRIBUTED = "distributed"
    COMPARISON = "comparison"


class OutputQualityMetric(str, Enum):
    """Metrics for evaluating output quality."""

    COHERENCE = "coherence"
    ENCODING_INTEGRITY = "encoding_integrity"
    SEQUENCE_COMPLETENESS = "sequence_completeness"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TOKEN_ACCURACY = "token_accuracy"


@dataclass
class ComparisonResult:
    """Result of comparing single-device vs distributed outputs."""

    single_device_output: str
    distributed_output: str
    are_identical: bool
    similarity_score: float  # 0.0 to 1.0
    differences: List[str]
    quality_metrics: Dict[OutputQualityMetric, float]
    test_timestamp: datetime
    model_id: ModelId
    prompt: str
    seed: Optional[int] = None


@dataclass
class DeterministicResult:
    """Result of deterministic testing with controlled seeds."""

    seed: int
    prompt: str
    outputs: List[str]  # Multiple runs with same seed
    consistency_score: float  # How consistent outputs are
    is_deterministic: bool
    variations: List[str]  # Differences between runs
    test_timestamp: datetime
    model_id: ModelId


@dataclass
class ValidationScore:
    """Comprehensive validation scoring."""

    overall_score: float  # 0.0 to 1.0
    metric_scores: Dict[OutputQualityMetric, float]
    corruption_detected: bool
    encoding_issues: int
    sequence_gaps: int
    semantic_coherence: float
    details: str


class DistributedInferenceValidator:
    """
    Comprehensive validator for distributed inference outputs.

    Provides comparison between single-device and distributed outputs,
    deterministic testing capabilities, and output quality metrics.
    """

    def __init__(
        self,
        single_device_runner=None,
        distributed_runner=None,
        validation_timeout: float = 30.0,
    ):
        """
        Initialize the validator.

        Args:
            single_device_runner: Runner for single-device inference
            distributed_runner: Runner for distributed inference
            validation_timeout: Timeout for validation operations
        """
        self.single_device_runner = single_device_runner
        self.distributed_runner = distributed_runner
        self.validation_timeout = validation_timeout
        self.test_history: List[ComparisonResult] = []
        self.deterministic_history: List[DeterministicResult] = []

    async def compare_single_vs_distributed(
        self,
        prompt: str,
        model_id: ModelId,
        seed: Optional[int] = None,
        max_tokens: int = 100,
    ) -> ComparisonResult:
        """
        Compare outputs between single-device and distributed inference.

        Args:
            prompt: Input prompt for generation
            model_id: Model to use for inference
            seed: Random seed for deterministic generation
            max_tokens: Maximum tokens to generate

        Returns:
            ComparisonResult with detailed comparison
        """
        if seed is not None:
            random.seed(seed)

        # Generate single-device output
        single_output = await self._generate_single_device(
            prompt, model_id, max_tokens, seed
        )

        # Generate distributed output
        distributed_output = await self._generate_distributed(
            prompt, model_id, max_tokens, seed
        )

        # Compare outputs
        are_identical = single_output == distributed_output
        similarity_score = self._calculate_similarity(single_output, distributed_output)
        differences = self._find_differences(single_output, distributed_output)
        quality_metrics = self._evaluate_quality_metrics(
            single_output, distributed_output
        )

        result = ComparisonResult(
            single_device_output=single_output,
            distributed_output=distributed_output,
            are_identical=are_identical,
            similarity_score=similarity_score,
            differences=differences,
            quality_metrics=quality_metrics,
            test_timestamp=datetime.now(),
            model_id=model_id,
            prompt=prompt,
            seed=seed,
        )

        self.test_history.append(result)
        return result

    async def run_deterministic_test(
        self,
        seed: int,
        prompt: str,
        model_id: ModelId,
        num_runs: int = 3,
        max_tokens: int = 100,
    ) -> DeterministicResult:
        """
        Run deterministic tests with controlled random seeds.

        Args:
            seed: Random seed to use
            prompt: Input prompt
            model_id: Model to use
            num_runs: Number of runs to perform
            max_tokens: Maximum tokens per run

        Returns:
            DeterministicResult with consistency analysis
        """
        outputs = []

        for run in range(num_runs):
            # Use same seed for each run to test determinism
            output = await self._generate_distributed(
                prompt, model_id, max_tokens, seed
            )
            outputs.append(output)

        # Analyze consistency
        consistency_score = self._calculate_consistency(outputs)
        is_deterministic = consistency_score > 0.95  # 95% similarity threshold
        variations = self._find_variations(outputs)

        result = DeterministicResult(
            seed=seed,
            prompt=prompt,
            outputs=outputs,
            consistency_score=consistency_score,
            is_deterministic=is_deterministic,
            variations=variations,
            test_timestamp=datetime.now(),
            model_id=model_id,
        )

        self.deterministic_history.append(result)
        return result

    def validate_output_quality(
        self, tokens: List[EnhancedTokenChunk], expected_output: Optional[str] = None
    ) -> ValidationScore:
        """
        Evaluate output quality using multiple metrics.

        Args:
            tokens: List of enhanced token chunks to validate
            expected_output: Optional expected output for comparison

        Returns:
            ValidationScore with detailed metrics
        """
        # Reconstruct text from tokens
        output_text = "".join(token.text for token in tokens)

        # Calculate individual metrics
        metric_scores = {}

        # Encoding integrity
        metric_scores[OutputQualityMetric.ENCODING_INTEGRITY] = (
            self._check_encoding_integrity(output_text)
        )

        # Sequence completeness
        metric_scores[OutputQualityMetric.SEQUENCE_COMPLETENESS] = (
            self._check_sequence_completeness(tokens)
        )

        # Semantic coherence
        metric_scores[OutputQualityMetric.COHERENCE] = self._check_semantic_coherence(
            output_text
        )

        # Token accuracy (if we have validation results)
        metric_scores[OutputQualityMetric.TOKEN_ACCURACY] = self._check_token_accuracy(
            tokens
        )

        # Semantic similarity (if expected output provided)
        if expected_output:
            metric_scores[OutputQualityMetric.SEMANTIC_SIMILARITY] = (
                self._calculate_similarity(output_text, expected_output)
            )

        # Calculate overall score
        overall_score = sum(metric_scores.values()) / len(metric_scores)

        # Count issues
        corruption_detected = any(
            token.validation_status == ValidationStatus.CORRUPTED for token in tokens
        )
        encoding_issues = sum(
            1
            for token in tokens
            if token.validation_result
            and token.validation_result.error_type == CorruptionType.ENCODING_CORRUPTION
        )
        sequence_gaps = self._count_sequence_gaps(tokens)

        return ValidationScore(
            overall_score=overall_score,
            metric_scores=metric_scores,
            corruption_detected=corruption_detected,
            encoding_issues=encoding_issues,
            sequence_gaps=sequence_gaps,
            semantic_coherence=metric_scores[OutputQualityMetric.COHERENCE],
            details=f"Validated {len(tokens)} tokens with {encoding_issues} encoding issues",
        )

    def get_test_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all tests."""
        if not self.test_history:
            return {"message": "No tests run yet"}

        total_tests = len(self.test_history)
        identical_outputs = sum(
            1 for result in self.test_history if result.are_identical
        )
        avg_similarity = (
            sum(result.similarity_score for result in self.test_history) / total_tests
        )

        # Quality metrics averages
        quality_averages = {}
        for metric in OutputQualityMetric:
            scores = [
                result.quality_metrics.get(metric, 0.0) for result in self.test_history
            ]
            quality_averages[metric.value] = (
                sum(scores) / len(scores) if scores else 0.0
            )

        # Deterministic test stats
        deterministic_tests = len(self.deterministic_history)
        deterministic_success = sum(
            1 for result in self.deterministic_history if result.is_deterministic
        )

        return {
            "total_comparison_tests": total_tests,
            "identical_outputs": identical_outputs,
            "identical_rate": identical_outputs / total_tests
            if total_tests > 0
            else 0.0,
            "average_similarity": avg_similarity,
            "quality_metrics_averages": quality_averages,
            "deterministic_tests": deterministic_tests,
            "deterministic_success_rate": (
                deterministic_success / deterministic_tests
                if deterministic_tests > 0
                else 0.0
            ),
            "last_test_time": (
                self.test_history[-1].test_timestamp.isoformat()
                if self.test_history
                else None
            ),
        }

    # Private helper methods

    async def _generate_single_device(
        self, prompt: str, model_id: ModelId, max_tokens: int, seed: Optional[int]
    ) -> str:
        """Generate output using single-device inference."""
        if not self.single_device_runner:
            # Fallback: simulate single device output
            return f"Single device output for: {prompt[:50]}..."

        # TODO: Implement actual single-device generation
        # This would integrate with the actual runner
        return f"Single device: {prompt}"

    async def _generate_distributed(
        self, prompt: str, model_id: ModelId, max_tokens: int, seed: Optional[int]
    ) -> str:
        """Generate output using distributed inference."""
        if not self.distributed_runner:
            # Fallback: simulate distributed output
            return f"Distributed output for: {prompt[:50]}..."

        # TODO: Implement actual distributed generation
        # This would integrate with the actual distributed runner
        return f"Distributed: {prompt}"

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if text1 == text2:
            return 1.0

        # Simple character-level similarity
        if not text1 or not text2:
            return 0.0

        # Levenshtein distance approximation
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0

        # Count matching characters
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        return matches / max_len

    def _find_differences(self, text1: str, text2: str) -> List[str]:
        """Find specific differences between texts."""
        differences = []

        if len(text1) != len(text2):
            differences.append(f"Length difference: {len(text1)} vs {len(text2)}")

        # Find character differences
        for i, (c1, c2) in enumerate(zip(text1, text2)):
            if c1 != c2:
                differences.append(f"Position {i}: '{c1}' vs '{c2}'")
                if len(differences) > 10:  # Limit output
                    differences.append("... (more differences)")
                    break

        return differences

    def _evaluate_quality_metrics(
        self, single_output: str, distributed_output: str
    ) -> Dict[OutputQualityMetric, float]:
        """Evaluate quality metrics for both outputs."""
        metrics = {}

        # Encoding integrity
        metrics[OutputQualityMetric.ENCODING_INTEGRITY] = min(
            self._check_encoding_integrity(single_output),
            self._check_encoding_integrity(distributed_output),
        )

        # Coherence
        metrics[OutputQualityMetric.COHERENCE] = min(
            self._check_semantic_coherence(single_output),
            self._check_semantic_coherence(distributed_output),
        )

        # Similarity between outputs
        metrics[OutputQualityMetric.SEMANTIC_SIMILARITY] = self._calculate_similarity(
            single_output, distributed_output
        )

        return metrics

    def _calculate_consistency(self, outputs: List[str]) -> float:
        """Calculate consistency score across multiple outputs."""
        if len(outputs) <= 1:
            return 1.0

        # Compare all pairs
        total_comparisons = 0
        total_similarity = 0.0

        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarity = self._calculate_similarity(outputs[i], outputs[j])
                total_similarity += similarity
                total_comparisons += 1

        return total_similarity / total_comparisons if total_comparisons > 0 else 0.0

    def _find_variations(self, outputs: List[str]) -> List[str]:
        """Find variations between multiple outputs."""
        if len(outputs) <= 1:
            return []

        variations = []
        base_output = outputs[0]

        for i, output in enumerate(outputs[1:], 1):
            if output != base_output:
                variations.append(f"Run {i + 1} differs from run 1")

        return variations

    def _check_encoding_integrity(self, text: str) -> float:
        """Check encoding integrity of text."""
        try:
            # Try to encode/decode as UTF-8
            text.encode("utf-8").decode("utf-8")

            # Check for common encoding issues
            encoding_issues = 0
            if "�" in text:  # Replacement character
                encoding_issues += text.count("�")

            # Check for invalid Unicode sequences
            try:
                text.encode("ascii", errors="strict")
            except UnicodeEncodeError:
                # Non-ASCII is fine, but check for malformed sequences
                pass

            # Score based on issues found
            if encoding_issues == 0:
                return 1.0
            else:
                return max(0.0, 1.0 - (encoding_issues / len(text)))

        except UnicodeError:
            return 0.0

    def _check_sequence_completeness(self, tokens: List[EnhancedTokenChunk]) -> float:
        """Check if token sequence is complete."""
        if not tokens:
            return 0.0

        # Check for gaps in sequence positions
        positions = [
            token.sequence_position for token in tokens if token.sequence_position > 0
        ]
        if not positions:
            return 1.0  # No sequence info available

        positions.sort()
        expected_positions = list(range(positions[0], positions[-1] + 1))
        missing_count = len(set(expected_positions) - set(positions))

        completeness = 1.0 - (missing_count / len(expected_positions))
        return max(0.0, completeness)

    def _check_semantic_coherence(self, text: str) -> float:
        """Basic semantic coherence check using heuristics."""
        if not text:
            return 0.0

        # Simple heuristics for coherence
        score = 1.0

        # Check for repeated characters (sign of corruption)
        for char in set(text):
            if char * 10 in text:  # 10+ repeated characters
                score -= 0.2

        # Check for random character sequences
        import re

        random_patterns = [
            r"[^\w\s]{5,}",  # 5+ consecutive non-word characters
            r"\d{10,}",  # 10+ consecutive digits
            r"[A-Z]{10,}",  # 10+ consecutive uppercase
        ]

        for pattern in random_patterns:
            if re.search(pattern, text):
                score -= 0.1

        # Check for proper sentence structure
        sentences = text.split(".")
        if len(sentences) > 1:
            # Basic sentence structure check
            proper_sentences = sum(
                1
                for s in sentences
                if s.strip() and len(s.strip()) > 3 and " " in s.strip()
            )
            sentence_score = proper_sentences / len(sentences)
            score = (score + sentence_score) / 2

        return max(0.0, min(1.0, score))

    def _check_token_accuracy(self, tokens: List[EnhancedTokenChunk]) -> float:
        """Check accuracy of tokens based on validation results."""
        if not tokens:
            return 0.0

        valid_tokens = sum(
            1 for token in tokens if token.validation_status == ValidationStatus.VALID
        )

        return valid_tokens / len(tokens)

    def _count_sequence_gaps(self, tokens: List[EnhancedTokenChunk]) -> int:
        """Count gaps in token sequence."""
        positions = [
            token.sequence_position for token in tokens if token.sequence_position > 0
        ]

        if len(positions) <= 1:
            return 0

        positions.sort()
        gaps = 0

        for i in range(1, len(positions)):
            if positions[i] - positions[i - 1] > 1:
                gaps += positions[i] - positions[i - 1] - 1

        return gaps
