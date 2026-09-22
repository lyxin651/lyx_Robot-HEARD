"""ASR package."""

from .base import ASRBackend, ASRResult
from .batch import (
    BatchProvenance,
    BatchResumeError,
    BatchSummary,
    build_batch_provenance,
    default_error_path,
    run_batch,
    run_metadata_path,
)
from .scoring import (
    NORMALIZATION_POLICY_V0,
    SUPPORTED_NORMALIZATION_POLICIES,
    CharacterErrorCounts,
    ScoringError,
    ScoringSummary,
    character_error_counts,
    compute_rtf,
    normalize_text,
    score_result_record,
    score_results_jsonl,
    validate_normalization_policy,
)

__all__ = [
    "ASRBackend",
    "ASRResult",
    "BatchProvenance",
    "BatchResumeError",
    "BatchSummary",
    "build_batch_provenance",
    "default_error_path",
    "run_batch",
    "run_metadata_path",
    "NORMALIZATION_POLICY_V0",
    "SUPPORTED_NORMALIZATION_POLICIES",
    "CharacterErrorCounts",
    "ScoringError",
    "ScoringSummary",
    "character_error_counts",
    "compute_rtf",
    "normalize_text",
    "score_result_record",
    "score_results_jsonl",
    "validate_normalization_policy",
]
