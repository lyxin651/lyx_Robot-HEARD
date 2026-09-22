from __future__ import annotations

import json
import math
import os
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

PathLike = Union[str, Path]
NORMALIZATION_POLICY_V0 = "v0_nfkc_casefold_strip_space_punct"
SUPPORTED_NORMALIZATION_POLICIES = (NORMALIZATION_POLICY_V0,)


class ScoringError(ValueError):
    """Raised when a result record cannot be scored safely."""


@dataclass(frozen=True)
class CharacterErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_chars: int
    cer: Optional[float]

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class ScoringSummary:
    records: int
    reference_scored: int
    audio_sec_sum: float
    decode_sec_sum: float
    global_rtf: float
    cer_s: int
    cer_d: int
    cer_i: int
    cer_n: int
    global_cer: Optional[float]
    normalization_policy: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_normalization_policy(policy: str) -> str:
    if not isinstance(policy, str) or not policy.strip():
        raise ScoringError("normalization_policy must be a non-empty string")
    if policy != NORMALIZATION_POLICY_V0:
        supported = ", ".join(SUPPORTED_NORMALIZATION_POLICIES)
        raise ScoringError(
            f"unsupported normalization_policy={policy!r}; supported: {supported}"
        )
    return policy


def normalize_text(
    text: str,
    *,
    policy: str = NORMALIZATION_POLICY_V0,
) -> str:
    """Apply the explicit V0 scoring normalizer.

    This is a local diagnostic policy, not a claim of equivalence to the
    official MISP scorer. It applies Unicode NFKC, casefolding, and removes
    Unicode whitespace plus punctuation characters.
    """
    validate_normalization_policy(policy)
    if not isinstance(text, str):
        raise ScoringError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def character_error_counts(reference: str, hypothesis: str) -> CharacterErrorCounts:
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise ScoringError("reference and hypothesis must be strings")

    # State: (edit_cost, substitutions, deletions, insertions).
    # Match the released MISP compute-wer.py alignment semantics exactly:
    # candidates are considered in deletion, insertion, diagonal order and a
    # candidate replaces the current best only when its cost is strictly lower.
    # Therefore equal-cost ties prefer deletion, then insertion, then diagonal.
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]

    for ref_index, ref_char in enumerate(reference, start=1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            deletion_prev = previous[hyp_index]
            deletion = (
                deletion_prev[0] + 1,
                deletion_prev[1],
                deletion_prev[2] + 1,
                deletion_prev[3],
            )

            insertion_prev = current[hyp_index - 1]
            insertion = (
                insertion_prev[0] + 1,
                insertion_prev[1],
                insertion_prev[2],
                insertion_prev[3] + 1,
            )

            diagonal_prev = previous[hyp_index - 1]
            if ref_char == hyp_char:
                diagonal = diagonal_prev
            else:
                diagonal = (
                    diagonal_prev[0] + 1,
                    diagonal_prev[1] + 1,
                    diagonal_prev[2],
                    diagonal_prev[3],
                )

            best = deletion
            if insertion[0] < best[0]:
                best = insertion
            if diagonal[0] < best[0]:
                best = diagonal

            current.append(best)
        previous = current

    _, substitutions, deletions, insertions = previous[-1]
    reference_chars = len(reference)
    errors = substitutions + deletions + insertions
    cer = errors / reference_chars if reference_chars > 0 else None
    return CharacterErrorCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_chars=reference_chars,
        cer=cer,
    )


def compute_rtf(*, decode_sec: float, audio_sec: float) -> float:
    if isinstance(decode_sec, bool) or not isinstance(decode_sec, (int, float)):
        raise ScoringError("decode_sec must be a finite non-negative number")
    if not math.isfinite(float(decode_sec)) or decode_sec < 0:
        raise ScoringError("decode_sec must be a finite non-negative number")
    if isinstance(audio_sec, bool) or not isinstance(audio_sec, (int, float)):
        raise ScoringError("audio_sec must be a finite positive number")
    if not math.isfinite(float(audio_sec)) or audio_sec <= 0:
        raise ScoringError("audio_sec must be a finite positive number")
    return float(decode_sec) / float(audio_sec)


def score_result_record(
    record: Mapping[str, Any],
    *,
    normalization_policy: str = NORMALIZATION_POLICY_V0,
) -> Dict[str, Any]:
    validate_normalization_policy(normalization_policy)
    if not isinstance(record, Mapping):
        raise ScoringError("result record must be a mapping")
    if record.get("status") != "success":
        raise ScoringError("S5 scoring accepts only status='success' records")

    segment_id = record.get("segment_id")
    if not isinstance(segment_id, str) or not segment_id.strip():
        raise ScoringError("result record must contain a non-empty segment_id")

    text_raw = record.get("text_raw")
    if not isinstance(text_raw, str):
        raise ScoringError(f"segment_id={segment_id!r}: text_raw must be a string")

    reference = record.get("reference")
    if reference is not None and not isinstance(reference, str):
        raise ScoringError(
            f"segment_id={segment_id!r}: reference must be a string or null"
        )

    audio_sec = record.get("audio_sec")
    decode_sec = record.get("decode_sec")
    rtf = compute_rtf(decode_sec=decode_sec, audio_sec=audio_sec)
    text_norm = normalize_text(text_raw, policy=normalization_policy)

    scored = dict(record)
    scored.update(
        {
            "result_schema_version": 2,
            "normalization_policy": normalization_policy,
            "text_norm": text_norm,
            "rtf": rtf,
        }
    )

    if reference is None:
        scored.update(
            {
                "reference_norm": None,
                "cer_s": None,
                "cer_d": None,
                "cer_i": None,
                "cer_n": None,
                "cer": None,
            }
        )
        return scored

    reference_norm = normalize_text(reference, policy=normalization_policy)
    counts = character_error_counts(reference_norm, text_norm)
    scored.update(
        {
            "reference_norm": reference_norm,
            "cer_s": counts.substitutions,
            "cer_d": counts.deletions,
            "cer_i": counts.insertions,
            "cer_n": counts.reference_chars,
            "cer": counts.cer,
        }
    )
    return scored


class _SummaryAccumulator:
    def __init__(self, normalization_policy: str):
        self.normalization_policy = normalization_policy
        self.records = 0
        self.reference_scored = 0
        self.audio_sec_sum = 0.0
        self.decode_sec_sum = 0.0
        self.cer_s = 0
        self.cer_d = 0
        self.cer_i = 0
        self.cer_n = 0

    def add(self, record: Mapping[str, Any]) -> None:
        self.records += 1
        self.audio_sec_sum += float(record["audio_sec"])
        self.decode_sec_sum += float(record["decode_sec"])
        if record.get("reference") is not None:
            self.reference_scored += 1
            self.cer_s += int(record["cer_s"])
            self.cer_d += int(record["cer_d"])
            self.cer_i += int(record["cer_i"])
            self.cer_n += int(record["cer_n"])

    def finish(self) -> ScoringSummary:
        if self.records <= 0:
            raise ScoringError("cannot summarize an empty result set")
        if self.audio_sec_sum <= 0:
            raise ScoringError("total audio duration must be positive")

        errors = self.cer_s + self.cer_d + self.cer_i
        return ScoringSummary(
            records=self.records,
            reference_scored=self.reference_scored,
            audio_sec_sum=self.audio_sec_sum,
            decode_sec_sum=self.decode_sec_sum,
            global_rtf=self.decode_sec_sum / self.audio_sec_sum,
            cer_s=self.cer_s,
            cer_d=self.cer_d,
            cer_i=self.cer_i,
            cer_n=self.cer_n,
            global_cer=errors / self.cer_n if self.cer_n > 0 else None,
            normalization_policy=self.normalization_policy,
        )


def score_results_jsonl(
    input_path: PathLike,
    output_path: PathLike,
    *,
    normalization_policy: str = NORMALIZATION_POLICY_V0,
    overwrite: bool = False,
) -> ScoringSummary:
    validate_normalization_policy(normalization_policy)
    source = Path(input_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve(strict=False)

    if source == output:
        raise ScoringError(
            "input_path and output_path must differ so text_raw/results remain auditable"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"scored output already exists; choose a new path or use overwrite=True: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    seen = set()
    accumulator = _SummaryAccumulator(normalization_policy)

    try:
        with source.open("r", encoding="utf-8") as source_handle, temporary.open(
            "w", encoding="utf-8"
        ) as output_handle:
            for line_number, raw_line in enumerate(source_handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ScoringError(
                        f"{source}: line {line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise ScoringError(
                        f"{source}: line {line_number}: result record must be a JSON object"
                    )

                segment_id = record.get("segment_id")
                if not isinstance(segment_id, str) or not segment_id.strip():
                    raise ScoringError(
                        f"{source}: line {line_number}: missing/non-string segment_id"
                    )
                if segment_id in seen:
                    raise ScoringError(
                        f"{source}: line {line_number}, segment_id={segment_id!r}: "
                        "duplicate result segment"
                    )
                seen.add(segment_id)

                scored = score_result_record(
                    record,
                    normalization_policy=normalization_policy,
                )
                output_handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
                accumulator.add(scored)

            output_handle.flush()
            os.fsync(output_handle.fileno())

        summary = accumulator.finish()
        temporary.replace(output)
        return summary
    finally:
        if temporary.exists():
            temporary.unlink()
