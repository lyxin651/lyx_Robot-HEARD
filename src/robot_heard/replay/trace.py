"""Append-only JSONL trace and run-provenance persistence for R1."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from robot_heard.replay.events import ContractValidationError, TraceEvent


PathLike = Union[str, Path]


class TraceWriter:
    """Persist one event per JSONL line and a separate run sidecar.

    A fresh run refuses to open either existing artifact.  Each append is
    flushed and fsynced, matching the repository's durable JSONL evidence
    pattern.  No resume or post-processing rewrite is supported in R1.
    """

    REQUIRED_PROVENANCE_FIELDS = (
        "schema_version",
        "run_id",
        "replay_mode",
        "source_identity",
        "source_sample_rate",
        "channel_ids",
        "packet_samples",
        "queue_policy",
        "consumer_identity",
        "consumer_capabilities",
        "oracle_condition",
        "clock_policy",
        "trace_path",
    )

    def __init__(
        self,
        trace_path: PathLike,
        run_provenance_path: PathLike,
        run_provenance: Mapping[str, Any],
        *,
        schema_version: Optional[str] = None,
    ) -> None:
        self.trace_path = Path(trace_path).expanduser().resolve(strict=False)
        self.run_provenance_path = (
            Path(run_provenance_path).expanduser().resolve(strict=False)
        )
        if self.trace_path.exists():
            raise FileExistsError(f"trace already exists; fresh run refused: {self.trace_path}")
        if self.run_provenance_path.exists():
            raise FileExistsError(
                f"run provenance already exists; fresh run refused: {self.run_provenance_path}"
            )
        if not isinstance(run_provenance, Mapping):
            raise ContractValidationError("run_provenance must be a mapping")
        provenance = dict(run_provenance)
        missing = [
            field
            for field in self.REQUIRED_PROVENANCE_FIELDS
            if field not in provenance
        ]
        if missing:
            raise ContractValidationError(
                "run provenance missing required fields: " + ", ".join(missing)
            )
        effective_schema_version = schema_version or provenance["schema_version"]
        if not isinstance(effective_schema_version, str) or not effective_schema_version:
            raise ContractValidationError("schema_version must be a non-empty string")
        if provenance["schema_version"] != effective_schema_version:
            raise ContractValidationError(
                "event and run provenance schema_version must match"
            )
        if provenance["trace_path"] != str(self.trace_path):
            raise ContractValidationError("trace_path must equal the resolved trace path")

        self.schema_version = effective_schema_version
        self._next_event_index = 0
        self._closed = False
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_provenance_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        created_trace = False
        try:
            self._handle = self.trace_path.open("x", encoding="utf-8")
            created_trace = True
            self._write_sidecar(provenance)
        except Exception:
            if self._handle is not None:
                self._handle.close()
            if created_trace and self.trace_path.exists():
                self.trace_path.unlink()
            raise

    def _write_sidecar(self, provenance: Mapping[str, Any]) -> None:
        created = False
        try:
            with self.run_provenance_path.open("x", encoding="utf-8") as handle:
                created = True
                json.dump(dict(provenance), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            if created and self.run_provenance_path.exists():
                self.run_provenance_path.unlink()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def append(self, event: TraceEvent) -> None:
        """Append exactly one next event and durably flush it."""

        if self._closed:
            raise RuntimeError("cannot append to a closed trace")
        if event.schema_version != self.schema_version:
            raise ContractValidationError(
                "event schema_version does not match trace schema_version"
            )
        if event.event_index != self._next_event_index:
            raise ContractValidationError(
                f"event_index must be exactly {self._next_event_index}; got {event.event_index}"
            )
        record = event.to_record()
        self._handle.write(
            json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._next_event_index += 1

    def close(self) -> None:
        if not self._closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
