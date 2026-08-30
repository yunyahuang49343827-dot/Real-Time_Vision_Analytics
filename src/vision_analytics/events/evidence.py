"""Config-driven JPG evidence capture for rule-generated events."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Iterable, Mapping, Sequence

import cv2
import yaml

from vision_analytics.events.schema import EVENT_TYPES, SEVERITIES, STATUSES, EventRecord
from vision_analytics.tracking.schema import TrackRecord

EVIDENCE_MANIFEST_FIELDS = (
    "event_id", "video_id", "frame_index", "timestamp_seconds", "event_type",
    "severity", "status", "evidence_path", "file_size_bytes",
)


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    capture_event_types: frozenset[str]
    capture_severities: frozenset[str]
    capture_statuses: frozenset[str]
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        if not self.capture_event_types <= EVENT_TYPES:
            raise ValueError("evidence policy contains unsupported event type")
        if not self.capture_severities <= SEVERITIES:
            raise ValueError("evidence policy contains unsupported severity")
        if not self.capture_statuses <= STATUSES:
            raise ValueError("evidence policy contains unsupported status")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

    def should_capture(self, event: EventRecord) -> bool:
        return (
            event.event_type in self.capture_event_types
            and (
                event.severity in self.capture_severities
                or event.status in self.capture_statuses
            )
        )


@dataclass(frozen=True, slots=True)
class EvidenceManifestRecord:
    event_id: str
    video_id: str
    frame_index: int
    timestamp_seconds: float
    event_type: str
    severity: str
    status: str
    evidence_path: str
    file_size_bytes: int

    def to_row(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "event_type": self.event_type,
            "severity": self.severity,
            "status": self.status,
            "evidence_path": self.evidence_path,
            "file_size_bytes": self.file_size_bytes,
        }


def load_evidence_policy(path: Path) -> EvidencePolicy:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = payload.get("evidence_policy")
    if not isinstance(raw, Mapping):
        raise ValueError("evidence_policy is required")
    return EvidencePolicy(
        capture_event_types=frozenset(raw.get("capture_event_types", ())),
        capture_severities=frozenset(raw.get("capture_severities", ())),
        capture_statuses=frozenset(raw.get("capture_statuses", ())),
        jpeg_quality=int(raw.get("jpeg_quality", 90)),
    )


class EvidenceCapture:
    """Write at most one annotated current-frame JPG for each eligible event."""

    def __init__(self, policy: EvidencePolicy, output_dir: Path, relative_dir: Path) -> None:
        if relative_dir.is_absolute():
            raise ValueError("relative_dir must be relative")
        self.policy = policy
        self.output_dir = output_dir
        self.relative_dir = relative_dir
        self.manifest_records: list[EvidenceManifestRecord] = []
        self._captured: dict[str, EventRecord] = {}

    @staticmethod
    def filename(event_id: str) -> str:
        if not event_id or PurePath(event_id).name != event_id or event_id in {".", ".."}:
            raise ValueError("event_id is not safe for an evidence filename")
        return f"{event_id}.jpg"

    @staticmethod
    def _track_by_id(tracks: Sequence[TrackRecord], track_id: int | None) -> TrackRecord | None:
        if track_id is None:
            return None
        return next((track for track in tracks if track.track_id == track_id), None)

    @staticmethod
    def _draw_track(frame: object, track: TrackRecord | None, color: tuple[int, int, int], label: str) -> None:
        if track is None:
            return
        x1, y1, x2, y2 = (
            round(track.bbox.x1), round(track.bbox.y1),
            round(track.bbox.x2), round(track.bbox.y2),
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    def _annotate(self, frame: object, event: EventRecord, tracks: Sequence[TrackRecord]) -> object:
        image = frame.copy()
        primary = self._track_by_id(tracks, event.track_id)
        secondary = self._track_by_id(tracks, event.secondary_track_id)
        self._draw_track(image, primary, (0, 255, 255), f"PRIMARY T{event.track_id}")
        self._draw_track(image, secondary, (255, 0, 255), f"SECONDARY T{event.secondary_track_id}")

        context = event.zone_id or event.line_id or ""
        if event.rule_value:
            context = f"{context} {event.rule_value}".strip()
        lines = [
            f"{event.event_type} | {event.severity}",
            f"t={event.timestamp_seconds:.2f}s | primary T{event.track_id if event.track_id is not None else '-'}",
        ]
        if event.secondary_track_id is not None:
            lines.append(f"secondary T{event.secondary_track_id}")
        if context:
            lines.append(context)
        panel_height = 18 + 30 * len(lines)
        cv2.rectangle(image, (12, 12), (min(image.shape[1] - 12, 750), panel_height), (0, 0, 0), -1)
        for index, text in enumerate(lines):
            cv2.putText(image, text, (24, 42 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def capture(
        self,
        frame: object,
        event: EventRecord,
        tracks: Sequence[TrackRecord] = (),
    ) -> EventRecord:
        if not self.policy.should_capture(event):
            return event
        if event.event_id in self._captured:
            return self._captured[event.event_id]

        filename = self.filename(event.event_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        absolute_path = self.output_dir / filename
        relative_path = (self.relative_dir / filename).as_posix()
        annotated = self._annotate(frame, event, tracks)
        written = cv2.imwrite(
            str(absolute_path), annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self.policy.jpeg_quality],
        )
        if not written or not absolute_path.is_file() or absolute_path.stat().st_size <= 0:
            raise OSError(f"failed to write evidence image: {absolute_path}")
        if cv2.imread(str(absolute_path)) is None:
            raise OSError(f"evidence image is not readable: {absolute_path}")
        if absolute_path.stem != event.event_id:
            raise OSError("evidence filename does not match event_id")

        updated = replace(event, evidence_path=relative_path)
        self._captured[event.event_id] = updated
        self.manifest_records.append(EvidenceManifestRecord(
            event_id=event.event_id,
            video_id=event.video_id,
            frame_index=event.frame_index,
            timestamp_seconds=event.timestamp_seconds,
            event_type=event.event_type,
            severity=event.severity,
            status=event.status,
            evidence_path=relative_path,
            file_size_bytes=absolute_path.stat().st_size,
        ))
        return updated

    def capture_events(
        self,
        frame: object,
        events: Iterable[EventRecord],
        tracks: Sequence[TrackRecord] = (),
    ) -> list[EventRecord]:
        return [self.capture(frame, event, tracks) for event in events]

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVIDENCE_MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(record.to_row() for record in self.manifest_records)
