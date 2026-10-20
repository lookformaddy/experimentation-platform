"""Exposure and event logging: an append-only store.

Everything analysis needs arrives as events. An *exposure* event
(``event_name == "exposure"``) records that a unit entered a variant;
metric events reuse the metric's name (a conversion event for
``signup`` is named ``signup``, a continuous observation for
``revenue`` is named ``revenue`` and carries ``value``).

The store is append-only and defensive in three specific ways:

- **Deduplication by event id.** Producers retry; an ``event_id`` that
  was already accepted is dropped and counted, never double-counted.
- **Duplicate exposures.** A unit exposed twice (page refresh, retry)
  still counts once: analysis reads the *first* exposure per unit.
  Repeats are kept in the log and counted, not silently merged.
- **Late events.** Each experiment has a watermark (its newest event
  timestamp). An event arriving more than ``late_window_seconds``
  behind the watermark is flagged late: stored, counted, and excluded
  from metric computation unless the caller opts in — a backfill must
  be a decision, not an accident.

Persistence is JSON Lines: one event per line, loadable back into an
identical store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

EXPOSURE = "exposure"


@dataclass(frozen=True)
class Event:
    event_id: str
    experiment_id: str
    unit_id: str
    variant: str
    event_name: str
    timestamp: datetime
    value: float | None = None
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "experiment_id": self.experiment_id,
            "unit_id": self.unit_id,
            "variant": self.variant,
            "event_name": self.event_name,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            event_id=data["event_id"],
            experiment_id=data["experiment_id"],
            unit_id=data["unit_id"],
            variant=data["variant"],
            event_name=data["event_name"],
            timestamp=ts,
            value=data.get("value"),
            attributes=data.get("attributes") or {},
        )


@dataclass
class IngestResult:
    accepted: bool
    duplicate: bool = False
    late: bool = False


class EventStore:
    def __init__(self, late_window_seconds: float = 3600.0) -> None:
        self.late_window_seconds = late_window_seconds
        self._events: list[Event] = []
        self._seen_ids: set[str] = set()
        self._late_ids: set[str] = set()
        self._watermarks: dict[str, datetime] = {}
        self.duplicates_dropped = 0
        self.duplicate_exposures = 0
        self._exposed: dict[tuple[str, str], str] = {}  # (exp, unit) -> variant

    def append(self, event: Event) -> IngestResult:
        if event.event_id in self._seen_ids:
            self.duplicates_dropped += 1
            return IngestResult(accepted=False, duplicate=True)
        watermark = self._watermarks.get(event.experiment_id)
        late = (
            watermark is not None
            and (watermark - event.timestamp).total_seconds()
            > self.late_window_seconds
        )
        self._seen_ids.add(event.event_id)
        self._events.append(event)
        if late:
            self._late_ids.add(event.event_id)
        if watermark is None or event.timestamp > watermark:
            self._watermarks[event.experiment_id] = event.timestamp
        if event.event_name == EXPOSURE:
            key = (event.experiment_id, event.unit_id)
            if key in self._exposed:
                self.duplicate_exposures += 1
            else:
                self._exposed[key] = event.variant
        return IngestResult(accepted=True, late=late)

    def extend(self, events: list[Event]) -> list[IngestResult]:
        return [self.append(e) for e in events]

    def is_late(self, event: Event) -> bool:
        return event.event_id in self._late_ids

    def events_for(
        self, experiment_id: str, include_late: bool = True
    ) -> list[Event]:
        return [
            e
            for e in self._events
            if e.experiment_id == experiment_id
            and (include_late or e.event_id not in self._late_ids)
        ]

    def all_events(self) -> list[Event]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    @property
    def late_count(self) -> int:
        return len(self._late_ids)

    def exposed_units(self, experiment_id: str) -> dict[str, set[str]]:
        """variant -> set of units, first exposure per unit only."""
        out: dict[str, set[str]] = {}
        for (exp, unit), variant in self._exposed.items():
            if exp == experiment_id:
                out.setdefault(variant, set()).add(unit)
        return out

    # -- JSONL persistence -------------------------------------------------
    def save_jsonl(self, path: str | Path) -> int:
        lines = [json.dumps(e.to_dict()) for e in self._events]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    @classmethod
    def load_jsonl(
        cls, path: str | Path, late_window_seconds: float = 3600.0
    ) -> "EventStore":
        store = cls(late_window_seconds=late_window_seconds)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                store.append(Event.from_dict(json.loads(line)))
        return store
