from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.events.engine import EventEngine, EventPolicy, IntrusionRule
from vision_analytics.events.schema import EVENT_TYPES, EventRecord
from vision_analytics.spatial.direction import WrongWayRecord
from vision_analytics.spatial.dwell import LongDwellRecord, StationaryVehicleRecord
from vision_analytics.spatial.line_crossing import CrossingRecord
from vision_analytics.spatial.proximity import ProximityRecord
from vision_analytics.spatial.zone import ZoneObservation


def policy() -> EventPolicy:
    severity = {event_type: "INFO" for event_type in EVENT_TYPES}
    severity.update({"WRONG_WAY": "CRITICAL", "LONG_DWELL": "WARNING", "PROXIMITY_WARNING": "WARNING"})
    status = {event_type: "DETECTED" for event_type in EVENT_TYPES}
    status.update({"WRONG_WAY": "REVIEW_REQUIRED", "PROXIMITY_WARNING": "REVIEW_REQUIRED"})
    return EventPolicy(
        severity=severity,
        status=status,
        intrusion_rules={"restricted": IntrusionRule("restricted", "CRITICAL")},
    )


def zone(transition: str, *, class_name: str = "car", zone_id: str = "zone") -> ZoneObservation:
    return ZoneObservation(
        video_id="video", source_id="source", zone_id=zone_id,
        frame_index=10, timestamp_seconds=1.0, track_id=1,
        class_name=class_name, transition=transition, center_x=10, center_y=10,
    )


def test_all_existing_rules_normalize_to_expected_event_types() -> None:
    engine = EventEngine(policy())
    engine.normalize_line_crossings([CrossingRecord("video", "source", "line", 1, 0.1, 1, "car", "A_TO_B", 5, 5)])
    engine.normalize_zone_transitions([zone("ENTER"), zone("EXIT")])
    engine.normalize_wrong_way([WrongWayRecord("video", "source", "zone", 2, 0.2, 1, "car", "UP", ("DOWN",), 8, 30)])
    engine.normalize_long_dwell([LongDwellRecord("video", "source", "zone", 1, "car", 0, 5, 5, 5)], frame_index=50)
    engine.normalize_stationary_vehicles([StationaryVehicleRecord("video", "source", "zone", 1, "car", 0, 5, 5, 0.001, 0.003)], frame_index=50)
    engine.normalize_proximity([ProximityRecord("video", "source", "zone", 60, 6, 2, 3, "car", 0.01, 0.012, 8)])
    assert {item.event_type for item in engine.records} == {
        "LINE_CROSSING", "ZONE_ENTRY", "ZONE_EXIT", "WRONG_WAY",
        "LONG_DWELL", "STATIONARY_VEHICLE", "PROXIMITY_WARNING",
    }


def test_severity_status_and_rule_source_are_policy_driven() -> None:
    event = EventEngine(policy()).normalize_wrong_way([
        WrongWayRecord("video", "source", "zone", 2, 0.2, 1, "car", "UP", ("DOWN",), 8, 30)
    ])[0]
    assert (event.severity, event.status) == ("CRITICAL", "REVIEW_REQUIRED")
    assert event.rule_source == "spatial.direction"


def test_unique_readable_event_ids_and_optional_fields() -> None:
    engine = EventEngine(policy())
    events = engine.normalize_zone_transitions([zone("ENTER"), zone("EXIT")])
    assert events[0].event_id == "video-EVT-000001"
    assert len({item.event_id for item in events}) == 2
    assert events[0].secondary_track_id is None
    assert events[0].line_id is None
    row = events[0].to_row()
    assert row["secondary_track_id"] == "" and row["line_id"] == ""


def test_pedestrian_restricted_zone_creates_intrusion_with_zone_severity() -> None:
    events = EventEngine(policy()).normalize_zone_transitions([
        zone("ENTER", class_name="person", zone_id="restricted")
    ])
    assert [item.event_type for item in events] == ["ZONE_ENTRY", "PEDESTRIAN_INTRUSION"]
    assert events[1].severity == "CRITICAL"
    assert events[1].rule_source == "events.pedestrian_intrusion"


def test_non_person_and_unrestricted_zone_do_not_create_intrusion() -> None:
    engine = EventEngine(policy())
    car_events = engine.normalize_zone_transitions([zone("ENTER", zone_id="restricted")])
    person_events = engine.normalize_zone_transitions([zone("ENTER", class_name="person")])
    assert [item.event_type for item in car_events + person_events] == ["ZONE_ENTRY", "ZONE_ENTRY"]


def test_upstream_record_is_not_normalized_twice() -> None:
    engine = EventEngine(policy())
    crossing = CrossingRecord("video", "source", "line", 1, 0.1, 1, "car", "A_TO_B", 5, 5)
    assert len(engine.normalize_line_crossings([crossing])) == 1
    assert engine.normalize_line_crossings([crossing]) == []
    assert len(engine.records) == 1


def test_schema_supports_confirmed_status() -> None:
    event = EventRecord(
        event_id="id", video_id="video", source_id="source",
        event_type="ZONE_ENTRY", frame_index=0, timestamp_seconds=0,
        severity="INFO", status="CONFIRMED", rule_source="manual.review",
    )
    assert event.status == "CONFIRMED"


def test_stationary_threshold_trace_uses_configured_duration() -> None:
    event = EventEngine(policy()).normalize_stationary_vehicles(
        [StationaryVehicleRecord(
            "video", "source", "zone", 1, "car", 0, 6, 6, 0.001, 0.003,
        )],
        frame_index=60,
        duration_thresholds={"zone": 5.0},
    )[0]
    assert event.threshold == "movement<=0.003;duration>=5s"
