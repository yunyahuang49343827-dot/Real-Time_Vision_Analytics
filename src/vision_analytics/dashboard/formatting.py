"""Pure display formatting and governed interpretation wording."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

SUPPORTED_STATUSES = frozenset({"CREATED", "PROCESSING", "COMPLETED", "FAILED"})
PROXIMITY_WORDING = (
    "接近警示僅代表影像座標中的接近關係，供人工檢視，"
    "不代表實際距離、碰撞風險或事故機率。"
)
WRONG_WAY_REVIEW_WORDING = "逆向事件為規則判定候選，需要人工確認，不代表已確認交通違規。"
COUNTING_WORDING = "通過計數線為虛擬線的 crossing count，不代表完整交通流量普查。"

EVENT_LABELS = {
    "LINE_CROSSING": "LINE_CROSSING｜通過計數線",
    "ZONE_ENTRY": "ZONE_ENTRY｜進入區域",
    "ZONE_EXIT": "ZONE_EXIT｜離開區域",
    "WRONG_WAY": "WRONG_WAY｜逆向候選",
    "LONG_DWELL": "LONG_DWELL｜長時間停留",
    "STATIONARY_VEHICLE": "STATIONARY_VEHICLE｜靜止車輛",
    "PEDESTRIAN_INTRUSION": "PEDESTRIAN_INTRUSION｜行人進入監控區",
    "PROXIMITY_WARNING": "PROXIMITY_WARNING｜接近警示",
}

VISUALIZATION_MODES = {
    "追蹤／移動軌跡": "tracking_browser_video",
    "交通活動熱圖": "heatmap_browser_video",
}


def normalize_progress(value: object) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number):
        return 0
    return round(min(1.0, max(0.0, number)) * 100)


def format_timestamp(seconds: object) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value) or value < 0:
        return "—"
    minutes, remainder = divmod(value, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def status_label(status: str) -> str:
    if status not in SUPPORTED_STATUSES:
        return "UNKNOWN｜未知"
    return {
        "CREATED": "CREATED｜已建立",
        "PROCESSING": "PROCESSING｜分析中",
        "COMPLETED": "COMPLETED｜已完成",
        "FAILED": "FAILED｜失敗",
    }[status]


def preferred_browser_artifact_key(references: Mapping[str, object]) -> str | None:
    """Never fall back to the known browser-incompatible raw OpenCV artifact."""
    return "processed_browser_video" if references.get("processed_browser_video") else None


def visualization_artifact_key(
    references: Mapping[str, object], mode_label: str,
) -> str | None:
    """Resolve an already-produced browser artifact; never use a raw-video fallback."""
    key = VISUALIZATION_MODES.get(mode_label)
    return key if key and references.get(key) else None


def event_display_label(event_type: str) -> str:
    return EVENT_LABELS.get(event_type, event_type)


def event_interpretation(event_type: str, status: str) -> str:
    if event_type == "PROXIMITY_WARNING":
        return PROXIMITY_WORDING
    if event_type == "WRONG_WAY" and status == "REVIEW_REQUIRED":
        return WRONG_WAY_REVIEW_WORDING
    return "規則產生的系統事件，投入作業使用前請先人工檢視情境。"


def event_table_rows(events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [{
        "event_id": event.get("event_id"),
        "事件 ID": event.get("event_id"),
        "事件類型": event_display_label(str(event.get("event_type", ""))),
        "嚴重度": event.get("severity"),
        "狀態": event.get("status"),
        "影格": event.get("frame_index"),
        "時間": format_timestamp(event.get("timestamp_seconds")),
        "主要 Track ID": event.get("track_id"),
        "說明": event_interpretation(
            str(event.get("event_type", "")), str(event.get("status", "")),
        ),
    } for event in events]
