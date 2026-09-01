"""Small Streamlit rendering components with no analytics computation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st

from .formatting import COUNTING_WORDING, event_table_rows, format_timestamp

RESPONSIVE_VIDEO_CSS = """
<style>
div[data-testid="stVideo"] {
    max-width: 100%;
    overflow-x: hidden;
}
div[data-testid="stVideo"] video {
    display: block;
    width: 100%;
    max-width: 100%;
    height: auto;
    object-fit: contain;
}
</style>
"""


def render_header() -> None:
    st.title("即時視覺分析與事件偵測系統")
    st.caption("以 FastAPI 工作流程執行交通影片分析，視覺化介面由 Streamlit 提供。")


def render_overview(result: Mapping[str, object], status: str) -> None:
    traffic = result.get("traffic_analytics", {})
    events = result.get("event_summary", [])
    total_events = sum(int(item.get("count", 0)) for item in events)
    peak_start = traffic.get("peak_interval_start_seconds")
    peak_end = traffic.get("peak_interval_end_seconds")
    peak_label = (
        f"{format_timestamp(peak_start)}–{format_timestamp(peak_end)}"
        if peak_start is not None and peak_end is not None else "無資料"
    )
    columns = st.columns(4)
    columns[0].metric("通過計數線", int(traffic.get("total_line_crossing_count", 0)))
    columns[1].metric("規則事件", total_events)
    columns[2].metric("尖峰區間", peak_label)
    columns[3].metric("分析狀態", status)
    st.caption(COUNTING_WORDING)


def render_video_metadata(metadata: Mapping[str, object]) -> None:
    st.subheader("影片資訊")
    st.dataframe(pd.DataFrame([metadata]), use_container_width=True, hide_index=True)


def render_processed_video(data: bytes) -> None:
    """Render processed video in a bounded, responsive center column."""
    st.markdown(RESPONSIVE_VIDEO_CSS, unsafe_allow_html=True)
    _, center, _ = st.columns([1, 8, 1], gap="small")
    with center:
        st.video(data, width="stretch")


def render_traffic_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    st.subheader("交通分析")
    labels = {
        "class_distribution_csv": "類別分布",
        "direction_distribution_csv": "方向分布",
        "traffic_over_time_csv": "交通量時間序列",
    }
    for key, title in labels.items():
        frame = tables.get(key)
        if frame is None or frame.empty:
            continue
        st.markdown(f"**{title}**")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if key == "class_distribution_csv" and {"class_name", "crossing_count"} <= set(frame.columns):
            st.bar_chart(frame.set_index("class_name")["crossing_count"])
        if key == "direction_distribution_csv" and {"crossing_direction", "crossing_count"} <= set(frame.columns):
            st.bar_chart(frame.set_index("crossing_direction")["crossing_count"])


def render_events(events: Sequence[Mapping[str, object]]) -> None:
    st.subheader("事件檢視")
    if not events:
        st.info("此分析沒有規則產生的事件。")
        return
    st.dataframe(pd.DataFrame(event_table_rows(events)), use_container_width=True, hide_index=True)
