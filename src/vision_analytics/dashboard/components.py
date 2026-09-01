"""Small Streamlit rendering components with no analytics computation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st

from .formatting import COUNTING_WORDING, event_table_rows, format_timestamp


def render_header() -> None:
    st.title("Real-Time Vision Analytics")
    st.caption("Job-based traffic video analysis powered by the Stage 20 FastAPI backend.")


def render_overview(result: Mapping[str, object], status: str) -> None:
    traffic = result.get("traffic_analytics", {})
    events = result.get("event_summary", [])
    total_events = sum(int(item.get("count", 0)) for item in events)
    peak_start = traffic.get("peak_interval_start_seconds")
    peak_end = traffic.get("peak_interval_end_seconds")
    peak_label = (
        f"{format_timestamp(peak_start)}–{format_timestamp(peak_end)}"
        if peak_start is not None and peak_end is not None else "Not available"
    )
    columns = st.columns(4)
    columns[0].metric("Line Crossings", int(traffic.get("total_line_crossing_count", 0)))
    columns[1].metric("Rule Events", total_events)
    columns[2].metric("Peak Interval", peak_label)
    columns[3].metric("Status", status)
    st.caption(COUNTING_WORDING)


def render_video_metadata(metadata: Mapping[str, object]) -> None:
    st.subheader("Video Metadata")
    st.dataframe(pd.DataFrame([metadata]), use_container_width=True, hide_index=True)


def render_traffic_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    st.subheader("Traffic Analytics")
    labels = {
        "class_distribution_csv": "Class Distribution",
        "direction_distribution_csv": "Direction Distribution",
        "traffic_over_time_csv": "Traffic Over Time",
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
    st.subheader("Event Review")
    if not events:
        st.info("No rule-generated events were returned for this job.")
        return
    st.dataframe(pd.DataFrame(event_table_rows(events)), use_container_width=True, hide_index=True)
