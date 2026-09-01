"""Streamlit presentation layer for the Stage 20 job API."""

from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.dashboard.api_client import (  # noqa: E402
    DashboardApiError,
    VisionAnalyticsApiClient,
    load_dashboard_config,
)
from vision_analytics.dashboard.components import (  # noqa: E402
    render_events,
    render_header,
    render_overview,
    render_traffic_tables,
    render_video_metadata,
)
from vision_analytics.dashboard.formatting import (  # noqa: E402
    event_interpretation,
    normalize_progress,
    status_label,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "dashboard.yaml"
SESSION_DEFAULTS = {
    "job_id": None,
    "job_status": None,
    "last_uploaded_file_identity": None,
    "results": None,
    "events": [],
    "processed_video": None,
    "analytics_tables": {},
    "evidence_cache": {},
    "uploader_generation": 0,
}


@st.cache_resource
def _client(config):
    return VisionAnalyticsApiClient(config)


def _initialize_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value


def _reset() -> None:
    generation = int(st.session_state.get("uploader_generation", 0)) + 1
    for key, value in SESSION_DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value
    st.session_state.uploader_generation = generation


def _friendly_error(error: DashboardApiError) -> None:
    st.error(f"{error.message} ({error.code})")


def _load_completed_job(client: VisionAnalyticsApiClient, job_id: str) -> None:
    if st.session_state.results is None:
        st.session_state.results = client.get_results(job_id).model_dump(mode="json")
    if not st.session_state.events:
        st.session_state.events = [item.model_dump(mode="json") for item in client.get_events(job_id)]


def _load_result_artifacts(client: VisionAnalyticsApiClient, job_id: str) -> None:
    result = st.session_state.results or {}
    references = result.get("artifacts", {})
    if st.session_state.processed_video is None and references.get("processed_video"):
        try:
            st.session_state.processed_video = client.get_artifact(job_id, "processed_video")
        except DashboardApiError as error:
            st.warning(f"Processed video unavailable: {error.message}")
    tables = dict(st.session_state.analytics_tables)
    for key in ("class_distribution_csv", "direction_distribution_csv", "traffic_over_time_csv"):
        if key in tables or not references.get(key):
            continue
        try:
            tables[key] = pd.read_csv(io.BytesIO(client.get_artifact(job_id, key)))
        except (DashboardApiError, ValueError, pd.errors.ParserError) as error:
            message = error.message if isinstance(error, DashboardApiError) else "Malformed analytics artifact"
            st.warning(f"{key}: {message}")
    st.session_state.analytics_tables = tables


def main() -> None:
    config = load_dashboard_config(CONFIG_PATH)
    st.set_page_config(page_title=config.page_title, page_icon="🎥", layout="wide")
    client = _client(config)
    _initialize_state()
    render_header()

    controls = st.columns([1, 5])
    if controls[0].button("New Analysis", use_container_width=True):
        _reset(); st.rerun()

    backend_available = False
    try:
        health = client.health()
        backend_available = health.status == "ok"
        controls[1].success(f"Backend connected · {health.runtime_model}")
    except DashboardApiError:
        controls[1].error("Backend unavailable. Start FastAPI before submitting analysis.")

    st.subheader("Upload Traffic Video")
    uploaded = st.file_uploader(
        "Choose an MP4, MOV, or AVI video",
        type=list(config.supported_extensions),
        key=f"traffic_video_{st.session_state.uploader_generation}",
    )
    content = None
    identity = None
    if uploaded is not None:
        content = uploaded.getvalue()
        identity = f"{uploaded.name}:{len(content)}:{hashlib.sha256(content).hexdigest()}"
        st.write({"filename": uploaded.name, "size_bytes": len(content)})

    active = st.session_state.job_status in {"CREATED", "PROCESSING"}
    analyze = st.button(
        "Analyze Video", type="primary", disabled=not backend_available or uploaded is None or active,
    )
    if analyze and uploaded is not None and content is not None:
        if active and identity == st.session_state.last_uploaded_file_identity:
            st.warning("This upload already has an active job.")
        else:
            try:
                created = client.create_job(
                    filename=uploaded.name, content=content,
                    content_type=uploaded.type or "application/octet-stream",
                )
                st.session_state.job_id = created.job_id
                st.session_state.job_status = created.status.value
                st.session_state.last_uploaded_file_identity = identity
                st.session_state.results = None
                st.session_state.events = []
                st.session_state.processed_video = None
                st.session_state.analytics_tables = {}
                st.session_state.evidence_cache = {}
            except DashboardApiError as error:
                _friendly_error(error)

    job_id = st.session_state.job_id
    if job_id:
        try:
            job = client.get_job(job_id)
            st.session_state.job_status = job.status.value
            progress = normalize_progress(job.progress)
            st.subheader("Analysis Status")
            st.write(f"Job `{job_id}` · {status_label(job.status.value)}")
            st.progress(progress / 100, text=f"{progress}%")
            if job.status.value == "FAILED":
                detail = job.error
                st.error(
                    f"{detail.message if detail else 'Analysis failed'} "
                    f"({detail.code if detail else 'JOB_FAILED'})"
                )
            elif job.status.value == "COMPLETED":
                _load_completed_job(client, job_id)
        except DashboardApiError as error:
            _friendly_error(error)

    if st.session_state.job_status == "COMPLETED" and st.session_state.results:
        result = st.session_state.results
        render_overview(result, "COMPLETED")
        render_video_metadata(result["video_metadata"])
        _load_result_artifacts(client, str(job_id))

        st.subheader("Processed Video")
        if st.session_state.processed_video:
            st.video(st.session_state.processed_video)
        else:
            st.warning("Processed video is unavailable for this completed job.")

        render_traffic_tables(st.session_state.analytics_tables)
        events = st.session_state.events
        render_events(events)
        evidence_events = [event for event in events if event.get("evidence_path")]
        st.subheader("Evidence Review")
        if not evidence_events:
            st.info("No evidence snapshots are available for this job.")
        else:
            by_id = {str(event["event_id"]): event for event in evidence_events}
            selected_id = st.selectbox(
                "Select an event", options=list(by_id),
                format_func=lambda value: (
                    f"{value} · {by_id[value]['event_type']} · {by_id[value]['severity']}"
                ),
            )
            selected = by_id[selected_id]
            st.caption(event_interpretation(
                str(selected["event_type"]), str(selected["status"]),
            ))
            st.write({
                "event_type": selected["event_type"], "severity": selected["severity"],
                "frame_id": selected["frame_index"],
                "timestamp_seconds": selected["timestamp_seconds"],
                "primary_track_id": selected.get("track_id"),
            })
            cache = dict(st.session_state.evidence_cache)
            if selected_id not in cache:
                try:
                    cache[selected_id] = client.get_evidence(str(job_id), selected_id)
                    st.session_state.evidence_cache = cache
                except DashboardApiError as error:
                    st.warning(f"Evidence unavailable: {error.message}")
            if cache.get(selected_id):
                st.image(cache[selected_id], caption=selected_id, use_container_width=True)

    if backend_available and st.session_state.job_status in {"CREATED", "PROCESSING"}:
        time.sleep(config.poll_interval_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
