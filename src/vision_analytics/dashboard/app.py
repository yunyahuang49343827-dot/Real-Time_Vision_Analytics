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
    render_processed_video,
    render_traffic_tables,
    render_video_metadata,
)
from vision_analytics.dashboard.formatting import (  # noqa: E402
    event_interpretation,
    event_display_label,
    normalize_progress,
    status_label,
    visualization_artifact_key,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "dashboard.yaml"
SESSION_DEFAULTS = {
    "job_id": None,
    "job_status": None,
    "last_uploaded_file_identity": None,
    "results": None,
    "events": [],
    "visualization_videos": {},
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
    videos = dict(st.session_state.visualization_videos)
    for mode in ("追蹤／移動軌跡", "交通活動熱圖"):
        browser_key = visualization_artifact_key(references, mode)
        if not browser_key or browser_key in videos:
            continue
        try:
            videos[browser_key] = client.get_artifact(job_id, browser_key)
        except DashboardApiError as error:
            st.warning(f"瀏覽器相容的分析影片無法取得：{error.message}")
    st.session_state.visualization_videos = videos
    tables = dict(st.session_state.analytics_tables)
    for key in ("class_distribution_csv", "direction_distribution_csv", "traffic_over_time_csv"):
        if key in tables or not references.get(key):
            continue
        try:
            tables[key] = pd.read_csv(io.BytesIO(client.get_artifact(job_id, key)))
        except (DashboardApiError, ValueError, pd.errors.ParserError) as error:
            message = error.message if isinstance(error, DashboardApiError) else "交通分析檔案格式錯誤"
            st.warning(f"{key}: {message}")
    st.session_state.analytics_tables = tables


def main() -> None:
    config = load_dashboard_config(CONFIG_PATH)
    st.set_page_config(page_title=config.page_title, page_icon="🎥", layout="wide")
    client = _client(config)
    _initialize_state()
    render_header()

    controls = st.columns([1, 5])
    if controls[0].button("新增分析", use_container_width=True):
        _reset(); st.rerun()

    backend_available = False
    try:
        health = client.health()
        backend_available = health.status == "ok"
        controls[1].success(f"後端已連線 · {health.runtime_model}")
    except DashboardApiError:
        controls[1].error("後端無法連線，請先啟動 FastAPI 再送出分析。")

    st.subheader("上傳交通影片")
    uploaded = st.file_uploader(
        "選擇 MP4、MOV 或 AVI 影片",
        type=list(config.supported_extensions),
        key=f"traffic_video_{st.session_state.uploader_generation}",
    )
    content = None
    identity = None
    if uploaded is not None:
        content = uploaded.getvalue()
        identity = f"{uploaded.name}:{len(content)}:{hashlib.sha256(content).hexdigest()}"
        st.write({"檔名": uploaded.name, "檔案大小（bytes）": len(content)})

    active = st.session_state.job_status in {"CREATED", "PROCESSING"}
    analyze = st.button(
        "開始分析", type="primary", disabled=not backend_available or uploaded is None or active,
    )
    if analyze and uploaded is not None and content is not None:
        if active and identity == st.session_state.last_uploaded_file_identity:
            st.warning("這個上傳檔案已有進行中的分析工作。")
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
                st.session_state.visualization_videos = {}
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
            st.subheader("分析狀態")
            st.write(f"工作 `{job_id}` · {status_label(job.status.value)}")
            st.progress(progress / 100, text=f"{progress}%")
            if job.status.value == "FAILED":
                detail = job.error
                st.error(
                    f"{detail.message if detail else '分析失敗'} "
                    f"({detail.code if detail else 'JOB_FAILED'})"
                )
            elif job.status.value == "COMPLETED":
                _load_completed_job(client, job_id)
        except DashboardApiError as error:
            _friendly_error(error)

    if st.session_state.job_status == "COMPLETED" and st.session_state.results:
        result = st.session_state.results
        render_overview(result, status_label("COMPLETED"))
        render_video_metadata(result["video_metadata"])
        _load_result_artifacts(client, str(job_id))

        st.subheader("分析結果影片")
        mode = st.radio(
            "視覺化模式", ("追蹤／移動軌跡", "交通活動熱圖"), horizontal=True,
        )
        artifact_key = visualization_artifact_key(result.get("artifacts", {}), mode)
        video = st.session_state.visualization_videos.get(artifact_key) if artifact_key else None
        if video:
            render_processed_video(video)
        else:
            st.warning("瀏覽器相容格式的分析影片目前無法使用。")

        render_traffic_tables(st.session_state.analytics_tables)
        events = st.session_state.events
        render_events(events)
        evidence_events = [event for event in events if event.get("evidence_path")]
        st.subheader("事件證據")
        if not evidence_events:
            st.info("此分析沒有可用的事件證據圖片。")
        else:
            by_id = {str(event["event_id"]): event for event in evidence_events}
            selected_id = st.selectbox(
                "選擇事件", options=list(by_id),
                format_func=lambda value: (
                    f"{value} · {event_display_label(str(by_id[value]['event_type']))} · "
                    f"{by_id[value]['severity']}"
                ),
            )
            selected = by_id[selected_id]
            st.caption(event_interpretation(
                str(selected["event_type"]), str(selected["status"]),
            ))
            st.write({
                "事件類型": event_display_label(str(selected["event_type"])),
                "嚴重度": selected["severity"],
                "影格": selected["frame_index"],
                "時間（秒）": selected["timestamp_seconds"],
                "主要 Track ID": selected.get("track_id"),
            })
            cache = dict(st.session_state.evidence_cache)
            if selected_id not in cache:
                try:
                    cache[selected_id] = client.get_evidence(str(job_id), selected_id)
                    st.session_state.evidence_cache = cache
                except DashboardApiError as error:
                    st.warning(f"事件證據無法取得：{error.message}")
            if cache.get(selected_id):
                st.image(cache[selected_id], caption=selected_id, use_container_width=True)

    if backend_available and st.session_state.job_status in {"CREATED", "PROCESSING"}:
        time.sleep(config.poll_interval_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
